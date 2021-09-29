import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32, BaseModule

from mmhoidet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                           build_assigner, build_sampler, multi_apply,
                           reduce_mean)
from mmhoidet.models.utils import build_transformer
from ..builder import HEADS, build_loss


@HEADS.register_module()
class QPICHead(BaseModule):
    """reimplements the QPIC Transformer head."""

    def __init__(self,
                 num_obj_classes,
                 num_verb_classes,
                 in_channels,
                 subject_category_id=0,
                 num_query=100,
                 num_reg_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 loss_obj_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_verb_cls=dict(
                     type='ElementWiseFocalLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='SmoothL1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         obj_cls_cost=dict(type='ClsSoftmaxCost', weight=1.),
                         verb_cls_cost=dict(type='ClsNoSoftmaxCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                         iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs
                 ):
        super(BaseModule, self).__init__()

        self.num_query = num_query
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.num_obj_classes = num_obj_classes
        self.num_verb_classes = num_verb_classes
        self.subject_category_id = subject_category_id

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_obj_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is QPICHead):
            assert isinstance(class_weight, float), 'Expected ' \
                                                    'class_weight to have type float. Found ' \
                                                    f'{type(class_weight)}.'
            bg_cls_weight = loss_obj_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                                                     'bg_cls_weight to have type float. Found ' \
                                                     f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_obj_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_obj_classes] = bg_cls_weight
            loss_obj_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_obj_cls:
                loss_obj_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        self.loss_obj_cls = build_loss(loss_obj_cls)
        self.loss_verb_cls = build_loss(loss_verb_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided ' \
                                            'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_obj_cls['loss_weight'] == assigner['obj_cls_cost']['weight']
            assert loss_verb_cls['loss_weight'] == assigner['verb_cls_cost']['weight']
            assert loss_bbox['loss_weight'] == assigner['reg_cost']['weight']
            assert loss_iou['loss_weight'] == assigner['iou_cost']['weight']
            self.assigner = build_assigner(assigner)
            # QPIC sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.obj_cls_out_channels = num_obj_classes if self.loss_obj_cls.use_sigmoid else num_obj_classes + 1
        self.verb_cls_out_channels = num_verb_classes if self.loss_verb_cls.use_sigmoid else num_verb_classes + 1
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
                                                 f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
                                                 f' and {num_feats}.'
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the QPIC head."""
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)
        # self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.fc_obj_cls = Linear(self.embed_dims, self.num_obj_classes + 1)
        self.fc_verb_cls = Linear(self.embed_dims, self.num_verb_classes)
        self.reg_ffn_sub = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_reg_fcs,
            self.act_cfg,
            dropout=0.0,
            add_residual=False)
        self.reg_ffn_obj = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_reg_fcs,
            self.act_cfg,
            dropout=0.0,
            add_residual=False)
        self.fc_sub_reg = Linear(self.embed_dims, 4)
        self.fc_obj_reg = Linear(self.embed_dims, 4)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is QPICHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        # super(AnchorFreeHead,
        #       self)._load_from_state_dict(state_dict, prefix, local_metadata,
        #                                   strict, missing_keys,
        #                                   unexpected_keys, error_msgs)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_sub_bboxes,
                      gt_obj_bboxes,
                      gt_obj_labels,
                      gt_verb_labels,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x, img_metas)  # forward

        loss_inputs = outs + (gt_sub_bboxes, gt_obj_bboxes, gt_obj_labels, gt_verb_labels, img_metas)
        return self.loss(*loss_inputs)

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores_list (list[Tensor]): Classification scores \
                    for each scale level. Each is a 4D-tensor with shape \
                    [nb_dec, bs, num_query, cls_out_channels]. Note \
                    `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                    outputs for each scale level. Each is a 4D-tensor with \
                    normalized coordinate format (cx, cy, w, h) and shape \
                    [nb_dec, bs, num_query, 4].
        """
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single, feats, img_metas_list)

    def forward_single(self, x, img_metas):
        """"Forward function for a single feature level.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0

        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        # _ represents memory (output results from encoder, with shape [bs, embed_dims, h, w])
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight,
                                       pos_embed)

        all_sub_bbox_preds = self.fc_sub_reg(self.activate(self.reg_ffn_sub(outs_dec))).sigmoid()
        all_obj_bbox_preds = self.fc_obj_reg(self.activate(self.reg_ffn_obj(outs_dec))).sigmoid()
        all_obj_cls_scores = self.fc_obj_cls(outs_dec)
        all_verb_cls_scores = self.fc_verb_cls(outs_dec)  # multi-label classification
        return all_sub_bbox_preds, all_obj_bbox_preds, all_obj_cls_scores, all_verb_cls_scores

    @force_fp32(apply_to=(
            'all_obj_cls_scores_list', 'all_verb_cls_scores_list', 'all_sub_bbox_preds_list',
            'all_obj_bbox_preds_list'))
    def loss(self,
             all_sub_bbox_preds_list,
             all_obj_bbox_preds_list,
             all_obj_cls_scores_list,
             all_verb_cls_scores_list,
             gt_sub_bboxes_list,
             gt_obj_bboxes_list,
             gt_obj_labels_list,
             gt_verb_labels_list,
             img_metas):
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # NOTE defaultly only the outputs from the last feature scale is used.
        all_obj_cls_scores = all_obj_cls_scores_list[-1]
        all_verb_cls_scores = all_verb_cls_scores_list[-1]
        all_sub_bbox_preds = all_sub_bbox_preds_list[-1]
        all_obj_bbox_preds = all_obj_bbox_preds_list[-1]

        num_dec_layers = len(all_obj_cls_scores)
        all_gt_sub_bboxes_list = [gt_sub_bboxes_list for _ in range(num_dec_layers)]
        all_gt_obj_boxes_list = [gt_obj_bboxes_list for _ in range(num_dec_layers)]
        all_gt_obj_labels_list = [gt_obj_labels_list for _ in range(num_dec_layers)]
        all_gt_verb_labels_list = [gt_verb_labels_list for _ in range(num_dec_layers)]

        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_bbox, losses_iou, losses_obj_cls, losses_verb_cls = multi_apply(
            self.loss_single, all_sub_bbox_preds, all_obj_bbox_preds, all_obj_cls_scores, all_verb_cls_scores,
            all_gt_sub_bboxes_list, all_gt_obj_boxes_list, all_gt_obj_labels_list, all_gt_verb_labels_list,
            img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_obj_cls'] = losses_obj_cls[-1]
        loss_dict['loss_verb_cls'] = losses_verb_cls[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_bbox_i, loss_iou_i, loss_obj_cls_i, loss_verb_cls_i in zip(losses_bbox[:-1],
                                                                            losses_iou[:-1],
                                                                            losses_obj_cls[:-1],
                                                                            losses_verb_cls[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_obj_cls'] = loss_obj_cls_i
            loss_dict[f'd{num_dec_layer}.loss_verb_cls'] = loss_verb_cls_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    sub_bbox_preds,
                    obj_bbox_preds,
                    obj_cls_scores,
                    verb_cls_scores,
                    gt_sub_bboxes_list,
                    gt_obj_bboxes_list,
                    gt_obj_labels_list,
                    gt_verb_labels_list,
                    img_metas):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = obj_cls_scores.size(0)
        obj_cls_scores_list = [obj_cls_scores[i] for i in range(num_imgs)]
        verb_cls_scores_list = [verb_cls_scores[i] for i in range(num_imgs)]
        sub_bbox_preds_list = [sub_bbox_preds[i] for i in range(num_imgs)]
        obj_bbox_preds_list = [obj_bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(sub_bbox_preds_list,
                                           obj_bbox_preds_list,
                                           obj_cls_scores_list, verb_cls_scores_list,
                                           gt_sub_bboxes_list, gt_obj_bboxes_list, gt_obj_labels_list,
                                           gt_verb_labels_list,
                                           img_metas)
        (sub_bbox_targets_list, sub_bbox_weights_list, obj_bbox_targets_list, obj_bbox_weights_list, obj_labels_list,
         obj_label_weights_list, verb_labels_list, verb_label_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        # obj/verb labels
        obj_labels = torch.cat(obj_labels_list, 0)
        verb_labels = torch.cat(verb_labels_list, 0).long()  # targets should be long in Focal Loss
        obj_label_weights = torch.cat(obj_label_weights_list, 0)  # this
        verb_label_weights = torch.cat(verb_label_weights_list, 0)
        # sub/obj bbox
        sub_bbox_targets = torch.cat(sub_bbox_targets_list, 0)
        sub_bbox_weights = torch.cat(sub_bbox_weights_list, 0)
        obj_bbox_targets = torch.cat(obj_bbox_targets_list, 0)
        obj_bbox_weights = torch.cat(obj_bbox_weights_list, 0)

        # object classification loss
        obj_cls_scores = obj_cls_scores.reshape(-1, self.obj_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                obj_cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_obj_cls = self.loss_obj_cls(
            obj_cls_scores, obj_labels, obj_label_weights, avg_factor=cls_avg_factor)  # CEL

        # verb classification (multi-label classification) loss
        verb_cls_scores = verb_cls_scores.reshape(-1, self.verb_cls_out_channels)  # without softmax
        loss_verb_cls = self.loss_verb_cls(verb_cls_scores, verb_labels, verb_label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_obj_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, sub_bbox_pred, obj_bbox_pred in zip(img_metas, sub_bbox_preds, obj_bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = sub_bbox_pred.new_tensor([img_w, img_h, img_w,
                                              img_h]).unsqueeze(0).repeat(sub_bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        sub_bbox_preds = sub_bbox_preds.reshape(-1, 4)
        obj_bbox_preds = obj_bbox_preds.reshape(-1, 4)

        sub_bboxes = bbox_cxcywh_to_xyxy(sub_bbox_preds) * factors
        sub_bboxes_gt = bbox_cxcywh_to_xyxy(sub_bbox_targets) * factors

        obj_bboxes = bbox_cxcywh_to_xyxy(obj_bbox_preds) * factors
        obj_bboxes_gt = bbox_cxcywh_to_xyxy(obj_bbox_targets) * factors

        # regression L1 loss
        loss_sub_bbox = self.loss_bbox(sub_bbox_preds, sub_bbox_targets, sub_bbox_weights, avg_factor=num_total_pos)
        loss_obj_bbox = self.loss_bbox(obj_bbox_preds, obj_bbox_targets, obj_bbox_weights, avg_factor=num_total_pos)
        loss_bbox = loss_sub_bbox + loss_obj_bbox

        # regression IoU loss, defaultly GIoU loss
        # loss_iou = self.loss_iou(
        #     sub_bboxes, obj_bboxes, sub_bboxes_gt, obj_bboxes_gt, sub_bbox_weights, avg_factor=num_total_pos)
        loss_sub_iou = self.loss_iou(sub_bboxes, sub_bboxes_gt, sub_bbox_weights, avg_factor=num_total_pos)
        loss_obj_iou = self.loss_iou(obj_bboxes, obj_bboxes_gt, obj_bbox_weights, avg_factor=num_total_pos)
        loss_iou = loss_sub_iou + loss_obj_iou

        return loss_bbox, loss_iou, loss_obj_cls, loss_verb_cls,

    def get_targets(self,
                    sub_bbox_preds_list,
                    obj_bbox_preds_list,
                    obj_cls_scores_list,
                    verb_cls_scores_list,
                    gt_sub_bboxes_list,
                    gt_obj_bboxes_list,
                    gt_obj_labels_list,
                    gt_verb_labels_list,
                    img_metas
                    ):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            sub_bbox_preds_list ():
            obj_bbox_preds_list ():
            obj_cls_scores_list ():
            verb_cls_scores_list ():
            gt_sub_bboxes_list ():
            gt_obj_bboxes_list ():
            gt_obj_labels_list ():
            gt_verb_labels_list ():
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """

        (sub_bbox_targets_list, sub_bbox_weights_list, obj_bbox_targets_list, obj_bbox_weights_list, obj_labels_list,
         obj_label_weights_list, verb_labels_list, verb_label_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, sub_bbox_preds_list, obj_bbox_preds_list, obj_cls_scores_list,
            verb_cls_scores_list,
            gt_sub_bboxes_list, gt_obj_bboxes_list, gt_obj_labels_list, gt_verb_labels_list, img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (sub_bbox_targets_list, sub_bbox_weights_list, obj_bbox_targets_list, obj_bbox_weights_list, obj_labels_list,
                obj_label_weights_list, verb_labels_list, verb_label_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           sub_bbox_pred,
                           obj_bbox_pred,
                           obj_cls_score,
                           verb_cls_score,
                           gt_sub_bboxes,
                           gt_obj_bboxes,
                           gt_obj_labels,
                           gt_verb_labels,
                           img_meta):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = sub_bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(sub_bbox_pred, obj_bbox_pred, obj_cls_score, verb_cls_score, gt_sub_bboxes,
                                             gt_obj_bboxes, gt_obj_labels, gt_verb_labels, img_meta)
        # TODO:finish this
        sampling_result = self.sampler.sample(assign_result, sub_bbox_pred, obj_bbox_pred,
                                              gt_sub_bboxes, gt_obj_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # obj_label and verb_label targets
        obj_labels = gt_sub_bboxes.new_full((num_bboxes,),
                                            self.num_obj_classes,
                                            dtype=torch.long)
        verb_labels = gt_sub_bboxes.new_full((num_bboxes, self.num_verb_classes),
                                             0,
                                             dtype=torch.float32)
        obj_labels[pos_inds] = gt_obj_labels[sampling_result.pos_assigned_gt_inds]
        verb_labels[pos_inds] = gt_verb_labels[sampling_result.pos_assigned_gt_inds]
        obj_label_weights = gt_sub_bboxes.new_ones(num_bboxes)
        verb_label_weights = gt_sub_bboxes.new_ones(num_bboxes)

        # subject/object bbox targets
        sub_bbox_targets = torch.zeros_like(sub_bbox_pred)
        obj_bbox_targets = torch.zeros_like(obj_bbox_pred)
        sub_bbox_weights = torch.zeros_like(sub_bbox_pred)
        sub_bbox_weights[pos_inds] = 1.0
        obj_bbox_weights = torch.zeros_like(obj_bbox_pred)
        obj_bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = sub_bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        pos_gt_sub_bboxes_normalized = sampling_result.pos_gt_sub_bboxes / factor
        pos_gt_obj_bboxes_normalized = sampling_result.pos_gt_obj_bboxes / factor
        pos_gt_sub_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_sub_bboxes_normalized)
        pos_gt_obj_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_obj_bboxes_normalized)
        sub_bbox_targets[pos_inds] = pos_gt_sub_bboxes_targets
        obj_bbox_targets[pos_inds] = pos_gt_obj_bboxes_targets
        return (sub_bbox_targets, sub_bbox_weights, obj_bbox_targets, obj_bbox_weights,
                obj_labels, obj_label_weights, verb_labels, verb_label_weights, pos_inds, neg_inds)

    def simple_test(self,
                    feats,
                    img_metas,
                    rescale=False):
        """Test det hois without test-time augmentation."""
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_hois(*outs, img_metas, rescale=rescale)
        return results_list

    @force_fp32(apply_to=('all_sub_bbox_preds_list', 'all_obj_bbox_preds_list', 'all_obj_cls_scores_list', 'all_verb_cls_scores_list'))
    def get_hois(self,
                   all_sub_bbox_preds_list,
                   all_obj_bbox_preds_list,
                   all_obj_cls_scores_list,
                   all_verb_cls_scores_list,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_verb_cls_scores_list ():
            all_obj_cls_scores_list ():
            all_obj_bbox_preds_list ():
            all_sub_bbox_preds_list ():
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.
        sub_bbox_preds = all_sub_bbox_preds_list[-1][-1]
        obj_bbox_preds = all_obj_bbox_preds_list[-1][-1]
        obj_cls_scores = all_obj_cls_scores_list[-1][-1]
        verb_cls_scores = all_verb_cls_scores_list[-1][-1]

        result_list = []
        for img_id in range(len(img_metas)):
            sub_bbox_pred = sub_bbox_preds[img_id]
            obj_bbox_pred = obj_bbox_preds[img_id]
            obj_cls_score = obj_cls_scores[img_id]
            verb_cls_score = verb_cls_scores[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']  # todo: how to consider the pad
            proposals = self._get_hois_single(sub_bbox_pred, obj_bbox_pred, obj_cls_score, verb_cls_score,
                                              img_shape, scale_factor, rescale)
            result_list.append(proposals)

        return result_list

    def _get_hois_single(self,
                         sub_bbox_pred,
                         obj_bbox_pred,
                         obj_cls_score,
                         verb_cls_score,
                         img_shape,
                         scale_factor,
                         rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(sub_bbox_pred) == len(obj_bbox_pred) == len(obj_cls_score) == len(verb_cls_score)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)

        obj_prob = F.softmax(obj_cls_score, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)  # shaped: [num_query, ] and [num_query, ]

        verb_scores = verb_cls_score.sigmoid()

        sub_bboxes = bbox_cxcywh_to_xyxy(sub_bbox_pred)
        sub_bboxes[:, 0::2] = sub_bboxes[:, 0::2] * img_shape[1]
        sub_bboxes[:, 1::2] = sub_bboxes[:, 1::2] * img_shape[0]
        sub_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        sub_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        obj_bboxes = bbox_cxcywh_to_xyxy(obj_bbox_pred)
        obj_bboxes[:, 0::2] = obj_bboxes[:, 0::2] * img_shape[1]
        obj_bboxes[:, 1::2] = obj_bboxes[:, 1::2] * img_shape[0]
        obj_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        obj_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        if rescale:
            sub_bboxes /= sub_bboxes.new_tensor(scale_factor)
            obj_bboxes /= obj_bboxes.new_tensor(scale_factor)
        # TODO: this line is strange.
        sub_labels = torch.full(obj_labels.shape, self.subject_category_id, dtype=obj_labels.dtype, device=obj_labels.device)
        labels = torch.cat((sub_labels, obj_labels), dim=0)
        bboxes = torch.cat((sub_bboxes, obj_bboxes), dim=0)

        hoi_scores = verb_scores * obj_scores.unsqueeze(1)

        ids = torch.arange(bboxes.shape[0])

        return labels, hoi_scores, bboxes, ids[:ids.shape[0] // 2], ids[ids.shape[0] // 2:]
