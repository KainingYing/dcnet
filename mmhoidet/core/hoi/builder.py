from mmcv.utils import Registry, build_from_cfg

HOI_ASSIGNERS = Registry('hoi_assigner')
HOI_SAMPLERS = Registry('hoi_sampler')


def build_assigner(cfg, **default_args):
    """Builder of hoi assigner."""
    return build_from_cfg(cfg, HOI_ASSIGNERS, default_args)


def build_sampler(cfg, **default_args):
    """Builder of HOI sampler."""
    return build_from_cfg(cfg, HOI_SAMPLERS, default_args)
