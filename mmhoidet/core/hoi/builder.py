from mmcv.utils import Registry, build_from_cfg

HOIASSIGNERS = Registry('hoi_assigner')
HOISAMPLERS = Registry('hoi_sampler')


def build_assigner(cfg, **default_args):
    """Builder of hoi assigner."""
    return build_from_cfg(cfg, HOIASSIGNERS, default_args)


def build_sampler(cfg, **default_args):
    """Builder of HOI sampler."""
    return build_from_cfg(cfg, HOISAMPLERS, default_args)
