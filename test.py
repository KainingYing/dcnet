from mmcv.utils import Registry

CONVERTERS = Registry('converter')


@CONVERTERS.register_module()
class Converter1(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b


if __name__ == '__main__':
    converter_cfg = dict(type='Converter1', a=1, b=2)
    converter = CONVERTERS.build(converter_cfg)
