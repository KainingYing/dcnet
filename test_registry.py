import mmcv
# 0. 先构建一个全局的 CATS 注册器类
CATS = mmcv.Registry('cat')

# 通过装饰器方式作用在想要加入注册器的具体类中
#===============================================================
# 1. 不需要传入任何参数，此时默认实例化的配置字符串是 str (类名)
@CATS.register_module(force=True)
class BritishShorthair:
    pass
# 类实例化
CATS.get('BritishShorthair')(**args)

#==============================================================
# 2.传入指定 str，实例化时候只需要传入对应相同 str 即可
@CATS.register_module(name='Siamese')
class SiameseCat:
    pass
# 类实例化
CATS.get('Siamese')(**args)

#===============================================================
# 3.如果出现同名 Registry Key，可以选择报错或者强制覆盖
# 如果指定了 force=True，那么不会报错
# 此时 Registry 的 Key 中，Siamese2Cat 类会覆盖 SiameseCat 类
# 否则会报错
@CATS.register_module(name='Siamese',force=True)
class Siamese2Cat:
    pass
# 类实例化
CATS.get('Siamese')(**args)

#==============================================================
# 4. 可以直接注册类
class Munchkin:
    pass
CATS.register_module(Munchkin)

# 类实例化
CATS.get('Munchkin')(**args)