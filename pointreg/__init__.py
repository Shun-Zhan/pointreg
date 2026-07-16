"""部分重叠点云配准工具包（PointReg）。

本包实现斯坦福兔子（Stanford Bunny）多视角扫描的两帧点云配准，
重点解决低重叠（overlap < 0.3）场景。这里只对外暴露最常用的入口：
配准配置、结果数据结构，以及一站式的 ``register_pair`` 配准函数。
"""

from .models import ICPRecord, RegistrationConfig, RegistrationResult
from .pipeline import register_pair

__all__ = ["ICPRecord", "RegistrationConfig", "RegistrationResult", "register_pair"]
