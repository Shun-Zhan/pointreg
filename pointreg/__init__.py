"""部分重合点云配准工具包（pointreg）。

对外暴露最常用的入口：配置/结果数据类，以及两种配准入口——
register_pair（直接两帧配准）与 register_dataset_pair（桥接法）。
"""

from .models import ICPRecord, RegistrationConfig, RegistrationResult
from .pipeline import register_pair
from .dataset import register_dataset_pair

# 定义 `from pointreg import *` 时导出的公共名字
__all__ = ["ICPRecord", "RegistrationConfig", "RegistrationResult", "register_pair", "register_dataset_pair"]
