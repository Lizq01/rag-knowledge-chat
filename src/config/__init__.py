"""config 层：所有参数与环境变量的唯一出口。

settings 负责读 .env、算路径常量、设镜像环境变量；这里整体转发，
让上层继续写 `from src.config import XXX` 而不用关心内部分层。
"""
from src.config.settings import *  # noqa: F401,F403
