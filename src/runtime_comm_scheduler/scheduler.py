"""准入 scheduler 边界；具体实现从 M2 开始。"""

from .intent import CommIntent


class AdmissionScheduler:
    def submit(self, intent: CommIntent):
        """提交 intent；具体的 plan 校验和准入逻辑尚未实现。"""
        raise NotImplementedError

    def close(self) -> None:
        """释放 scheduler 资源。"""
        raise NotImplementedError
