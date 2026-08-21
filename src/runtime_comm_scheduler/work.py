"""Scheduled Work compatibility boundary."""


class ScheduledWork:
    """Wrapper boundary for a deferred underlying distributed Work object."""

    def wait(self, timeout=None):
        raise NotImplementedError

    def is_completed(self) -> bool:
        raise NotImplementedError

    def get_future(self):
        raise NotImplementedError
