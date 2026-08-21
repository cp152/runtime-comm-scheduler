"""Megatron adapter boundary.

This file documents the intended integration point; it deliberately contains
no Megatron imports or runtime behavior yet.
"""


class MegatronGradientSyncAdapter:
    """Future adapter for start_grad_sync/finish_grad_sync."""

    def start_grad_sync(self, bucket):
        raise NotImplementedError

    def finish_grad_sync(self, bucket):
        raise NotImplementedError
