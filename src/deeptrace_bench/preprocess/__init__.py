"""Turning raw items into model inputs (audio segments, face crops)."""


class PreprocessError(RuntimeError):
    """An item that can't be turned into model input.

    ``reason`` is a short code (``too_short``, ``no_face``, ``unreadable``) that is recorded
    as the item's status rather than dropping the item. Face detectors fail more often on
    darker skin, so silently skipping failures would bias exactly the group gaps we measure.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
