from config import STATUS_LABELS


def describe_order(status: str) -> str:
    return STATUS_LABELS[status]
