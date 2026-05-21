from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score


def get_metric_collection() -> MetricCollection:
    return MetricCollection({"IoU": BinaryJaccardIndex(), "dice": BinaryF1Score()})
