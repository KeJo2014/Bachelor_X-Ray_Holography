from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAveragePrecision,
    MulticlassF1Score,
    MulticlassAccuracy,
)


def get_metric_collection(num_classes: int) -> MetricCollection:
    return MetricCollection(
        {
            "mAP": MulticlassAveragePrecision(num_classes=num_classes, average="macro"),
            "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
            "f1_micro": MulticlassF1Score(num_classes=num_classes, average="micro"),
            "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
        }
    )
