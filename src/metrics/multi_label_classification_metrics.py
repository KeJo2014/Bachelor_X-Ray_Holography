from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelF1Score,
    MultilabelHammingDistance,
)


def get_metric_collection(num_classes: int) -> MetricCollection:
    return MetricCollection(
        {
            "mAP": MultilabelAveragePrecision(num_labels=num_classes, average="macro"),
            "f1_macro": MultilabelF1Score(num_labels=num_classes, average="macro"),
            "f1_micro": MultilabelF1Score(num_labels=num_classes, average="micro"),
            "hamming": MultilabelHammingDistance(num_labels=num_classes),
        }
    )
