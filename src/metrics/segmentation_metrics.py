from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryF1Score,
    BinaryFBetaScore,
)


def get_metric_collection() -> MetricCollection:
    return MetricCollection(
        {
            "IoU": BinaryJaccardIndex(),
            "dice": BinaryF1Score(),
            "F2_Score": BinaryFBetaScore(
                beta=2.0
            ),  # Recall is more important than Precision
        }
    )
