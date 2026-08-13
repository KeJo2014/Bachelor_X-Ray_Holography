from torchmetrics import MetricCollection
from torchmetrics.clustering import AdjustedRandScore, NormalizedMutualInfoScore


def get_metric_collection() -> MetricCollection:
    return MetricCollection(
        {
            "adjusted_rand_score": AdjustedRandScore(),
            "normalized_mutual_info_score": NormalizedMutualInfoScore(),
        }
    )
