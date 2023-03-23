import torch


class ExampleLnNormMetric:
    """A metric to report the L-whatever norm of some prediction.

    metrics_components:
      - - node_features
        - mean
        - functional: !!python/object:nequip_example_extension.metrics.ExampleLnNormMetric {"ord": 1}

    Here `node_features` will be `key`.  `mean` indicates the reduction over atoms/graphs/etc. that should be done outside of this class.
    """

    ord: float = 2

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
    ):
        assert not mean  # see loss_terms.py
        pred = pred[key]
        return torch.linalg.vector_norm(pred, ord=self.ord, dim=-1)

    def get_name(self, short_name: str) -> str:
        """Return a shortened name for this metric.

        Args:
            short_name: a (possibly abbreviated) name for the field this metric is applied to.
        """
        return f"{short_name}_L{self.ord}"
