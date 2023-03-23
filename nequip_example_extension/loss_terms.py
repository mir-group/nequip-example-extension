from nequip.data import AtomicDataDict


class MyL1Loss:
    """Minimal custom loss function example.

    To use this in a training, set in your YAML file:

        loss_coeffs:
            forces:
                - 0.1
                - !!python/object:nequip_example_extension.loss_terms.MyL1Loss {}

    This funny syntax tells PyYAML to construct an object of this class and put it in the config.
    In this case, the `key` argument to `__call__` will be `"forces"`.

    See `SimpleLoss` (https://github.com/mir-group/nequip/blob/main/nequip/train/_loss.py#L11)
    for the actual implementation of L1/2  loss, which handles some more edge cases.
    """

    def __call__(
        self,
        pred: AtomicDataDict.Type,
        ref: AtomicDataDict.Type,
        key: str,
        mean: bool = True,
    ):
        """
        Args:
            pred: output of the model (can be a batch)
            ref:  training data (can be a batch)
            key:  which key to compute loss on, from the `loss_coeffs` config shown above.
            mean: whether to return a single scalar loss value.  Can basically be ignored (`assert mean`)
                as long as you don't want to use this loss as a metric (if you do, return instead a
                per-(atom/graph/whatever) tensor of losses).
        """
        assert mean
        pred = pred[key]
        ref = ref[key]
        term = (pred - ref).abs().mean()
        return term
