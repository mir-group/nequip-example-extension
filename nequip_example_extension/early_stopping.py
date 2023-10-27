from typing import Tuple, Optional


class MyEarlyStop:
    """Minimal custom early stopping condition.

    To use this in a training, set in your YAML file:

        early_stopping_custom_events:
            - !!python/object:nequip_example_extension.early_stopping.MyEarlyStop {}

    This funny syntax tells PyYAML to construct an object of this class and put it in the config.

    Note:
        Custom conditions that are stateful are NOT presently supported
        and will not work correctly under training restarts. Please file
        an issue if stateful custom early stopping conditions are needed
        for your workflow.
    """

    def __call__(self, metrics_dict: dict) -> Tuple[bool, Optional[str]]:
        """
        Args:
            metrics_dict:  a dict of current metrics like `validation_loss`, etc.
        """
        # As a toy example, stop if the validation loss is less than 0.26
        if metrics_dict["validation_loss"] < 0.26:
            return True, "validation_loss was smaller than 0.26"
        else:
            return False, None
