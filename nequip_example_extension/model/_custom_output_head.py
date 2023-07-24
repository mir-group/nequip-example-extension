from typing import Optional
from nequip.nn import SequentialGraphNetwork, AtomwiseLinear, GraphModuleMixin
from nequip.model._scaling import GlobalRescale
from nequip.data import register_fields, AtomicDataset, AtomicDataDict

# We have to tell `nequip` what kind of field `my_custom_field` is
register_fields(node_fields=["my_custom_field"])


def CustomOutputHead(config, model: SequentialGraphNetwork) -> SequentialGraphNetwork:
    """Model builder that adds an extra linear output head to a NequIP model.

    See configs/minimal_custom_field.yaml for how to use.
    """
    model.insert_from_parameters(
        # see nequip/model/_eng.py for the names of all modules in a NequIP model
        # we put it after the 2nd to last linear projection into the smaller node features
        after="conv_to_output_hidden",
        # name for our new module
        name="custom_output_head",
        # hardcoded parameters from the builder
        # we want in this case a 1 scalar prediction (1x0e) in the field
        params=dict(irreps_out="1x0e", out_field="my_custom_field"),
        # config from which to pull other parameters
        shared_params=config,
        # the module to add:
        builder=AtomwiseLinear,
    )
    return model


# If your modifications are more extensive, you can also copy and update the full model builder from `nequip/model/_eng.py`, for example.
# This can be pariticularly helpful if you are _replacing_, rather than adding, modules.

# Scaling is also extremely important for reliable predictions.
# This new model builder will apply a global rescaling to the custom field:

# !!! Note that global shifting may not make any sense for your property, in which case you should consider no shifting, or per-atom shifting. !!!


def RescaleCustomField(
    model: GraphModuleMixin,
    config,
    initialize: bool,
    dataset: Optional[AtomicDataset] = None,
):
    return GlobalRescale(
        model=model,
        config=config,
        dataset=dataset,
        initialize=initialize,
        module_prefix="global_rescale_custom_field",
        default_scale="dataset_my_custom_field_std",  # <-- default to computing the training dataset RMS of the custom field as the scaling factor
        default_shift="dataset_my_custom_field_mean",  # similar.  !!! Think carefully about what is right for your application! !!!
        default_scale_keys=["my_custom_field"],
        default_shift_keys=[],
    )
