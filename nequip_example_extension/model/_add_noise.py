from nequip.nn import SequentialGraphNetwork

from nequip_example_extension.nn import AddNoiseModule


def AddNoiseToPairEnergies(
    config, model: SequentialGraphNetwork
) -> SequentialGraphNetwork:
    """Model builder that adds an `AddNoiseModule` to an Allegro model to add noise to the pair energies.

    See configs/minimal_custom_module.yaml for how to use.
    """
    model.insert_from_parameters(
        # see allegro/models/_allegro.py for the names of all modules in an Allegro model
        # `"edge_eng"` is the final readout MLP
        after="edge_eng",
        # name for our new module
        name="add_noise",
        # hardcoded parameters from the builder
        params=dict(field="edge_energy"),
        # config from which to pull other parameters--- this means we can set
        # `noise_sigma` in our YAML config file!
        shared_params=config,
        # the module to add:
        builder=AddNoiseModule,
    )
    return model
