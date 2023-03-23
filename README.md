# Example extension package for `nequip`

![](./logo.png)

This repository serves as an example and template for writing extension packages for the [`nequip` framework](https://github.com/mir-group/nequip) for (equivariant) machine learning on atomic systems. Extension packages to `nequip` can implement new models, new model components and wrappers that can be mixed with or added to existing models, 

For an interactive tutorial please see the last section [the Allegro tutorial Colab](https://colab.research.google.com/drive/1yq2UwnET4loJYg_Fptt9kpklVaZvoHnq).

## FAQ

**What about LAMMPS?**

If your model modules / additions are fully TorchScript compatible, then your modified/extended models **can be used in the corresponding LAMMPS plugin for MD** ([`pair_nequip`](https://github.com/mir-group/pair_nequip) for the general case, and [`pair_allegro`](https://github.com/mir-group/pair_allegro) for Allegro models.)

**Extension package vs pull request?**

We welcome pull requests and contributions to the [`nequip`](https://github.com/mir-group/nequip) and [`allegro`](https://github.com/mir-group/allegro/) repositories and generally smaller **new features of general interest should be pull requests** on those repositories. If you aren't sure what is most appropriate for your idea, please always feel free to reach out via GitHub Discussions or email at albym[at]seas[dot]harvard[dot]edu.

## Contents

### Model customization
The `nequip` framework, for which Allegro is an extension package, makes it easy to modify or extend models while preserving compatability with all the `nequip-*` tools and LAMMPS pair styles.

To illustrate this, we demonstrate a simple modification to an Allegro model that adds random noise to the predicted pairwise energies before they are summed into per-atom and total energies. To do this, we define two kinds of `nequip` extensions:

 1. A *module*, which includes code we want to put in our model. Modules are just PyTorch `torch.nn.Module`s that include some extra information for `nequip` about the irreps of the data they expect to input and output. See [`nequip_example_extension.nn.AddNoiseModule`](./nequip_example_extension/nn/_add_noise.py).
 2. A *model builder*, a function that takes the config (and optionally a model returned by previous model builders) and returns a new version of the model. The model builder is responsible in this case for adding our new module to an existing Allegro model. See [`nequip_example_extension.model.AddNoiseToPairEnergies`](./nequip_example_extension/model/_add_noise.py).

Extensions like this can be used from the existing YAML config files you are familar with; see [`configs/minimal_allegro_with_custom_module.yaml`](./configs/minimal_allegro_with_custom_module.yaml).

### Custom loss functions and metrics

See `loss_terms.py` and `metrics.py`.