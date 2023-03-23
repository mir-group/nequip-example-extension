import torch

from e3nn import o3

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin


class AddNoiseModule(GraphModuleMixin, torch.nn.Module):
    """Model module that adds Gaussian noise to an arbitrary field.

    See configs/minimal_custom_module.yaml for how to use.
    """

    field: str
    noise_sigma: float
    _dim: int

    def __init__(
        self,
        field: str,
        noise_sigma: float = 0.0,
        irreps_in=None,
    ) -> None:
        super().__init__()
        self.field = field
        self.noise_sigma = noise_sigma
        # We have to tell `GraphModuleMixin` what fields we expect in the input and output
        # and what their irreps will be. Having basic geometry information (positions and edges)
        # in the input is assumed.
        # We will save the unmodified version of `field` in `field + '_noiseless'`
        # we need to tell the framework what irreps this new output field
        # `field + '_noiseless'` will have--- the same as `field`:
        self._init_irreps(
            irreps_out={field + "_noiseless": irreps_in[field]}, irreps_in=irreps_in
        )
        # this is just an e3nn.o3.Irreps...
        field_irreps: o3.Irreps = self.irreps_in[field]
        # ...whose properties we can save for later, for example:
        self._dim = field_irreps.dim

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """Run the module.
        The module both takes and returns an `AtomicDataDict.Type` = `Dict[str, torch.Tensor]`.
        Keys that the module does not modify/add are expected to be propagated to the output unchanged.
        """
        noiseless = data[self.field]
        data[self.field + "_noiseless"] = noiseless
        data[self.field] = noiseless + self.noise_sigma * torch.randn(
            (len(noiseless), self._dim), dtype=noiseless.dtype, device=noiseless.device
        )
        return data
