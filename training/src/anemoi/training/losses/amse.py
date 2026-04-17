from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch_harmonics as th
from anemoi.training.losses.base import BaseLoss
from einops import rearrange
from torch.amp.autocast_mode import autocast

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

from .utils.regrid import get_regrid_config

LOGGER = logging.getLogger(__name__)


class BaseMSHLoss(BaseLoss, ABC):
    """
    Abstract base class for Modified Spherical Harmonic (MSH) loss functions, as described
    in Subich et al., "Fixing the Double Penalty in Data-Driven Weather Forecasting Through
    a Modified Spherical Harmonic Loss Function", arXiv:2501.19374, 2025.

    Implements the shared `forward` template and the AMSE core formula:

        AMSE = sum_{spectral dims}(
            (sqrt(PSD_pred) - sqrt(PSD_target))^2
            + 2 * max(PSD_pred, PSD_target) * (1 - coherence)
        )

    where coherence = cross / (sqrt(PSD_pred) * sqrt(PSD_target) + eps).

    Subclasses must implement:
        - `_ensure_initialised(device)`: initialise any device-dependent components.
        - `_compute_amse(pred, target)`: return AMSE of shape [B, T, E, n_vars].
    """

    eps: float = 1e-8

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        # shape of inputs: [batch_size, output_times, ensemble, num_gridpoints, num_vars]
        self.batch_size, self.output_times, self.ensemble_size, self.n_gridpoints, self.n_vars = (
            pred.shape
        )

        # ensure all components are initialised and on the same device
        self._ensure_initialised(pred.device)

        # compute AMSE in full precision: [batch_size, output_times, ensemble_size, n_vars]
        with autocast("cuda", enabled=False):
            amse = self._compute_amse(pred, target)

        # add a dummy grid dimension to match expected shape for further processing
        amse = amse.unsqueeze(-2)  # shape: [batch_size, output_times, ensemble_size, 1, n_vars]

        # apply weights to the AMSE calculation, ignoring dummy grid dimension
        amse = self.scale(
            amse,
            subset_indices=scaler_indices,
            without_scalers=([3] if amse.shape[3] == 1 else without_scalers),
            grid_shard_slice=grid_shard_slice,
        )

        # use the base class reduction:
        # - sums over (dummy) grid dimension
        # - averages over batch, output_times, and ensemble dimensions
        # - either averages (default) or sums over variable dimension
        return self.reduce(amse, squash=squash, squash_mode="avg", group=group)

    @abstractmethod
    def _ensure_initialised(self, device: torch.device) -> None:
        """Initialise any device-dependent components (e.g. kernels, weight tensors)."""
        ...

    @abstractmethod
    def _compute_amse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute AMSE between pred and target.

        Args:
            pred: shape [batch_size, output_times, ensemble_size, n_gridpoints, n_vars]
            target: shape [batch_size, output_times, ensemble_size, n_gridpoints, n_vars]

        Returns:
            AMSE of shape [batch_size, output_times, ensemble_size, n_vars]
        """
        ...

    def _amse_core(
        self,
        psd_pred: torch.Tensor,
        psd_target: torch.Tensor,
        cross: torch.Tensor,
        spectral_sum_dims: tuple[int, ...],
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute AMSE given power spectral densities and cross-spectral density.

        Args:
            psd_pred: Power spectral density of predictions.
            psd_target: Power spectral density of targets.
            cross: Real part of the cross-spectral density, Re[A * conj(B)].
            spectral_sum_dims: Dimensions to sum over after computing per-component AMSE.
            weight: Optional per-frequency weight tensor, broadcastable to psd_pred.

        Returns:
            AMSE summed over spectral_sum_dims.
        """
        amp_pred = torch.sqrt(psd_pred + self.eps)
        amp_target = torch.sqrt(psd_target + self.eps)
        coherence = cross / (amp_pred * amp_target + self.eps)
        amse = (amp_pred - amp_target) ** 2 + 2 * torch.maximum(psd_pred, psd_target) * (
            1 - coherence
        )
        if weight is not None:
            amse = amse * weight
        return amse.sum(dim=spectral_sum_dims)


class GlobalSpectralLoss(BaseMSHLoss, ABC):
    """
    Abstract base class for global MSH losses that use the Spherical Harmonic Transform (SHT).

    Uses the torch-harmonics `RealSHT` on an equiangular grid. Implements `_compute_amse`
    by computing SHT spectral coefficients, reducing over the zonal-wavenumber (l) dimension
    to obtain per-total-wavenumber (k) PSDs, and calling `_amse_core` to sum over k.

    Subclasses must implement `_prepare_grid`, which converts the native model grid to the
    2D equiangular lat/lon layout expected by the SHT kernel.

    See `ReducedGaussianMSHLoss` and `LatLonMSHLoss` for concrete implementations.

    Note on index labelling: "k" denotes total wavenumber and "l" the zonal wavenumber,
    following the paper notation. torch-harmonics uses "l"/"m" instead.
    """

    def __init__(
        self,
        ignore_nans: bool = False,
        nlats: int | None = None,
        nlons: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        self.nlats = nlats
        self.nlons = nlons
        self.k = nlons // 2  # maximum total wavenumber
        self._sht_kernel: th.RealSHT | None = None
        self._device: torch.device | None = None

    def _ensure_initialised(self, device: torch.device) -> None:
        """Initialise the SHT kernel on the given device."""
        if self._device != device:
            self._device = device
            self._sht_kernel = th.RealSHT(
                self.nlats,
                self.nlons,
                grid="equiangular",
                lmax=self.k,
            ).to(device)

    @abstractmethod
    def _prepare_grid(self, tensor: torch.Tensor, requires_grad: bool) -> torch.Tensor:
        """
        Convert the input tensor to the 2D equiangular layout expected by the SHT kernel.

        Args:
            tensor: shape [batch_size, output_times, ensemble_size, n_gridpoints, n_vars]
            requires_grad: whether to retain gradients through this operation

        Returns:
            Tensor of shape [batch_size * output_times * ensemble_size * n_vars, nlats, nlons]
        """
        ...

    def _calculate_spectral_coefficients(
        self, tensor: torch.Tensor, requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Compute SHT spectral coefficients.

        Args:
            tensor: shape [batch_size * output_times * ensemble_size * n_vars, nlats, nlons]
            requires_grad: whether to retain gradients

        Returns:
            Complex tensor of shape [batch_size * output_times * ensemble_size, k, k+1, n_vars]
        """
        if self._sht_kernel is None:
            err_msg = "SHT kernel not initialised. Call _ensure_initialised first."
            raise RuntimeError(err_msg)

        grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
        with grad_context:
            spec_coeffs = self._sht_kernel.forward(tensor)
        return rearrange(
            spec_coeffs,
            "(bs_ot_ens n_vars) k kp1 -> bs_ot_ens k kp1 n_vars",
            bs_ot_ens=self.batch_size * self.output_times * self.ensemble_size,
            n_vars=self.n_vars,
            k=self.k,
            kp1=self.k + 1,
        )

    def _compute_amse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute AMSE via the SHT path.

        Prepares the grid, computes SHT coefficients, reduces over the l-dimension to get
        per-k PSD, then calls `_amse_core` to sum over k.
        """
        # prepare grid: [B*T*E*V, nlats, nlons]
        pred_grid = self._prepare_grid(pred, requires_grad=True)
        target_grid = self._prepare_grid(target, requires_grad=False)

        # spectral coefficients: [bte, k, k+1, n_vars] (complex)
        sc_pred = self._calculate_spectral_coefficients(pred_grid, requires_grad=True)
        sc_target = self._calculate_spectral_coefficients(target_grid, requires_grad=False)

        # reduce over the l-dimension (dim=-2) to get per-k PSD: [bte, k, n_vars]
        psd_pred = (sc_pred.real**2 + sc_pred.imag**2).sum(dim=-2)
        psd_target = (sc_target.real**2 + sc_target.imag**2).sum(dim=-2)
        # cross-spectrum summed over l: [bte, k, n_vars]
        cross = (sc_pred.real * sc_target.real + sc_pred.imag * sc_target.imag).sum(dim=-2)

        # reshape to [batch_size, output_times, ensemble_size, k, n_vars]
        psd_pred = rearrange(
            psd_pred,
            "(bs ot ens) k n_vars -> bs ot ens k n_vars",
            bs=self.batch_size,
            ot=self.output_times,
            ens=self.ensemble_size,
            k=self.k,
            n_vars=self.n_vars,
        )
        psd_target = rearrange(
            psd_target,
            "(bs ot ens) k n_vars -> bs ot ens k n_vars",
            bs=self.batch_size,
            ot=self.output_times,
            ens=self.ensemble_size,
            k=self.k,
            n_vars=self.n_vars,
        )
        cross = rearrange(
            cross,
            "(bs ot ens) k n_vars -> bs ot ens k n_vars",
            bs=self.batch_size,
            ot=self.output_times,
            ens=self.ensemble_size,
            k=self.k,
            n_vars=self.n_vars,
        )

        # AMSE: sum over k (dim=-2) → [batch_size, output_times, ensemble_size, n_vars]
        return self._amse_core(psd_pred, psd_target, cross, spectral_sum_dims=(-2,))


class ReducedGaussianMSHLoss(GlobalSpectralLoss):
    """
    MSH loss for models trained on a reduced Gaussian grid (e.g. O96, N320).

    Regrids inputs to an equiangular lat/lon grid using a precomputed sparse regridding
    matrix, then applies the Spherical Harmonic Transform (SHT) for spectral decomposition.
    """

    name: str = "msh"

    def __init__(
        self,
        ignore_nans: bool = False,
        input_grid: str = "O96",
        **kwargs,
    ) -> None:
        regrid_config = get_regrid_config(input_grid.lower())
        super().__init__(
            ignore_nans=ignore_nans,
            nlats=regrid_config.output_nlats,
            nlons=regrid_config.output_nlons,
            **kwargs,
        )
        self.regrid_config = regrid_config

    def _prepare_grid(self, tensor: torch.Tensor, requires_grad: bool) -> torch.Tensor:
        """
        Regrid from the reduced Gaussian grid to an equiangular lat/lon grid.

        Args:
            tensor: shape [batch_size, output_times, ensemble_size, n_gridpoints, n_vars]
            requires_grad: whether to retain gradients through regridding

        Returns:
            Tensor of shape [batch_size * output_times * ensemble_size * n_vars, nlats, nlons]
        """
        return self.regrid_config.regrid_training(
            tensor,
            regrid_matrix=self.regrid_config.regridding_matrix,
            requires_grad=requires_grad,
        )


class LatLonMSHLoss(GlobalSpectralLoss):
    """
    MSH loss for models trained on a global regular lat/lon grid.

    No regridding is required — inputs are reshaped directly to [nlats, nlons] for the
    Spherical Harmonic Transform (SHT). The SHT grid type (e.g. "equiangular",
    "legendre-gauss") can be configured via `sht_grid`.
    """

    name: str = "msh_global_latlon"

    def __init__(
        self,
        ignore_nans: bool = False,
        grid_shape: list[int] | None = None,
        sht_grid: str = "equiangular",
        **kwargs,
    ) -> None:
        if grid_shape is None:
            err_msg = "grid_shape must be provided for LatLonMSHLoss, e.g. [721, 1440]"
            raise ValueError(err_msg)
        nlats, nlons = grid_shape
        super().__init__(ignore_nans=ignore_nans, nlats=nlats, nlons=nlons, **kwargs)
        self.sht_grid_type = sht_grid

    def _ensure_initialised(self, device: torch.device) -> None:
        """Initialise the SHT kernel on the given device, using the configured grid type."""
        if self._device != device:
            self._device = device
            self._sht_kernel = th.RealSHT(
                self.nlats,
                self.nlons,
                grid=self.sht_grid_type,
                lmax=self.k,
            ).to(device)

    def _prepare_grid(self, tensor: torch.Tensor, requires_grad: bool) -> torch.Tensor:
        """
        Reshape the flat gridpoint dimension to (nlats, nlons).

        Args:
            tensor: shape [batch_size, output_times, ensemble_size, nlats * nlons, n_vars]
            requires_grad: whether to retain gradients

        Returns:
            Tensor of shape [batch_size * output_times * ensemble_size * n_vars, nlats, nlons]
        """
        grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
        with grad_context:
            return rearrange(
                tensor,
                "b t e (h w) v -> (b t e v) h w",
                h=self.nlats,
                w=self.nlons,
            )


class RegionalSpectralLoss(BaseMSHLoss, ABC):
    """
    Abstract base class for MSH losses on regional regular lat/lon grids.

    Implements the shared `forward` path: reshapes inputs from flat-grid to [B, T, E, V, H, W]
    and delegates to `_spectral_amse_from_2d`. Subclasses implement the specific spectral
    transform (2D FFT or DCT-II) and precompute the appropriate frequency weights.

    See `RegionalFFTLoss` and `RegionalDCTLoss` for concrete implementations.
    """

    def __init__(
        self,
        ignore_nans: bool = False,
        grid_shape: list[int] | None = None,
        frequency_power_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        if grid_shape is None:
            err_msg = "grid_shape must be provided for RegionalSpectralLoss, e.g. [808, 621]"
            raise ValueError(err_msg)
        self.nlats, self.nlons = grid_shape
        self.n_gridpoints = self.nlats * self.nlons
        self.frequency_power_weight = frequency_power_weight
        self.register_buffer("combined_weights", self._precompute_combined_weights())

    @abstractmethod
    def _precompute_combined_weights(self) -> torch.Tensor:
        """Precompute and return the combined per-frequency weight tensor."""
        ...

    @abstractmethod
    def _spectral_amse_from_2d(
        self, pred_2d: torch.Tensor, target_2d: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spectral AMSE from pre-shaped 2D tensors.

        Args:
            pred_2d: shape [B, T, E, V, H, W]
            target_2d: shape [B, T, E, V, H, W]

        Returns:
            AMSE of shape [B, T, E, V]
        """
        ...

    def _ensure_initialised(self, _device: torch.device) -> None:
        """Move precomputed weights to the given device."""

    def _compute_amse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Reshape inputs from flat-grid layout to [B, T, E, V, H, W] and delegate to
        `_spectral_amse_from_2d`.
        """
        pred_2d = rearrange(pred, "b t e (h w) v -> b t e v h w", h=self.nlats, w=self.nlons)
        target_2d = rearrange(target, "b t e (h w) v -> b t e v h w", h=self.nlats, w=self.nlons)
        return self._spectral_amse_from_2d(pred_2d, target_2d)


class RegionalFFTLoss(RegionalSpectralLoss):
    """
    Regional MSH loss using a 2D real FFT (rfft2).

    Suitable for regional (non-global, non-periodic) domains. Note that the FFT implicitly
    assumes periodicity, which introduces spectral leakage at boundaries — see
    `RegionalDCTLoss` for a boundary-aware alternative.
    """

    name: str = "msh_regional"

    def _precompute_combined_weights(self) -> torch.Tensor:
        """Combined rfft2 correction and frequency scaling weights, shape [1, 1, 1, 1, H, W//2+1]."""
        return self._precompute_rfft2_weights() * self._precompute_rfft2_frequency_scaling()

    def _precompute_rfft2_weights(self) -> torch.Tensor:
        """
        Weights of shape [1, 1, 1, 1, H, W//2+1] for rfft2 so that a weighted sum over
        the rfft2 output reproduces the full fft2 energy (DC and Nyquist bins are not
        doubled; all other bins are doubled to account for the conjugate half).
        """
        Wr = self.nlons // 2 + 1
        wts = torch.full((self.nlats, Wr), 2.0)
        wts[:, 0] = 1.0  # DC column (kx = 0)
        if self.nlons % 2 == 0:
            wts[:, -1] = 1.0  # Nyquist column (if nlons is even)
        return wts.view(1, 1, 1, 1, self.nlats, -1)

    def _precompute_rfft2_frequency_scaling(self) -> torch.Tensor:
        """
        Power-law frequency boost weights of shape [1, 1, 1, 1, H, W//2+1].
        Returns all-ones (no boost) when frequency_power_weight == 0 (default).
        Normalised so the mean weight is 1, preserving the overall loss scale.
        """
        if self.frequency_power_weight == 0.0:
            return torch.ones((1, 1, 1, 1, self.nlats, self.nlons // 2 + 1))

        kx = torch.fft.rfftfreq(self.nlons, d=1.0).view(1, 1, 1, 1, 1, -1)
        ky = torch.fft.fftfreq(self.nlats, d=1.0).view(1, 1, 1, 1, -1, 1)
        freq_radius = torch.sqrt(kx**2 + ky**2)
        hf_boost = (1.0 + freq_radius) ** self.frequency_power_weight
        hf_boost /= hf_boost.mean()
        return hf_boost

    def _spectral_amse_from_2d(
        self, pred_2d: torch.Tensor, target_2d: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute FFT-based AMSE from pre-shaped 2D tensors.

        Args:
            pred_2d: shape [B, T, E, V, H, W]
            target_2d: shape [B, T, E, V, H, W]

        Returns:
            AMSE of shape [B, T, E, V], summed over (H, W//2+1) frequency bins.
        """
        with torch.enable_grad():
            fft_pred = torch.fft.rfft2(pred_2d, dim=(-2, -1), norm="ortho")
        with torch.no_grad():
            fft_target = torch.fft.rfft2(target_2d, dim=(-2, -1), norm="ortho")

        psd_pred = fft_pred.real**2 + fft_pred.imag**2
        psd_target = fft_target.real**2 + fft_target.imag**2
        cross = fft_pred.real * fft_target.real + fft_pred.imag * fft_target.imag

        amse = self._amse_core(
            psd_pred,
            psd_target,
            cross,
            spectral_sum_dims=(-2, -1),
            weight=self.combined_weights,
        )
        return amse / self.n_gridpoints
