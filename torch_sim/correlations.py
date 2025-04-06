"""Correlation function calculators for time series data.

Module provides efficient calculator for time correlation functions,
including both autocorrelation and cross-correlation functionality.
LeveragesFFT-based methods for performance and supports both CPU and
GPU acceleration through PyTorch.

The primary `CorrelationCalculator` class provides on-the-fly
correlation calculations during simulation runs, and a `CircularBuffer`
utility class assist in data storage without frequent reallocations.

Example:
    Computing Velocity Autocorrelation Function in loop::

        corr_calc = CorrelationCalculator(
            window_size=100,
            delta_t=10,
            quantities={"velocity": lambda state: state.velocities},
        )

        for step in range(n_steps):
            state = integrator.step(state)
            corr_calc.update(state, step)

            # Periodically retrieve correlation functions
            if step % 1000 == 0:
                acfs = corr_calc.get_auto_correlations()
                # Process or save acfs...
"""

__status__ = "Unit tested"
__author__ = "Stefan Bringuier"
__email__ = "stefan.bringuier@gmail.com"

from collections.abc import Callable

import torch

from torch_sim.state import SimState


class CircularBuffer:
    """Circular buffer for storing time series data.

    Provides a fixed-size circular buffer optimized for storing
    and retrieving time series data, with minimal memory allocation.

    Attributes:
        size: Maximum number of elements to store
        buffer: Storage for the data
        head: Current write position
        count: Number of elements currently stored
        device: Device where the buffer is stored
    """

    def __init__(self, size: int, device: torch.device | None = None) -> None:
        """Initialize a circular buffer.

        Args:
            size: Maximum number of elements to store
            device: Device for tensor storage (CPU or GPU)
        """
        self.size = size
        self.buffer: torch.Tensor | None = None
        self.head = 0
        self.count = 0
        self.device = device

    def append(self, value: torch.Tensor) -> None:
        """Append a new value to the buffer.

        Args:
            value: New tensor to store
        """
        if self.buffer is None:
            # Initialize buffer shape as first value
            shape = (self.size, *value.shape)
            self.buffer = torch.zeros(shape, device=self.device, dtype=value.dtype)

        if self.buffer is not None:
            self.buffer[self.head] = value
            self.head = (self.head + 1) % self.size
            self.count = min(self.count + 1, self.size)

    def get_array(self) -> torch.Tensor:
        """Get the current buffer contents as a tensor.

        Returns:
            Tensor containing the buffered data in chron. order
        """
        if self.count == 0 or self.buffer is None:
            return torch.empty(0, device=self.device)

        if self.count < self.size:
            # Filled portion only!
            return self.buffer[: self.count]

        # Chronological order
        # Avoid unnecessary copy if unwrapped
        if self.head == 0:
            return self.buffer

        return torch.cat([self.buffer[self.head :], self.buffer[: self.head]])

    @property
    def is_full(self) -> bool:
        """Check if the buffer is full.

        Returns:
            True if buffer contains size elements, False otherwise
        """
        return self.count == self.size


class CorrelationCalculator:
    """Efficient on-the-fly correlation function calculator.

    Manage the calculation of time correlation functions during
    simulation, with support for both autocorrelation and cross-correlation
    of arbitrary quantities. It maintains a sliding window of historical data
    and performs efficient updates.

    Attributes:
        window_size: Number of steps to keep in memory
        delta_t: Steps between correlation calculations
        quantities: Map of quantity names to their calculators
        buffers: Circular buffers for storing historical data
        correlations: Current correlation results
        device: Device where calculations are performed
    """

    def __init__(
        self,
        window_size: int,
        delta_t: int = 1,
        quantities: dict[str, Callable[[SimState], torch.Tensor]] | None = None,
        device: torch.device | None = None,
        *,
        normalize: bool = True,
    ) -> None:
        """Initialize a correlation calculator.

        Args:
            window_size: Number of steps to keep in memory
            delta_t: step between correlation calculations
            quantities: Dictionary mapping names to functions that calculate
                       quantities from a SimState
            device: Device for tensor storage and computation
            normalize: Whether to normalize correlation functions to [0,1]
        """
        self.window_size = window_size
        self.delta_t = delta_t
        self.quantities = quantities or {}
        self.device = device
        self.normalize = normalize

        self.buffers = {
            name: CircularBuffer(window_size, device=device) for name in self.quantities
        }

        self.correlations: dict[str, torch.Tensor] = {}
        self.cross_correlations: dict[tuple[str, str], torch.Tensor] = {}

    def add_quantity(
        self, name: str, calculator: Callable[[SimState], torch.Tensor]
    ) -> None:
        """Track a new simulation quantity.

        Args:
            name: Name of the quantity
            calculator: Function that calculates quantity from a SimState
        """
        if name in self.quantities:
            raise ValueError(f"Quantity {name} already exists")

        self.quantities[name] = calculator
        self.buffers[name] = CircularBuffer(self.window_size, device=self.device)

    def update(self, state: SimState, step: int) -> None:
        """Update correlation calculations with new state data.

        Args:
            state: Current simulation state
            step: Current simulation step
        """
        if step % self.delta_t != 0:
            return

        # Single pass update
        buffer_count = 0
        buffer_total = len(self.buffers)

        for name, calc in self.quantities.items():
            value = calc(state)
            self.buffers[name].append(value)
            if self.buffers[name].count > 1:
                buffer_count += 1

        # Correlations if we have enough data
        if buffer_count == buffer_total and buffer_total > 0:
            self._compute_correlations()

    def _compute_correlations(self) -> None:  # noqa: C901, PLR0915
        """Compute correlation functions using FFT for efficiency."""
        # Autocorrelations
        for name, buf in self.buffers.items():
            data = buf.get_array()
            if len(data) == 0:
                continue

            original_shape = data.shape

            # Reshape to [time_steps, flattened_dim]
            if len(original_shape) > 1:
                data = data.reshape(original_shape[0], -1)

            n_dims = data.shape[1] if len(data.shape) > 1 else 1

            if n_dims > 1:
                # Pre-allocate/Precompute
                acf = torch.zeros(
                    (original_shape[0], n_dims), device=self.device, dtype=data.dtype
                )

                data_centered = data - data.mean(dim=0, keepdim=True)

                if data_centered.shape[1] <= 128:  # Batch Threshold
                    # Transpose for batch FFT (dimensions become batch)
                    data_batch = data_centered.T  # Shape: [n_dims, time_steps]

                    # Batch FFT processing
                    n_fft = 2 * data_batch.shape[1]
                    fft_batch = torch.fft.rfft(data_batch, n=n_fft)
                    power_batch = torch.abs(fft_batch) ** 2
                    corr_batch = torch.fft.irfft(power_batch)[:, : data_batch.shape[1]]

                    corr_batch = corr_batch.T  # Shape: [time_steps, n_dims]

                    if self.normalize:
                        norms = corr_batch[0].clone()
                        mask = norms > 1e-10
                        if mask.any():
                            corr_batch[:, mask] = corr_batch[:, mask] / norms[
                                mask
                            ].unsqueeze(0)

                    acf = corr_batch.reshape(original_shape)
                else:
                    # Fallback for very high-dimensional data
                    for i in range(n_dims):
                        dim_data = data_centered[:, i]

                        # FFT: n=2*len for zero-padding
                        n_fft = 2 * len(dim_data)
                        fft = torch.fft.rfft(dim_data, n=n_fft)
                        power = torch.abs(fft) ** 2
                        corr = torch.fft.irfft(power)[: len(dim_data)]

                        if self.normalize and corr[0] > 1e-10:
                            corr = corr / corr[0]

                        acf[:, i] = corr

                    # Reshape back to match input dimensions
                    acf = acf.reshape(original_shape)
            else:
                # Single dimension case
                dim_data = data - data.mean()

                n_fft = 2 * len(dim_data)
                fft = torch.fft.rfft(dim_data, n=n_fft)
                power = torch.abs(fft) ** 2
                corr = torch.fft.irfft(power)[: len(dim_data)]

                if self.normalize and corr[0] > 1e-10:
                    corr = corr / corr[0]

                acf = corr

            self.correlations[name] = acf

        # Cross-correlations
        names = list(self.buffers.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i + 1 :]:
                data1 = self.buffers[name1].get_array()
                data2 = self.buffers[name2].get_array()

                if len(data1) == 0 or len(data2) == 0:
                    continue

                min_len = min(len(data1), len(data2))
                data1 = data1[:min_len]
                data2 = data2[:min_len]

                # Multidimensional data
                if len(data1.shape) > 1 or len(data2.shape) > 1:
                    # For now, simplify by taking mean across dimensions
                    if len(data1.shape) > 1:
                        # More efficient with tuple unpacking
                        non_time_dims = tuple(range(1, len(data1.shape)))
                        data1 = torch.mean(data1, dim=non_time_dims)
                    if len(data2.shape) > 1:
                        non_time_dims = tuple(range(1, len(data2.shape)))
                        data2 = torch.mean(data2, dim=non_time_dims)

                # Center data
                data1 = data1 - data1.mean()
                data2 = data2 - data2.mean()

                n_fft = 2 * min_len
                fft1 = torch.fft.rfft(data1, n=n_fft)
                fft2 = torch.fft.rfft(data2, n=n_fft)
                ccf = torch.fft.irfft(fft1 * fft2.conj())[:min_len]

                if self.normalize and torch.abs(ccf[0]) > 1e-10:
                    ccf = ccf / ccf[0]

                self.cross_correlations[(name1, name2)] = ccf

    def get_auto_correlations(self) -> dict[str, torch.Tensor]:
        """Get autocorrelation results.

        Returns:
            Dictionary mapping quantity names to their correlation tensors
        """
        return self.correlations

    def get_cross_correlations(self) -> dict[tuple[str, str], torch.Tensor]:
        """Get cross-correlation results.

        Returns:
            Dictionary mapping pairs of quantity names to their
            cross-correlation tensors
        """
        return self.cross_correlations

    def reset(self) -> None:
        """Reset all buffers and correlations."""
        self.buffers = {
            name: CircularBuffer(self.window_size, device=self.device)
            for name in self.quantities
        }
        self.correlations = {}
        self.cross_correlations = {}

    def to(self, device: torch.device) -> "CorrelationCalculator":
        """Move calculator to specified device.

        Args:
            device: Target device

        Returns:
            Self, for method chaining
        """
        # Skip if already on target device
        if self.device == device:
            return self

        self.device = device

        new_buffers = {}
        for name, buf in self.buffers.items():
            new_buf = CircularBuffer(self.window_size, device=device)
            if buf.buffer is not None:
                data = buf.get_array().to(device)
                # Larger buffers use a batch
                if len(data) > 100:
                    # Balances memory transfer
                    batch_size = 20
                    for i in range(0, len(data), batch_size):
                        batch = data[i : min(i + batch_size, len(data))]
                        for j in range(len(batch)):
                            new_buf.append(batch[j])
                else:
                    for i in range(len(data)):
                        new_buf.append(data[i])
            new_buffers[name] = new_buf

        self.buffers = new_buffers

        # Move correlations
        if self.correlations:
            self.correlations = {
                name: corr.to(device) for name, corr in self.correlations.items()
            }

        if self.cross_correlations:
            self.cross_correlations = {
                names: corr.to(device) for names, corr in self.cross_correlations.items()
            }

        return self
