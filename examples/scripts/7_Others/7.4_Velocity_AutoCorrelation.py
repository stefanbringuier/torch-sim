"""Velocity autocorrelation example."""

# /// script
# dependencies = [
#     "ase>=3.24",
#     "matplotlib",
#     "numpy",
# ]
# ///

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import torch_sim as ts
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.properties.correlations import CorrelationCalculator
from torch_sim.units import MetalUnits as Units


def prepare_system() -> tuple[
    Any, Any, torch.Tensor, torch.Tensor, torch.device, torch.dtype, float
]:
    """Create and prepare Ar system with LJ potential."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    # Using solid Ar w/ LJ for ease
    atoms = bulk("Ar", crystalstructure="fcc", a=5.256, cubic=True)
    atoms = atoms.repeat((3, 3, 3))
    temperature = 50.0  # Kelvin
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    state = ts.io.atoms_to_state(atoms, device=device, dtype=dtype)

    epsilon = 0.0104  # eV
    sigma = 3.4  # Å
    cutoff = 2.5 * sigma

    lj_model = LennardJonesModel(
        sigma=sigma,
        epsilon=epsilon,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
        compute_forces=True,
    )

    timestep = 0.001  # ps (1 fs)
    dt = torch.tensor(timestep * Units.time, device=device, dtype=dtype)
    temp_kT = temperature * Units.temperature  # Convert K to internal units
    kT = torch.tensor(temp_kT, device=device, dtype=dtype)

    return state, lj_model, dt, kT, device, dtype, timestep


def plot_results(
    *,  # Force keyword-only arguments
    full_time: np.ndarray | None,
    running_avg_vacf: np.ndarray | None,
    last_window_vacf: np.ndarray | None,
    window_count: int,
    use_running_avg: bool = True,
) -> None:
    """Plot VACF results."""
    plt.figure(figsize=(10, 8))
    plot_vacf: np.ndarray | None = None

    if use_running_avg and running_avg_vacf is not None:
        plot_vacf = running_avg_vacf
        title = f"VACF (Average of {window_count} windows)"
    else:
        plot_vacf = last_window_vacf
        title = "VACF (Last window only)"

    if full_time is not None and plot_vacf is not None:
        plt.plot(full_time, plot_vacf, "b-", linewidth=2)
        plt.xlabel("Time (fs)", fontsize=12)
        plt.ylabel("VACF", fontsize=12)
        plt.title(title, fontsize=14)
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        plt.ylim(-0.6, 1.1)
        plt.tight_layout()
        plt.savefig("vacf_example.png")


def main() -> None:
    """Run velocity autocorrelation simulation using Lennard-Jones model."""
    state, lj_model, dt, kT, device, dtype, timestep = prepare_system()
    nve_init, nve_update = ts.integrators.nve(model=lj_model, dt=dt, kT=kT)
    state = nve_init(state)  # type: ignore[call-arg]

    correlation_dt = 10  # Step delta between correlations
    # Length of correlation: dt * correlation_dt * window_size
    window_size = 150
    use_running_average = True

    corr_calc = CorrelationCalculator(
        window_size=window_size,
        properties={"velocity": lambda s: s.velocities},  # type: ignore[attr-defined]
        device=device,
        normalize=True,
    )

    window_count = 0
    running_avg_vacf: np.ndarray | None = None
    last_window_vacf: np.ndarray | None = None
    full_time: np.ndarray | None = None
    all_window_vacfs: list[np.ndarray] = []

    def calc_vacf(state: Any, _: Any = None) -> torch.Tensor:
        """Calculate velocity autocorrelation."""
        nonlocal window_count, running_avg_vacf, last_window_vacf
        nonlocal full_time, all_window_vacfs

        corr_calc.update(state)
        buffer_filled = corr_calc.buffers["velocity"].count

        if buffer_filled == window_size:
            correlations = corr_calc.get_auto_correlations()

            # shape: (time, atoms, dims)
            raw_vacf = correlations["velocity"]
            # average over atoms and dimensions
            vacf = torch.mean(raw_vacf, dim=(1, 2))

            vacf_np = vacf.cpu().numpy()
            # Convert to fs for plotting
            time_steps = np.arange(len(vacf_np))
            time_np = time_steps * correlation_dt * timestep * 1000

            if full_time is None:
                full_time = time_np

            window_count += 1
            all_window_vacfs.append(vacf_np)

            if running_avg_vacf is None:
                running_avg_vacf = vacf_np.copy()
            elif use_running_average:
                # Compute new running average
                factor = 1.0 / window_count
                diff = vacf_np - running_avg_vacf
                running_avg_vacf = running_avg_vacf + diff * factor

            last_window_vacf = vacf_np.copy()
            corr_calc.reset()

        return torch.tensor([window_count], device=device, dtype=dtype)

    # The sampling frequency is now controlled entirely by the prop_calculators
    trajectory = "vacf_example.h5"
    reporter = ts.TrajectoryReporter(
        trajectory,
        state_frequency=100,
        prop_calculators={correlation_dt: {"vacf": calc_vacf}},
    )

    num_steps = 15000  # NOTE: short run
    for step in range(num_steps):
        state = nve_update(state)  # type: ignore[call-arg]
        reporter.report(state, step)

    reporter.close()

    plot_results(
        full_time=full_time,
        running_avg_vacf=running_avg_vacf,
        last_window_vacf=last_window_vacf,
        window_count=window_count,
        use_running_avg=use_running_average,
    )


if __name__ == "__main__":
    main()
