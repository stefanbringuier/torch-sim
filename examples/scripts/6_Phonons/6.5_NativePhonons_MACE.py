"""Calculate native phonon band structure for diamond Silicon using MACE potential.

Status: testing, experimental, abandoned

Issues/Bugs:
- Negative frequencies observed for 3-bands despite good settings.
- Not seeing any speed-ups over phonopy
"""

import torch
import matplotlib.pyplot as plt
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

from torch_sim.phonons import Phonons
from torch_sim.io import atoms_to_state
from torch_sim.models.mace import MaceModel
from torch_sim.neighbors import vesin_nl_ts
from torch_sim import optimize, frechet_cell_fire, generate_force_convergence_fn


def plot_band_structure(freqs, x_positions, special_points, special_labels, title=None):
    """Plot phonon band structure."""
    plt.figure(figsize=(10, 6), dpi=100)

    n_modes = freqs.shape[1]
    for i in range(n_modes):
        plt.plot(
            x_positions,
            freqs[:, i],
            color="blue",
            lw=2,
            alpha=0.7,
        )

    for x_pos in special_points:
        plt.axvline(x=x_pos, linestyle="--", color="gray", alpha=0.7)

    plt.ylabel("Frequency (THz)", fontsize=14)
    plt.xlim(x_positions[0], x_positions[-1])
    plt.ylim(bottom=None)
    plt.xticks(special_points, special_labels, fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if title:
        plt.title(title, fontsize=16)

    plt.tight_layout()
    return plt


def main():
    # Set device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64  # Changed to float64 for consistent data types

    # Create the silicon crystal structure (diamond)
    atoms = bulk("Si", "diamond")
    atoms.set_masses([28.0855] * len(atoms))  # amu
    si_state = atoms_to_state(atoms, device, dtype)
    si_state.masses.fill_(28.0855)  # Ensure correct mass

    # Load the MACE model
    print("Loading MACE model...")
    mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
    loaded_model = mace_mp(
        model=mace_checkpoint_url,
        return_raw_model=True,
        default_dtype=dtype,
        device=device,
    )

    # Create MACE model for torch_sim
    mace_model = MaceModel(
        model=loaded_model,
        device=device,
        dtype=dtype,
        neighbor_list_fn=vesin_nl_ts,
        compute_forces=True,
        compute_stress=True,  # Need stress for cell optimization
        enable_cueq=False,
    )

    # Structure optimization
    print("Optimizing structure...")
    optimizer_params = {
        "dt_max": 1.0,  # Maximum time step
        "dt_start": 0.1,  # Initial time step
        "alpha_start": 0.1,  # Initial mixing parameter
        "cell_factor": None,  # Auto scaling based on atom count
        "hydrostatic_strain": True,  # Maintain cell shape (cubic)
        "constant_volume": False,  # Allow volume to change
        "scalar_pressure": 0.0,  # No external pressure
    }

    convergence_fn = generate_force_convergence_fn(force_tol=0.001)

    # Run the optimization
    relaxed_state = optimize(
        system=si_state,
        model=mace_model,
        optimizer=frechet_cell_fire,
        convergence_fn=convergence_fn,
        max_steps=50,
        **optimizer_params,
    )

    npoints = 60
    supercell = (6, 6, 6)
    delta = 0.05

    print("Running phonon calculations...")
    phonons = Phonons(
        state=relaxed_state,
        calculator=mace_model,
        supercell=supercell,
        delta=delta,
    )

    phonons.run()

    bandpath = atoms.cell.bandpath("GXKGL", npoints=npoints)
    kpts = torch.tensor(bandpath.kpts, dtype=dtype, device=device)

    freqs = phonons.band_structure(kpts)

    # Internal units in torch_sim are sqrt(eV/Å²/amu)
    conversion_factor = 15.633302  # THz / sqrt(eV/Å²/amu)
    freqs_thz = freqs.cpu().numpy() * conversion_factor

    special_points_str = bandpath.path.replace(",", "")
    special_labels = [point.replace("G", "Γ") for point in special_points_str]
    x_special_points = bandpath.get_linear_kpoint_axis()[1]
    x_positions = bandpath.get_linear_kpoint_axis()[0]

    plt = plot_band_structure(
        freqs_thz,
        x_positions,
        x_special_points,
        special_labels,
        title=f"Phonon Band Structure for Diamond Si",
    )

    # Save the figure
    plt.savefig("Silicon_PhononBands.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
