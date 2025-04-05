"""Calculate phonon band structure for Argon using Lennard-Jones potential.

This example demonstrates how to:
1. Calculate phonons using the finite displacement method
2. Generate a phonon band structure along high symmetry points
3. Plot the band structure with proper labeling of high symmetry points
4. Compare with ASE's phonon implementation
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from ase.build import bulk
from ase.phonons import Phonons as ASE_Phonons
from ase.calculators.lj import LennardJones
import os
import shutil

from torch_sim.phonons import Phonons
from torch_sim.io import atoms_to_state
from torch_sim.models.lennard_jones import LennardJonesModel


def plot_band_structure(freqs, labels, title=None):
    """Plot phonon band structure from torch_sim."""
    plt.figure(figsize=(10, 6), dpi=100)

    # Generate x positions for frequency points
    x_positions = np.linspace(0, len(labels) - 1, len(freqs))

    # Plot torch_sim bands
    n_modes = freqs.shape[1]
    for i in range(n_modes):
        plt.plot(
            x_positions,
            freqs[:, i],
            color="blue",
            lw=2,
            alpha=0.7,
        )

    # Add vertical lines at high symmetry points
    for pos_idx, label in enumerate(labels):
        plt.axvline(x=pos_idx, linestyle="--", color="gray", alpha=0.7)

    # Create axis labels and formatting
    plt.ylabel("Frequency (THz)", fontsize=14)
    plt.xlim(0, len(labels) - 1)
    plt.ylim(bottom=0)
    plt.xticks(range(len(labels)), labels, fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if title:
        plt.title(title)

    plt.tight_layout()

    return plt


def main():
    # Set device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    # Create the argon crystal structure
    atoms = bulk("Ar", "fcc", a=5.26, cubic=False)
    atoms.set_masses([39.948] * len(atoms))  # amu
    ar_state = atoms_to_state(atoms, device, dtype)
    ar_state.masses.fill_(39.948)  # Ensure correct mass

    # LJ parameters
    sigma = 3.4  # Å
    epsilon = 0.010254  # eV
    cutoff = 10.0  # Å

    # Create LJ model for torch_sim
    lj_model = LennardJonesModel(
        sigma=sigma,
        epsilon=epsilon,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
        cutoff=cutoff,
    )

    # Define supercell and delta parameters
    npoints = 120
    supercell = (9, 9, 9)  # Smaller supercell for faster calculation
    delta = 0.001

    print("Running torch_sim phonon calculations...")
    # Create phonon calculator with supercell
    phonons = Phonons(
        state=ar_state,
        calculator=lj_model,
        supercell=supercell,
        delta=delta,
    )

    # Run the phonon calculation
    phonons.run()

    # Define bandpath along high symmetry points
    print("Calculating torch_sim band structure...")
    bandpath = atoms.cell.bandpath(npoints=npoints)

    # Get the band structure from torch_sim
    kpts = torch.tensor(bandpath.kpts, dtype=dtype, device=device)
    freqs = phonons.band_structure(kpts)

    # Convert frequencies to THz
    # Internal units in both torch_sim and ASE are sqrt(eV/Å²/amu)
    # The conversion factor is ħ / 2π × 10^12 Hz × √(1/eV·Å²·amu)
    conversion_factor = 15.633302  # THz / sqrt(eV/Å²/amu)
    torch_freqs_thz = freqs.cpu().numpy() * conversion_factor

    # Now calculate with ASE for comparison
    print("Running ASE phonon calculations...")

    # Create a temporary directory for ASE calculations
    ase_dir = "ase_phonon_calc"

    # Clean up any previous calculation
    if os.path.exists(ase_dir):
        shutil.rmtree(ase_dir)

    # Create LJ calculator for ASE
    ase_calc = LennardJones(epsilon=epsilon, sigma=sigma, rc=cutoff)

    # Use the same atoms object for ASE phonons
    # Will create its own copies for displacement calculations
    ase_ph = ASE_Phonons(
        atoms, calc=ase_calc, supercell=supercell, delta=delta, name=ase_dir
    )

    # Run the ASE phonon calculation
    print("Calculating displacements...")
    ase_ph.run()

    # Process the results and build force constants
    print("Reading forces and constructing force constants...")
    ase_ph.read(acoustic=True)

    # Get the ASE band structure object along the same path
    print("Calculating ASE band structure...")
    ase_bs = ase_ph.get_band_structure(bandpath)

    # Convert ASE frequency units (sqrt(eV/Å²/amu)) to THz
    ase_freqs_thz = ase_bs.energies[0] * conversion_factor

    # Extract high symmetry points info and their x-positions
    special_points = bandpath.path.replace(",", "")
    special_labels = [point.replace("G", "Γ") for point in special_points]

    # Get the special points x-coordinates from the bandpath object
    x_special_points = bandpath.get_linear_kpoint_axis()[1]

    # Create figure with two subplots for comparison
    fig = plt.figure(figsize=(12, 6), dpi=100)

    # Calculate max frequency value for consistent y-axis in THz
    max_torch_freq = torch_freqs_thz.max()
    max_ase_freq = ase_freqs_thz.max()
    y_max = max(max_torch_freq, max_ase_freq) * 1.05  # Add 5% margin

    # First subplot for torch_sim results
    ax1 = fig.add_subplot(121)

    # Use the same x-positions as ASE for exact alignment
    x_positions = bandpath.get_linear_kpoint_axis()[0]

    # Plot torch_sim bands (already in THz)
    for i in range(torch_freqs_thz.shape[1]):
        ax1.plot(x_positions, torch_freqs_thz[:, i], color="blue", lw=2)

    # Add vertical lines at high symmetry points
    for x_pos in x_special_points:
        ax1.axvline(x=x_pos, linestyle="--", color="gray", alpha=0.7)

    # Set x-axis ticks at special points
    ax1.set_xticks(x_special_points)
    ax1.set_xticklabels(special_labels)
    ax1.set_xlim(x_positions[0], x_positions[-1])
    ax1.set_ylim(0, y_max)
    ax1.set_ylabel("Frequency (THz)")
    ax1.set_title("torch_sim Phonons")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Second subplot for ASE results
    ax2 = fig.add_subplot(122)

    # Convert THz back to ASE internal units for plotting
    emax = y_max / conversion_factor

    # Plot ASE band structure using ASE's native plotting function
    ase_bs.plot(ax=ax2, emin=0.0, emax=emax)

    # Set y-axis label and title
    ax2.set_ylabel("Frequency (THz)")
    ax2.set_title("ASE Phonons")

    # Fix the y-axis ticks and labels for ASE plot
    # Define the tick positions in THz
    thz_ticks = np.linspace(0, y_max, 7)  # 7 evenly spaced ticks

    # Convert THz tick values to ASE's internal units (sqrt(eV/Å²/amu))
    ase_energy_ticks = thz_ticks / conversion_factor

    # Set the tick positions and labels
    ax2.set_yticks(ase_energy_ticks)
    ax2.set_yticklabels([f"{tick:.2f}" for tick in thz_ticks])

    plt.tight_layout()
    plt.savefig("argon_phonon_comparison.png", dpi=300, bbox_inches="tight")
    print("Done! Comparison plot saved to 'argon_phonon_comparison.png'")

    # Clean up ASE phonon calculation directory
    if os.path.exists(ase_dir):
        shutil.rmtree(ase_dir)


if __name__ == "__main__":
    main()
