"""Tests for phonon calculations using the finite displacement method."""

import pytest
import torch
import numpy as np
from ase.build import bulk
from ase.phonons import Phonons as ASE_Phonons

from torch_sim.phonons import Phonons
from torch_sim.io import atoms_to_state
from torch_sim.models.lennard_jones import LennardJonesModel


@pytest.fixture
def device():
    """Provide device for testing."""
    return torch.device("cpu")


@pytest.fixture
def ar_atoms():
    """Create crystalline argon using ASE with FCC structure."""
    atoms = bulk("Ar", "fcc", a=5.26, cubic=False)
    # Set correct mass for argon
    atoms.set_masses([39.948] * len(atoms))
    return atoms


@pytest.fixture
def ar_state(ar_atoms, device):
    """Create SimState from argon atoms."""
    state = atoms_to_state(ar_atoms, device, torch.float64)
    # Ensure correct mass
    state.masses.fill_(39.948)
    return state


@pytest.fixture
def lj_params():
    """Lennard-Jones parameters for Argon.

    From: https://openkim.org/files/MO_398194508715_001/Nguyen_2005_Ar.params
    """
    return {"sigma": 3.4, "epsilon": 0.010254, "cutoff": 15.3}


@pytest.fixture
def ar_lj_model(device, lj_params):
    """Create Lennard-Jones model for Argon with specific parameters."""
    return LennardJonesModel(
        sigma=lj_params["sigma"],
        epsilon=lj_params["epsilon"],
        device=device,
        dtype=torch.float64,
        compute_forces=True,
        compute_stress=False,
        cutoff=lj_params["cutoff"],
    )


@pytest.fixture
def ase_lj_calculator(lj_params):
    """Create an ASE calculator that uses the same LJ parameters."""
    from ase.calculators.lj import LennardJones

    return LennardJones(
        epsilon=lj_params["epsilon"], sigma=lj_params["sigma"], rc=lj_params["cutoff"]
    )


def test_compare_ase_torch_phonons(ar_atoms, ar_state, ar_lj_model, ase_lj_calculator):
    """Compare phonon calculations between ASE and torch_sim for Argon."""
    bandpath = ar_atoms.cell.bandpath(npoints=100)

    supercell = (3, 3, 3)
    delta = 0.01

    ase_ph = ASE_Phonons(
        ar_atoms, calc=ase_lj_calculator, supercell=supercell, delta=delta
    )
    ase_ph.run()
    ase_ph.read(acoustic=True)

    torch_ph = Phonons(
        state=ar_state,
        calculator=ar_lj_model,
        supercell=supercell,
        delta=delta,
    )
    torch_ph.run()

    bandpath_torch = torch.tensor(bandpath.kpts, dtype=torch.float64)

    ase_freqs = ase_ph.band_structure(bandpath.kpts)
    torch_freqs = torch_ph.band_structure(bandpath_torch).cpu().numpy()

    # Compare the acoustic modes (should be close to zero)
    ase_acoustic = np.sort(np.abs(ase_freqs))[:3]
    torch_acoustic = np.sort(np.abs(torch_freqs))[:3]

    print(f"ASE acoustic modes: {ase_acoustic}")
    print(f"torch_sim acoustic modes: {torch_acoustic}")

    assert np.allclose(ase_acoustic, 0.0, atol=1e-6)
    assert np.allclose(torch_acoustic, 0.0, atol=1e-6)

    ase_optical = np.sort(np.abs(ase_freqs))[3:]
    torch_optical = np.sort(np.abs(torch_freqs))[3:]

    print(f"ASE optical modes: {ase_optical}")
    print(f"torch_sim optical modes: {torch_optical}")

    assert np.allclose(ase_optical, torch_optical, rtol=0.1)

    ase_ph.clean()


def test_direct_model_phonons(ar_atoms, ar_state, ar_lj_model):
    """Test phonon calculation using direct model as calculator."""
    # Use smaller supercell for faster testing
    supercell = (3, 3, 3)
    delta = 0.01

    # Create phonon calculator directly using the model
    torch_ph = Phonons(
        state=ar_state,
        calculator=ar_lj_model,  # Directly pass the model
        supercell=supercell,
        delta=delta,
    )

    # Run the phonon calculation
    torch_ph.run()

    # Calculate phonons at gamma point
    gamma = torch.zeros((1, 3), dtype=torch.float64)
    freqs = torch_ph.band_structure(gamma)

    # Check that we have the correct shape
    assert freqs.shape[1] == 3 * ar_state.n_atoms

    # Acoustic modes should be close to zero at gamma
    acoustic_modes = torch.sort(torch.abs(freqs[0]))[0][:3]
    assert torch.allclose(acoustic_modes, torch.zeros_like(acoustic_modes), atol=1e-4)
