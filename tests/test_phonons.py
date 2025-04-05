"""Tests for phonon calculations using the finite displacement method."""

import pytest
import torch
import numpy as np
from ase.build import bulk
from ase.phonons import Phonons as ASE_Phonons
from ase.calculators.lj import LennardJones

from torch_sim.phonons import Phonons
from torch_sim.io import atoms_to_state
from torch_sim.models.lennard_jones import LennardJonesModel

__status__ = "not passing"


@pytest.fixture
def device():
    """Provide device for testing."""
    return torch.device("cpu")


@pytest.fixture
def ar_atoms():
    """Create crystalline argon using ASE with FCC structure."""
    atoms = bulk("Ar", "fcc", a=5.26, cubic=False)
    atoms.set_masses([39.948] * len(atoms))
    return atoms


@pytest.fixture
def ar_state(ar_atoms, device):
    """Create SimState from argon atoms."""
    state = atoms_to_state(ar_atoms, device, torch.float64)
    state.masses.fill_(39.948)
    return state


@pytest.fixture
def lj_params():
    """Lennard-Jones parameters for Argon."""
    return {"sigma": 3.4, "epsilon": 0.010254, "cutoff": 10.0}


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
    return LennardJones(
        epsilon=lj_params["epsilon"], sigma=lj_params["sigma"], rc=lj_params["cutoff"]
    )


def test_compare_ase_torch_phonons(ar_atoms, ar_state, ar_lj_model, ase_lj_calculator):
    """Compare phonon calculations between ASE and torch_sim for Argon."""
    npoints = 120
    bandpath = ar_atoms.cell.bandpath(npoints=npoints)

    supercell = (9, 9, 9)
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

    # Compare acoustic modes (should be close to zero)
    ase_acoustic = np.sort(np.abs(ase_freqs))[:3]
    torch_acoustic = np.sort(np.abs(torch_freqs))[:3]

    assert np.allclose(ase_acoustic, 0.0, atol=1e-3)
    assert np.allclose(torch_acoustic, 0.0, atol=1e-3)

    # Compare optical modes
    ase_optical = np.sort(np.abs(ase_freqs))[3:]
    torch_optical = np.sort(np.abs(torch_freqs))[3:]

    assert np.allclose(ase_optical, torch_optical, rtol=0.1)

    ase_ph.clean()


def test_frequency_conversion(ar_atoms, ar_state, ar_lj_model):
    """Test frequency conversion to physical units (THz)."""
    # Calculate at gamma point for simplicity
    gamma = torch.zeros((1, 3), dtype=torch.float64)

    supercell = (3, 3, 3)
    delta = 0.01

    torch_ph = Phonons(
        state=ar_state,
        calculator=ar_lj_model,
        supercell=supercell,
        delta=delta,
    )
    torch_ph.run()

    freqs = torch_ph.band_structure(gamma)

    conversion_factor = 15.633302  # THz / sqrt(eV/Å²/amu)
    freqs_thz = freqs.cpu().numpy() * conversion_factor

    assert freqs_thz.shape[1] == 3 * ar_state.n_atoms

    acoustic_modes_thz = np.sort(np.abs(freqs_thz[0]))[:3]
    assert np.allclose(acoustic_modes_thz, 0.0, atol=1e-3)


def test_direct_model_phonons(ar_atoms, ar_state, ar_lj_model):
    """Test phonon calculation using direct model as calculator."""
    # For faster test execution
    supercell = (3, 3, 3)
    delta = 0.01

    torch_ph = Phonons(
        state=ar_state,
        calculator=ar_lj_model,
        supercell=supercell,
        delta=delta,
    )
    torch_ph.run()

    gamma = torch.zeros((1, 3), dtype=torch.float64)
    freqs = torch_ph.band_structure(gamma)

    assert freqs.shape[1] == 3 * ar_state.n_atoms

    acoustic_modes = torch.sort(torch.abs(freqs[0]))[0][:3]
    assert torch.allclose(acoustic_modes, torch.zeros_like(acoustic_modes), atol=1e-4)
