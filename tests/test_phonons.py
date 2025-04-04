"""Tests for phonon calculations using the finite displacement method."""

import pytest
import torch
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.phonons import Phonons as ASEPhonons
from ase.dft.kpoints import monkhorst_pack

from torch_sim.phonons import Phonons, AtomicDisplacement
from torch_sim.state import SimState
from torch_sim.io import atoms_to_state
from torch_sim.transforms import get_pair_displacements
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.units import UnitConversion, BaseConstant as bc


@pytest.fixture
def device():
    """Provide device for testing."""
    return torch.device("cpu")


@pytest.fixture
def al_atoms():
    """Create aluminum crystal using ASE."""
    return bulk("Al", cubic=True, a=4.05)


@pytest.fixture
def si_atoms():
    """Create crystalline silicon using ASE."""
    return bulk("Si", "diamond", a=5.43, cubic=True)


@pytest.fixture
def cu_atoms():
    """Create crystalline copper using ASE."""
    return bulk("Cu", "fcc", a=3.58, cubic=True)


@pytest.fixture
def al_state(al_atoms, device):
    """Create SimState from aluminum atoms."""
    return atoms_to_state(al_atoms, device, torch.float64)


@pytest.fixture
def si_state(si_atoms, device):
    """Create SimState from silicon atoms."""
    return atoms_to_state(si_atoms, device, torch.float64)


@pytest.fixture
def cu_state(cu_atoms, device):
    """Create SimState from copper atoms."""
    return atoms_to_state(cu_atoms, device, torch.float64)


@pytest.fixture
def lj_calculator(device):
    """Create Lennard-Jones calculator."""
    return LennardJonesModel(
        sigma=2.5,
        epsilon=0.01,
        device=device,
        dtype=torch.float64,
        compute_forces=True,
        compute_stress=True,
        cutoff=8.0,
    )


def calculator_wrapper(model):
    """Wrap a model to be used as a calculator for Phonons."""

    def calc_function(state):
        return model(state)

    return calc_function


def test_atomic_displacement_init(al_state, lj_calculator):
    """Test initialization of AtomicDisplacement class."""
    # Test default initialization
    disp = AtomicDisplacement(
        state=al_state,
        calculator=calculator_wrapper(lj_calculator),
        supercell=(2, 2, 2),
    )

    assert disp.state is al_state
    assert disp._supercell == (2, 2, 2)
    assert disp.name == "displacement"
    assert disp.delta == 0.01
    assert disp.center_cell is False
    assert torch.all(
        disp.indices == torch.arange(al_state.n_atoms, device=al_state.device)
    )


def test_atomic_displacement_create_supercell(al_state, lj_calculator):
    """Test supercell creation in AtomicDisplacement."""
    disp = AtomicDisplacement(
        state=al_state,
        calculator=calculator_wrapper(lj_calculator),
        supercell=(2, 2, 2),
    )

    supercell = disp._create_supercell()

    # Check that supercell has correct number of atoms
    assert supercell.n_atoms == al_state.n_atoms * 2 * 2 * 2

    # Check that supercell has correct cell dimensions
    cell_vectors = supercell.cell.squeeze()
    original_cell = al_state.cell.squeeze()
    for i in range(3):
        assert torch.allclose(cell_vectors[i], original_cell[i] * 2)


def test_atomic_displacement_run(al_state, lj_calculator):
    """Test running atomic displacements."""
    disp = AtomicDisplacement(
        state=al_state,
        calculator=calculator_wrapper(lj_calculator),
        supercell=(1, 1, 1),
        delta=0.01,
    )

    # Run displacements
    disp.run()

    # Check that cache contains equilibrium forces
    eq_name = f"{disp.name}.eq"
    assert eq_name in disp.cache
    assert "forces" in disp.cache[eq_name]

    # Check that cache contains forces for all displacements
    n_displacements = len(disp.indices) * 3 * 2  # atoms * directions * signs
    assert len(disp.cache) == n_displacements + 1  # +1 for equilibrium


def test_phonons_init(al_state, lj_calculator):
    """Test initialization of Phonons class."""
    # Test default initialization
    ph = Phonons(
        state=al_state,
        calculator=calculator_wrapper(lj_calculator),
        supercell=(2, 2, 2),
    )

    assert ph.state is al_state
    assert ph._supercell == (2, 2, 2)
    assert ph.name == "phonon"
    assert ph.delta == 0.01
    assert ph.center_cell is False
    assert torch.all(
        ph.indices == torch.arange(al_state.n_atoms, device=al_state.device)
    )
    assert ph.C_N is None
    assert ph.D_N is None
    assert ph.m_inv_x is None


def test_phonons_run_and_read(al_state, lj_calculator):
    """Test running displacements and reading force constants."""
    ph = Phonons(
        state=al_state,
        calculator=calculator_wrapper(lj_calculator),
        supercell=(1, 1, 1),
        delta=0.01,
    )

    # Run displacements
    ph.run()

    # Read force constants
    ph.read(method="frederiksen", symmetrize=3, acoustic=True)

    # Check that force constants and dynamical matrix have been calculated
    assert ph.C_N is not None
    assert ph.D_N is not None
    assert ph.m_inv_x is not None

    # Check dimensions of force constants and dynamical matrix
    n_cells = ph._supercell[0] * ph._supercell[1] * ph._supercell[2]
    n_atoms = len(ph.indices)
    assert ph.C_N.shape == (n_cells, 3 * n_atoms, 3 * n_atoms)
    assert ph.D_N.shape == (n_cells, 3 * n_atoms, 3 * n_atoms)


def test_dynamical_matrix_at_gamma(al_state, lj_calculator):
    """Test computation of dynamical matrix at gamma point (q=0)."""
    ph = Phonons(
        state=al_state,
        calculator=calculator_wrapper(lj_calculator),
        supercell=(1, 1, 1),
        delta=0.01,
    )

    # Run and read
    ph.run()
    ph.read(method="frederiksen", acoustic=True)

    # Calculate dynamical matrix at gamma point
    q_gamma = torch.zeros(3, dtype=torch.float64, device=al_state.device)
    D_gamma = ph._compute_q_dynamical_matrix(q_gamma)

    # Check that dynamical matrix at gamma is real
    assert torch.allclose(D_gamma.imag, torch.zeros_like(D_gamma.imag), atol=1e-10)

    # Check that dynamical matrix at gamma is symmetric
    assert torch.allclose(D_gamma, D_gamma.transpose(-1, -2), atol=1e-10)

    # Check that the eigenvalues at gamma include three zeros (acoustic modes)
    eigvals = torch.linalg.eigvalsh(D_gamma)
    assert torch.sum(torch.abs(eigvals) < 1e-10) >= 3


def test_phonon_band_structure(al_state, lj_calculator):
    """Test calculation of phonon band structure."""
    ph = Phonons(
        state=al_state,
        calculator=calculator_wrapper(lj_calculator),
        supercell=(2, 2, 2),
        delta=0.01,
    )

    # Run and read
    ph.run()
    ph.read(method="frederiksen", acoustic=True)

    # Define a simple path in the Brillouin zone
    path = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Gamma
            [0.5, 0.0, 0.0],  # X
            [0.5, 0.5, 0.0],  # M
            [0.0, 0.0, 0.0],  # Gamma
        ],
        dtype=torch.float64,
        device=al_state.device,
    )

    # Calculate band structure
    omega_kl = ph.band_structure(path, modes=False)

    # Check shape: (n_kpoints, 3*n_atoms)
    assert omega_kl.shape == (4, 3 * len(ph.indices))

    # Also test with modes=True
    omega_kl, u_kl = ph.band_structure(path, modes=True)

    # Check shapes
    assert omega_kl.shape == (4, 3 * len(ph.indices))
    assert u_kl.shape == (4, 3 * len(ph.indices), len(ph.indices), 3)


def test_compare_with_ase(si_atoms, si_state, device):
    """Compare torch-sim phonon results with ASE results."""
    # Skip test if EMT calculator is not available
    try:
        calculator = EMT()
    except ImportError:
        pytest.skip("EMT calculator not available")

    # Set up ASE phonons calculator
    ase_ph = ASEPhonons(si_atoms.copy(), calculator, supercell=(2, 2, 2), delta=0.01)
    ase_ph.run()
    ase_ph.read(acoustic=True)

    # Set up torch-sim calculator
    def torch_calculator(state):
        atoms = state.to_atoms()
        atoms.calc = calculator
        forces = atoms.get_forces()
        return {"forces": torch.tensor(forces, device=device, dtype=torch.float64)}

    torch_ph = Phonons(
        state=si_state,
        calculator=torch_calculator,
        supercell=(2, 2, 2),
        delta=0.01,
    )
    torch_ph.run()
    torch_ph.read(method="frederiksen", acoustic=True)

    # Calculate band structure at the same points
    path = np.array(
        [
            [0.0, 0.0, 0.0],  # Gamma
            [0.5, 0.0, 0.0],  # X
        ]
    )

    ase_bands = ase_ph.band_structure(path)
    torch_path = torch.tensor(path, device=device, dtype=torch.float64)
    torch_bands = torch_ph.band_structure(torch_path)

    # ASE results are in eV, convert to same units for comparison
    # Now both should be in the same units
    for i in range(len(path)):
        # Compare only the positive frequencies, sorting both arrays
        ase_pos = np.sort(ase_bands[i][ase_bands[i] > 0])
        torch_pos = torch.sort(torch_bands[i][torch_bands[i] > 0])[0].cpu().numpy()

        # Check that frequencies are close, allowing for some numerical differences
        assert np.allclose(ase_pos, torch_pos, rtol=0.1, atol=0.1)


def test_acoustic_sum_rule(si_state, lj_calculator):
    """Test that the acoustic sum rule is enforced correctly."""
    ph = Phonons(
        state=si_state,
        calculator=calculator_wrapper(lj_calculator),
        supercell=(2, 2, 2),
        delta=0.01,
    )

    # Run and read without acoustic sum rule
    ph.run()
    ph.read(method="frederiksen", acoustic=False)

    # Calculate dynamical matrix at gamma point
    q_gamma = torch.zeros(3, dtype=torch.float64, device=si_state.device)
    D_gamma_without_asr = ph._compute_q_dynamical_matrix(q_gamma)

    # Eigenvalues without ASR - should have non-zero acoustic modes
    eigvals_without_asr = torch.linalg.eigvalsh(D_gamma_without_asr)

    # Now read with acoustic sum rule
    ph.read(method="frederiksen", acoustic=True)
    D_gamma_with_asr = ph._compute_q_dynamical_matrix(q_gamma)

    # Eigenvalues with ASR - should have zero acoustic modes
    eigvals_with_asr = torch.linalg.eigvalsh(D_gamma_with_asr)

    # Check that the ASR correctly enforces zero acoustic modes
    assert torch.min(torch.abs(eigvals_without_asr)) > 1e-10
    assert torch.sum(torch.abs(eigvals_with_asr) < 1e-10) >= 3


def test_symmetrize(al_state, lj_calculator):
    """Test symmetrization of force constants."""
    ph = Phonons(
        state=al_state,
        calculator=calculator_wrapper(lj_calculator),
        supercell=(2, 2, 2),
        delta=0.01,
    )

    # Run and read without symmetrization
    ph.run()
    ph.read(method="frederiksen", symmetrize=0, acoustic=False)
    C_N_orig = ph.C_N.clone()

    # Apply symmetrization once
    C_N_sym = ph._symmetrize(C_N_orig.clone())

    # Check that symmetrized force constants obey C_ij = C_ji
    n_cells = ph._supercell[0] * ph._supercell[1] * ph._supercell[2]
    for k in range(n_cells):
        C_ij = C_N_sym[k]
        assert torch.allclose(C_ij, C_ij.T, atol=1e-10)


def test_cutoff(al_state, lj_calculator):
    """Test cutoff application to force constants."""
    ph = Phonons(
        state=al_state,
        calculator=calculator_wrapper(lj_calculator),
        supercell=(3, 3, 3),
        delta=0.01,
    )

    # Run and read
    ph.run()

    # Read without cutoff
    ph.read(method="frederiksen", symmetrize=1, acoustic=True, cutoff=None)
    C_N_without_cutoff = ph.C_N.clone()

    # Read with a small cutoff
    cutoff = 5.0  # Angstrom
    ph.read(method="frederiksen", symmetrize=1, acoustic=True, cutoff=cutoff)
    C_N_with_cutoff = ph.C_N.clone()

    # Check that the cutoff has been applied (some elements should be zero)
    assert not torch.allclose(C_N_without_cutoff, C_N_with_cutoff)

    # Check that the number of non-zero elements is smaller with cutoff
    non_zero_without_cutoff = torch.sum(torch.abs(C_N_without_cutoff) > 1e-10)
    non_zero_with_cutoff = torch.sum(torch.abs(C_N_with_cutoff) > 1e-10)
    assert non_zero_with_cutoff < non_zero_without_cutoff
