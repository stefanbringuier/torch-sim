"""Tests for the Nudged Elastic Band (NEB) implementation.

This module contains tests for the NEB implementation using simple Lennard-Jones
dimer systems where we can analytically verify the minimum energy paths and
transition states.

NOTE: The current tests cover basic component functionality (tangents, spring
forces, projection) and an integration test on a path where the energy
monotonically decreases from the start. A test case specifically designed
to verify finding an intermediate saddle point (i.e., a path with an energy
barrier between endpoints, requiring the climbing image logic) is not yet
implemented.
"""

import torch

from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.neb import (
    compute_spring_forces,
    compute_tangents,
    interpolate_path,
    neb_optimize,
    project_forces,
)
from torch_sim.state import SimState


def create_lj_dimer(
    device: torch.device,
    dtype: torch.dtype,
    distance: float,
) -> SimState:
    """Create a Lennard-Jones dimer with specified interatomic distance.

    Args:
        device: Device to place tensors on
        dtype: Data type for tensors
        distance: Interatomic distance in Angstroms

    Returns:
        SimState containing the dimer configuration
    """
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
        device=device,
        dtype=dtype,
    )

    # Create cell as a global property with proper shape
    cell = torch.eye(3, device=device, dtype=dtype) * 10.0
    cell = cell.unsqueeze(0)  # Add batch dimension [1, 3, 3]

    return SimState(
        positions=positions,
        masses=torch.ones(2, device=device, dtype=dtype),
        cell=cell,
        pbc=False,
        atomic_numbers=torch.ones(2, device=device, dtype=torch.long),
    )


def test_compute_tangents():
    """Test tangent computation for NEB."""
    # Create a simple 1D energy profile with a maximum
    energies = torch.tensor([1.0, 2.0, 1.0])
    # Reshape images to [n_images, n_atoms, n_dims] = [3, 1, 1]
    images = torch.stack(
        [torch.tensor([[0.0]]), torch.tensor([[1.0]]), torch.tensor([[2.0]])]
    )

    # Create fixed mask (endpoints fixed)
    fixed_mask = torch.tensor([True, False, True])

    # Compute tangents
    tangents = compute_tangents(images, energies, fixed_mask)

    # Tangents should have the same shape as images
    assert tangents.shape == images.shape

    # For the middle image (maximum), tangent should be zero
    # Access shape [1, 1]
    assert torch.allclose(tangents[1], torch.zeros_like(tangents[1]))

    # Endpoints tangents are expected to be zero as they are fixed
    assert torch.allclose(tangents[0], torch.zeros_like(tangents[0]))
    assert torch.allclose(tangents[2], torch.zeros_like(tangents[2]))


def test_compute_spring_forces():
    """Test spring force computation."""
    device = torch.device("cpu")
    dtype = torch.float64

    # Create a simple path with 3 images
    images = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # Image 1
            [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],  # Image 2
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],  # Image 3
        ],
        device=device,
        dtype=dtype,
    )

    # Create spring constants and fixed mask
    spring_constants = torch.tensor([0.1, 0.1], device=device, dtype=dtype)
    fixed_mask = torch.tensor([True, False, True], device=device)

    # Compute spring forces
    spring_forces = compute_spring_forces(images, spring_constants, fixed_mask)

    # Check spring force properties
    assert spring_forces.shape == images.shape
    assert torch.allclose(
        spring_forces[0], torch.zeros_like(spring_forces[0])
    )  # Fixed image
    assert torch.allclose(
        spring_forces[2], torch.zeros_like(spring_forces[2])
    )  # Fixed image

    # Middle image should have spring forces pointing towards equilibrium
    # Forces on each atom are calculated based on unit vectors
    # dR1 = images[1] - images[0] = [[0.5, 0, 0], [0.5, 0, 0]], norm = 0.5
    # dR2 = images[2] - images[1] = [[0.5, 0, 0], [0.5, 0, 0]], norm = 0.5
    # unit1 = [[1, 0, 0], [1, 0, 0]]
    # unit2 = [[1, 0, 0], [1, 0, 0]]
    # F_spring(1) = k1 * unit2 * norm2 - k0 * unit1 * norm1
    # F_spring(1) = 0.1 * [[1,0,0],[1,0,0]]*0.5 - 0.1 * [[1,0,0],[1,0,0]]*0.5 = 0
    expected_force = torch.zeros_like(images[1])
    assert torch.allclose(spring_forces[1], expected_force, atol=1e-6)


def test_project_forces():
    """Test force projection."""
    device = torch.device("cpu")
    dtype = torch.float64

    # Create forces and tangents (unbatched case)
    forces = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], device=device, dtype=dtype)
    tangents = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device, dtype=dtype
    )

    # Project forces
    forces_perp = project_forces(forces, tangents)

    # Check that projected forces are perpendicular to tangents
    dot_products = torch.sum(forces_perp * tangents, dim=1)
    assert torch.allclose(dot_products, torch.zeros_like(dot_products))

    # Check that projection is correct
    expected_forces = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype
    )
    assert torch.allclose(forces_perp, expected_forces)

    # Test batched case
    batched_forces = forces.unsqueeze(0)  # [1, 2, 3]
    batched_tangents = tangents.unsqueeze(0)  # [1, 2, 3]
    batched_forces_perp = project_forces(batched_forces, batched_tangents)
    assert torch.allclose(batched_forces_perp[0], expected_forces)


def test_interpolate_path():
    """Test path interpolation."""
    device = torch.device("cpu")
    dtype = torch.float64

    # Create initial and final states
    initial_state = create_lj_dimer(device, dtype, 1.0)
    final_state = create_lj_dimer(device, dtype, 2.0)

    # Interpolate path
    images = interpolate_path(initial_state, final_state, n_images=3)

    # Check interpolation properties
    assert images.shape == (3, 2, 3)
    assert torch.allclose(images[0], initial_state.positions)  # First image
    assert torch.allclose(images[2], final_state.positions)  # Last image

    # Middle image should be halfway between
    expected_middle = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], device=device, dtype=dtype
    )
    assert torch.allclose(images[1], expected_middle)


def test_lj_dimer_neb_integration(lj_model: LennardJonesModel):
    """Integration test for NEB optimization on a Lennard-Jones dimer.

    This test verifies that the NEB components work together for a simple
    physical system. It runs a short optimization and checks for basic
    path properties. For this specific setup (compressed to stretched
    LJ dimer), the initial compressed state is expected to be the highest
    energy point.

    Note: This is an integration test, not a precise unit test.
    Tolerances are loose, and it mainly checks that the optimization runs
    and produces a physically plausible path outcome for this specific
    energy landscape.

    Physical parameters:
    - sigma = 3.405 Å (typical for Ar)
    - Initial state: r = 0.85 * sigma ≈ 2.89 Å (compressed, highest energy point)
    - Final state: r = 1.5 * sigma ≈ 5.11 Å (stretched, near zero energy)
    - Potential Minimum: r = 2^(1/6) * sigma ≈ 3.82 Å (not the path maximum)
    """
    device = lj_model.device
    dtype = lj_model.dtype

    # LJ parameters
    sigma = 3.405  # Å
    r_min = 0.85 * sigma  # Compressed state
    r_max = 1.5 * sigma  # Stretched state
    # r_potential_min = 2**(1/6) * sigma  # Potential minimum distance

    # Create initial state (compressed dimer)
    initial_state = create_lj_dimer(device, dtype, r_min)

    # Create final state (stretched dimer)
    final_state = create_lj_dimer(device, dtype, r_max)

    # Run NEB optimization - shorter run for integration test
    neb_state = neb_optimize(
        initial_state=initial_state,
        final_state=final_state,
        model=lj_model,
        n_images=9,  # Fewer images for faster test
        spring_constant=0.1,  # Moderate spring constant
        optimizer="fire",
        optimizer_kwargs={
            "dt_start": 0.01,
            "dt_max": 0.1,  # Conservative timestep
            "alpha_start": 0.1,
            "f_alpha": 0.99,
            "n_min": 5,
            "f_inc": 1.1,
            "f_dec": 0.5,
        },
        max_steps=500,  # Fewer steps for faster test
        climbing_image=True,
        # Activate climbing early, though it shouldn't affect endpoint max
        climbing_steps=100,
    )

    # --- Basic Checks for Integration Test ---

    assert neb_state.energies.shape == (9,)
    assert neb_state.images.shape == (9, 2, 3)
    assert neb_state.forces.shape == (9, 2, 3)
    assert neb_state.tangents.shape == (9, 2, 3)

    # Check endpoints are fixed
    assert torch.allclose(neb_state.images[0], initial_state.positions, atol=1e-6)
    assert torch.allclose(neb_state.images[-1], final_state.positions, atol=1e-6)

    # Find the highest energy image
    ts_idx = torch.argmax(neb_state.energies)
    ts_distance = torch.norm(
        neb_state.images[ts_idx, 1] - neb_state.images[ts_idx, 0]
    ).item()

    # For this system, the initial compressed state (index 0) should be highest E
    print(
        f"[Integration Test] Highest E image index: {ts_idx}, "
        f"Distance: {ts_distance:.3f} (Expected near {r_min:.3f})"
    )
    assert ts_idx == 0
    # Check the distance is indeed close to the initial distance
    assert abs(ts_distance - r_min) < 1e-3

    # Check basic energy profile: Highest energy (initial state)
    # should be greater than the final state energy.
    assert neb_state.energies[ts_idx] > neb_state.energies[-1]

    # Check forces on non-fixed images are reasonably small (some relaxation)
    max_force_norm = torch.max(torch.norm(neb_state.forces[1:-1], dim=(1, 2)))
    # Check that forces aren't excessively large either (indicates instability)
    # Allow somewhat larger forces as it might not fully converge in 500 steps
    print(f"[Integration Test] Max force norm on internal images: {max_force_norm:.4f}")
    assert max_force_norm < 10.0  # Relaxed tolerance from 100

    # Don't check for fine-grained spacing or force projection
