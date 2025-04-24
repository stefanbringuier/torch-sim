"""Nudged Elastic Band (NEB) implementation for transition state calculations.

This module implements the NEB method [jonsson1998]_ of Henkelman and Jónsson
[henkelman2000]_ for finding minimum energy paths and transition states between
atomic configurations.

References:
    .. [jonsson1998] H. Jónsson, G. Mills, K.W. Jacobsen, Nudged elastic band
        method for finding minimum energy paths of transitions, in: Classical and
        Quantum Dynamics in Condensed Phase Simulations, WORLD SCIENTIFIC, 1998:
        pp. 385-404. <https://doi.org/10.1142/9789812839664_0016>_.

    .. [henkelman2000] G. Henkelman, H. Jónsson, Improved tangent estimate in the
        nudged elastic band method for finding minimum energy paths and saddle
        points, The Journal of Chemical Physics 113 (2000) 9978-9985.
        <https://doi.org/10.1063/1.1323224>_.

"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.optimizers import (
    fire,
    frechet_cell_fire,
    gradient_descent,
    unit_cell_fire,
    unit_cell_gradient_descent,
)
from torch_sim.state import SimState
from torch_sim.trajectory import TrajectoryReporter
from torch_sim.transforms import get_pair_displacements


_EPS = torch.finfo(torch.float64).eps


@dataclass
class NEBState(SimState):
    """State class for Nudged Elastic Band optimization.

    Extends SimState to store NEB-specific attributes for tracking the
    evolution of atomic configurations along a reaction path. Each image
    represents a replica of the system at different points along the reaction
    coordinate.

    Attributes:
        images: Atomic configurations along the path [n_images, n_atoms, 3]
        energies: Energy of each image [n_images]
        forces: Forces on each image [n_images, n_atoms, 3]
        tangents: Local tangent vectors for each image [n_images, n_atoms, 3]
        spring_constants: Spring constants between images [n_images-1]
        fixed_mask: Boolean mask for fixed images (endpoints) [n_images]
    """

    images: torch.Tensor
    energies: torch.Tensor
    forces: torch.Tensor
    tangents: torch.Tensor
    spring_constants: torch.Tensor
    fixed_mask: torch.Tensor

    def __init__(
        self,
        positions: torch.Tensor,
        masses: torch.Tensor,
        cell: torch.Tensor,
        *,
        pbc: bool,
        atomic_numbers: torch.Tensor,
        images: torch.Tensor,
        energies: torch.Tensor,
        forces: torch.Tensor,
        tangents: torch.Tensor,
        spring_constants: torch.Tensor,
        fixed_mask: torch.Tensor,
    ) -> None:
        """Initialize NEBState.

        Args:
            positions: Base positions for SimState [n_atoms, 3]
            masses: Atomic masses [n_atoms]
            cell: Unit cell vectors [3, 3]
            pbc: Whether to use periodic boundary conditions
            atomic_numbers: Atomic numbers [n_atoms]
            images: NEB images [n_images, n_atoms, 3]
            energies: Image energies [n_images]
            forces: Image forces [n_images, n_atoms, 3]
            tangents: Image tangents [n_images, n_atoms, 3]
            spring_constants: Spring constants [n_images-1]
            fixed_mask: Fixed image mask [n_images]
        """
        super().__init__(positions, masses, cell, pbc, atomic_numbers)
        self.images = images
        self.energies = energies
        self.forces = forces
        self.tangents = tangents
        self.spring_constants = spring_constants
        self.fixed_mask = fixed_mask


def compute_tangents(  # noqa: C901
    images: torch.Tensor,
    energies: torch.Tensor,
    fixed_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute normalized tangent vectors for NEB images.

    Implements the improved tangent estimate of Henkelman and Jónsson
    [henkelman2000]_ to prevent kinking and corner-cutting artifacts in the
    reaction path.

    References:
        .. [henkelman2000] G. Henkelman, H. Jónsson, Improved tangent estimate
            in the nudged elastic band method for finding minimum energy
            paths and saddle points, The Journal of Chemical Physics 113
            (2000) 9978-9985. <https://doi.org/10.1063/1.1323224>_.

    Args:
        images: Atomic configurations [n_images, n_atoms, 3]
        energies: Energy of each image [n_images]
        fixed_mask: Boolean mask for fixed images [n_images]

    Returns:
        Local tangent vectors [n_images, n_atoms, 3], normalized
    """
    n_images = images.shape[0]
    tangents = torch.zeros_like(images)

    # Step 1: Directional vectors for consecutive images
    # R_{i+1} - R_i
    dR_forward = images[1:] - images[:-1]

    # Step 2: Energy differences
    # V_{i+1} - V_i
    dE_forward = energies[1:] - energies[:-1]

    # Tangents of non-fixed neighbors/images
    for i in range(1, n_images - 1):
        if fixed_mask[i]:
            continue

        dR_plus = dR_forward[i]  # R_{i+1} - R_i
        dR_minus = -dR_forward[i - 1]  # R_i - R_{i-1}

        dE_plus = dE_forward[i]  # V_{i+1} - V_i
        dE_minus = -dE_forward[i - 1]  # V_i - V_{i-1}

        # Step 3: Select based on profile
        # Case 1: Both neighbors at min
        if dE_plus > 0 and dE_minus > 0:
            # Weight higher-energy dir.
            abs_dE_plus = abs(dE_plus)
            abs_dE_minus = abs(dE_minus)

            if abs_dE_plus > abs_dE_minus:
                tangents[i] = dR_plus.clone()
            else:
                tangents[i] = dR_minus.clone()

        # Case 2: Both neighbors at max
        elif dE_plus < 0 and dE_minus < 0:
            # Weight lower-energy dir.
            abs_dE_plus = abs(dE_plus)
            abs_dE_minus = abs(dE_minus)

            # Catch! Equal slopes on max
            if torch.isclose(abs_dE_plus, abs_dE_minus, rtol=1e-6):
                # NOTE: When symm. energy max has equal slopes,
                # Need to bisect the angle between downhill dir.
                # At max both segments point downhill from image i

                hi_plus = dR_plus.clone()
                hi_minus = dR_minus.clone()

                # NOTE: When path symmetric path bisector is normalized
                # sum of unit vectors. Hence Normalize each direction
                norm_plus = torch.norm(hi_plus, dim=1, keepdim=True)
                mask_plus = norm_plus > _EPS
                hi_plus = torch.where(mask_plus, hi_plus / norm_plus, hi_plus)

                norm_minus = torch.norm(hi_minus, dim=1, keepdim=True)
                mask_minus = norm_minus > _EPS
                hi_minus = torch.where(mask_minus, hi_minus / norm_minus, hi_minus)

                # Bisect the angle, flip sign to get dir.
                tangents[i] = -(hi_plus + hi_minus)
            elif abs_dE_plus < abs_dE_minus:
                tangents[i] = dR_plus.clone()
            else:
                tangents[i] = dR_minus.clone()

        # Case 3: Energy is high i+1, low i-1 (uphill)
        elif dE_plus > 0 and dE_minus < 0:
            tangents[i] = dR_plus.clone()

        # Case 4: Energy is low i+1, high at i-1 (downhill)
        elif dE_plus < 0 and dE_minus > 0:
            tangents[i] = dR_minus.clone()

        # Case 5: Fallback, Plateau scenario (shouldn't happen!)
        elif abs(dE_plus) > abs(dE_minus):
            tangents[i] = dR_plus.clone()
        else:
            tangents[i] = dR_minus.clone()

        # Step 4: Normalize non-fixed neighbors/images
        norm = torch.norm(tangents[i], dim=1, keepdim=True)
        mask = norm > _EPS
        tangents[i] = torch.where(mask, tangents[i] / norm, tangents[i])

    return tangents


def compute_spring_forces(
    images: torch.Tensor,
    spring_constants: torch.Tensor,
    fixed_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute spring forces between images.

    Harmonic spring between adjacent images to maintain
    approximately equal spacing along the reaction path.
    Spring forces are only applied to non-fixed images.

    Args:
        images: Atomic configurations [n_images, n_atoms, 3]
        spring_constants: Spring constants between images [n_images-1]
        fixed_mask: Boolean mask for fixed images [n_images]

    Returns:
        Spring forces [n_images, n_atoms, 3]
    """
    n_images = images.shape[0]
    spring_forces = torch.zeros_like(images)

    # Distances and norms
    forward_diff = images[1:] - images[:-1]  # [n_images-1, n_atoms, 3]
    forward_norms = torch.norm(
        forward_diff, dim=2, keepdim=True
    )  # [n_images-1, n_atoms, 1]

    # Unit vectors for each displacement
    mask = forward_norms > _EPS
    forward_unit = torch.where(
        mask, forward_diff / forward_norms, torch.zeros_like(forward_diff)
    )

    # Spring forces non-fixed images
    for i in range(1, n_images - 1):
        if fixed_mask[i]:
            continue

        # Forward (i -> i+1) w/ unit vector
        k_forward = spring_constants[i]
        f_forward = k_forward * forward_unit[i] * forward_norms[i]

        # Backward (i -> i-1) w/ unit vector
        k_backward = spring_constants[i - 1]
        f_backward = -k_backward * forward_unit[i - 1] * forward_norms[i - 1]

        spring_forces[i] = f_forward + f_backward

    return spring_forces


def project_forces(
    forces: torch.Tensor,
    tangents: torch.Tensor,
) -> torch.Tensor:
    """Project forces perpendicular to the path.

    NEB method only uses the component of the force perpendicular
    to the path tangent. Prevents true force from interfering
    with spring forces to maintain the spacing.

    Args:
        forces: Forces [n_images?, n_atoms, 3]
        tangents: Normalized tangents [n_images?, n_atoms, 3]

    Returns:
        Forces projected perpendicular to tangents [n_images?, n_atoms, 3]
    """
    # TODO: Handle both batched and unbatched inputs
    if forces.dim() == 2:
        forces = forces.unsqueeze(0)
        tangents = tangents.unsqueeze(0)
        unbatched = True
    else:
        unbatched = False

    # F·t [n_images, n_atoms, 1]
    f_dot_t = torch.sum(forces * tangents, dim=2, keepdim=True)

    # F_perp = F - (F·t)t
    forces_perp = forces - f_dot_t * tangents

    return forces_perp[0] if unbatched else forces_perp


def interpolate_path(
    initial_state: SimState,
    final_state: SimState,
    n_images: int,
) -> torch.Tensor:
    """Generate initial NEB path by linear interpolation.

    Sequence of images between initial and final states using
    linear interpolation of atomic positions. Handle PBC appropriately.

    Args:
        initial_state: Initial atomic configuration
        final_state: Final atomic configuration
        n_images: Number of images along the path

    Returns:
        Interpolated atomic configurations [n_images, n_atoms, 3]
    """
    pos_i = initial_state.positions
    pos_f = final_state.positions

    if initial_state.pbc:
        # Ensure cell is 2D before calculating displacements
        cell = initial_state.cell
        if cell.ndim == 3 and cell.shape[0] == 1:
            cell = cell.squeeze(0)

        # Apply MIC
        dr, _ = get_pair_displacements(
            positions=pos_i,
            cell=cell,
            pbc=True,
            pairs=(
                torch.zeros(pos_i.shape[0], dtype=torch.long),
                torch.arange(pos_i.shape[0], dtype=torch.long),
            ),
        )
        pos_f = pos_i + dr

    # Interpolate
    t = torch.linspace(0, 1, n_images, device=pos_i.device)
    return pos_i.unsqueeze(0) + t.unsqueeze(1).unsqueeze(2) * (pos_f - pos_i).unsqueeze(0)


def _get_optimizer_functions(
    optimizer_name: str, model: ModelInterface, optimizer_kwargs: dict
) -> tuple[Callable, Callable]:
    """Get optimizer functions."""
    optimizer_factories = {
        "fire": fire,
        "gd": gradient_descent,
        "unit_cell_fire": unit_cell_fire,
        "unit_cell_gd": unit_cell_gradient_descent,
        "frechet_cell_fire": frechet_cell_fire,
    }

    if optimizer_name not in optimizer_factories:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Cast model to torch.nn.Module?
    factory = optimizer_factories[optimizer_name]
    opt_init, opt_update = factory(
        model=cast("torch.nn.Module", model), **optimizer_kwargs
    )
    return opt_init, opt_update


def neb_optimize(
    initial_state: SimState,
    final_state: SimState,
    model: ModelInterface,
    *,
    n_images: int = 7,
    spring_constant: float = 1.0,
    optimizer: str = "fire",
    optimizer_kwargs: dict | None = None,
    convergence_fn: Callable | None = None,
    trajectory_reporter: TrajectoryReporter | None = None,
    max_steps: int = 1000,
    climbing_image: bool = True,
    climbing_steps: int = 100,
) -> NEBState:
    """Run NEB optimization between two states.

    Implements the Nudged Elastic Band method to find the minimum energy path
    between two atomic configurations. Uses the improved tangent estimate and
    supports multiple optimization algorithms for efficient convergence.
    Optionally enables the climbing image NEB (CI-NEB) for more accurate
    transition state determination.

    Args:
        initial_state: Initial atomic configuration.
        final_state: Final atomic configuration.
        model: Energy/force model.
        n_images: Number of images along the path.
        spring_constant: Spring constant between images.
        optimizer: Optimization algorithm to use. Options:

            - "fire": Fast Inertial Relaxation Engine (default)
            - "gd": Gradient descent
            - "unit_cell_fire": FIRE with unit cell optimization
            - "unit_cell_gd": Gradient descent with unit cell optimization
            - "frechet_cell_fire": FIRE with Frechet cell parameterization

        optimizer_kwargs: Additional keyword arguments for the optimizer.
        convergence_fn: Function to check convergence.
        trajectory_reporter: For saving trajectory data.
        max_steps: Maximum optimization steps.
        climbing_image: Whether to enable climbing image NEB.
        climbing_steps: Number of steps before enabling climbing image.

    Returns:
        Optimized NEB path.
    """
    device = initial_state.positions.device
    dtype = initial_state.positions.dtype

    # Interpolate path
    images = interpolate_path(initial_state, final_state, n_images)

    # Fixed endpoints
    fixed_mask = torch.zeros(n_images, dtype=torch.bool, device=device)
    fixed_mask[0] = True
    fixed_mask[-1] = True

    spring_constants = torch.full(
        (n_images - 1,), spring_constant, device=device, dtype=dtype
    )

    neb_state = NEBState(
        positions=initial_state.positions,
        masses=initial_state.masses,
        cell=initial_state.cell,
        pbc=initial_state.pbc,
        atomic_numbers=initial_state.atomic_numbers,
        images=images,
        energies=torch.zeros(n_images, device=device, dtype=dtype),
        forces=torch.zeros_like(images),
        tangents=torch.zeros_like(images),
        spring_constants=spring_constants,
        fixed_mask=fixed_mask,
    )

    optimizer_kwargs = optimizer_kwargs or {}

    # Get optimizer functions once
    try:
        opt_init, opt_update = _get_optimizer_functions(
            optimizer, model, optimizer_kwargs
        )
    except ValueError as e:
        raise ValueError(f"Optimizer initialization failed: {e}") from e

    # Initialize image optimizer states, skip fixed endpoints
    opt_states = []
    for i in range(1, n_images - 1):
        state = SimState(
            positions=neb_state.images[i],
            masses=initial_state.masses,
            cell=initial_state.cell,
            pbc=initial_state.pbc,
            atomic_numbers=initial_state.atomic_numbers,
        )
        opt_states.append(opt_init(state))  # type: ignore[operator]

    for step in range(max_steps):
        for i in range(n_images):
            state = SimState(
                positions=neb_state.images[i],
                masses=initial_state.masses,
                cell=initial_state.cell,
                pbc=initial_state.pbc,
                atomic_numbers=initial_state.atomic_numbers,
            )

            # Use the forward method defined by ModelInterface.
            results = model.forward(state)
            neb_state.energies[i] = results["energy"]
            neb_state.forces[i] = results["forces"]

        neb_state.tangents = compute_tangents(
            neb_state.images,
            neb_state.energies,
            neb_state.fixed_mask,
        )

        # Highest energy image
        # Ensure index is an integer
        max_energy_idx_tensor = torch.argmax(neb_state.energies[1:-1]) + 1
        max_energy_idx = int(max_energy_idx_tensor.item())

        # NEB force projections
        forces_perp = project_forces(neb_state.forces, neb_state.tangents)

        spring_forces = compute_spring_forces(
            neb_state.images,
            neb_state.spring_constants,
            neb_state.fixed_mask,
        )

        # NEB forces: projected forces + spring forces
        neb_state.forces = forces_perp.clone()

        # Highest energy image modified. Removes spring forces
        # and inverts parallel component. This makes the image
        # climb up to the saddle point.
        if climbing_image and step >= climbing_steps:
            i = max_energy_idx
            if not fixed_mask[i]:
                parallel_force = (
                    torch.sum(
                        neb_state.forces[i] * neb_state.tangents[i],
                        dim=1,
                        keepdim=True,
                    )
                    * neb_state.tangents[i]
                )

                # Climbing image: F = F - 2(F·τ)τ
                neb_state.forces[i] = neb_state.forces[i] - 2 * parallel_force

                # Zero climbing image
                spring_forces[i] = 0

        # Non-climbing images
        neb_state.forces = neb_state.forces + spring_forces

        # Updates
        for i, opt_state in enumerate(opt_states, start=1):
            opt_state.positions = neb_state.images[i]
            opt_state.forces = neb_state.forces[i]

            opt_state = opt_update(opt_state)  # type: ignore[operator]
            neb_state.images[i] = opt_state.positions

        if trajectory_reporter is not None:
            trajectory_reporter.report(
                neb_state, step, model=cast("torch.nn.Module", model)
            )

        if convergence_fn is not None and convergence_fn(neb_state):
            break

    return neb_state
