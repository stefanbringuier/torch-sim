"""Phonon calculations using the finite-displacement method.

Inspired by the approach in ase.phonons module of Atomic Simulation Env.

References:
- A. H. Larsen, et al.,J. Phys.: Condens. Matter Vol. 29, 273002 (2017)
- D. Alfe, Comput. Phys. Commun. 180, 2622 (2009)
- Y. Wang et al., J. Phys.: Cond. Matter 22, 202201 (2010)

"""

__author__ = "Stefan Bringuier"
__email__ = "stefan.bringuier@gmail.com"

import math
from typing import Callable, Optional, Union, Tuple, List

import torch
from torch_sim.state import SimState


class AtomicDisplacement:
    """Base class for finite displacement calculations in a periodic system.

    This class provides functionality to create and manage atomic displacements
    for calculating numerical derivatives in periodic systems. It handles the
    creation of supercells, application of displacements, and caching of results.

    Attributes:
        state: SimState object containing the atomic configuration
        calculator: Function to compute forces for displaced configurations
        supercell: Tuple of integers specifying the supercell size (nx, ny, nz)
        name: Base name for saved files
        delta: Magnitude of displacement in Angstrom
        indices: Indices of atoms to displace
        cache: Dictionary to store results for each displacement
        device: Device where tensors are stored
        dtype: Data type of tensors
    """

    def __init__(
        self,
        state: SimState,
        calculator: Optional[Callable] = None,
        supercell: Tuple[int, int, int] = (1, 1, 1),
        name: Optional[str] = None,
        delta: float = 0.01,
        indices: Optional[torch.Tensor] = None,
        center_cell: bool = False,
    ):
        """Initialize the AtomicDisplacement class.

        Args:
            state: SimState object containing the atomic configuration
            calculator: Function that takes a SimState and returns forces
            supercell: Size of supercell as (nx, ny, nz) repetitions of unit cell
            name: Base name for saved results
            delta: Magnitude of displacement in Angstrom
            indices: Indices of atoms to displace, if None all atoms are displaced
            center_cell: If True, use the center cell of the supercell as reference
        """
        self.state = state
        self.calculator = calculator
        self._supercell = supercell
        self.name = name if name is not None else "displacement"
        self.delta = delta
        self.center_cell = center_cell

        if indices is None:
            self.indices = torch.arange(state.n_atoms, device=state.device)
        else:
            self.indices = indices.to(device=state.device)

        # Store number of atoms from indices to avoid repeated calculations
        self._natoms = len(self.indices)
        self._n_cells = self._supercell[0] * self._supercell[1] * self._supercell[2]

        self.device = state.device
        self.dtype = state.dtype

        # Cache for storing results (in memory, GPU-friendly)
        self.cache: dict[str, dict[str, torch.Tensor]] = {}

        # Compute reference cell offset
        self.offset = self._compute_offset()
        self.lattice_vectors = self._compute_lattice_vectors()

    def _compute_offset(self) -> int:
        """Compute the offset index of the reference cell in the supercell.

        Returns:
            int: Offset index of the reference cell
        """
        if not self.center_cell:
            # Origin cell
            return 0
        else:
            # Center cell
            nx, ny, nz = self._supercell
            return (nx // 2) * (ny * nz) + (ny // 2) * nz + (nz // 2)

    def _compute_lattice_vectors(self) -> torch.Tensor:
        """Compute the lattice vectors for cells in the supercell.

        Returns:
            torch.Tensor: Lattice vectors relative to the reference cell
        """
        nx, ny, nz = self._supercell

        # Cell index triplets (i, j, k)
        i = torch.arange(nx, device=self.device).repeat_interleave(ny * nz)
        j = torch.arange(ny, device=self.device).repeat(nx).repeat_interleave(nz)
        k = torch.arange(nz, device=self.device).repeat(nx * ny)

        ijk = torch.stack([i, j, k], dim=1)

        # Center vectors
        supercell_tensor = torch.tensor(self._supercell, device=self.device)
        if self.center_cell:
            ijk = ijk - supercell_tensor // 2
        else:
            ijk = ijk - supercell_tensor // 2
            ijk = ijk % supercell_tensor

        return ijk

    def _create_supercell(self, state: Optional[SimState] = None) -> SimState:
        """Create a supercell from the unit cell.

        Args:
            state: Optional SimState to use instead of self.state

        Returns:
            SimState: Supercell configuration
        """
        # Use provided state or default to self.state
        if state is None:
            state = self.state

        n_atoms = state.n_atoms

        # Create new positions array for all atoms in supercell
        positions = torch.zeros(
            (n_atoms * self._n_cells, 3), dtype=self.dtype, device=self.device
        )

        # Create arrays for atomic properties
        atomic_numbers = torch.zeros(
            n_atoms * self._n_cells,
            dtype=state.atomic_numbers.dtype,
            device=self.device,
        )
        masses = torch.zeros(
            n_atoms * self._n_cells, dtype=self.dtype, device=self.device
        )
        batch = torch.zeros(
            n_atoms * self._n_cells, dtype=torch.int64, device=self.device
        )

        cell_matrix = state.cell.squeeze()

        # Position each atom in respective supercell
        for i, (x, y, z) in enumerate(self.lattice_vectors):
            cell_pos = x * cell_matrix[0] + y * cell_matrix[1] + z * cell_matrix[2]

            # Add atoms to this cell
            start_idx = i * n_atoms
            end_idx = (i + 1) * n_atoms
            positions[start_idx:end_idx] = state.positions + cell_pos
            atomic_numbers[start_idx:end_idx] = state.atomic_numbers
            masses[start_idx:end_idx] = state.masses
            if state.batch is not None:
                batch[start_idx:end_idx] = state.batch

        supercell_cell = cell_matrix * torch.tensor(
            self._supercell, dtype=self.dtype, device=self.device
        ).unsqueeze(1)

        # Supercell SimState
        return SimState(
            positions=positions,
            masses=masses,
            cell=supercell_cell,
            pbc=state.pbc,
            atomic_numbers=atomic_numbers,
            batch=batch,
        )

    def _displace_atom(
        self, supercell: SimState, atom_index: int, direction: int, sign: int
    ) -> SimState:
        """Displace an atom in the supercell.

        Args:
            supercell: SimState containing the supercell
            atom_index: Index of atom to displace in the unit cell
            direction: Direction of displacement (0=x, 1=y, 2=z)
            sign: Sign of displacement (1 or -1)

        Returns:
            SimState: Supercell with displaced atom
        """
        displaced = supercell.clone()
        absolute_index = self.offset * self._natoms + atom_index
        displaced.positions[absolute_index, direction] += sign * self.delta

        return displaced

    def _get_displacement_name(self, atom_index: int, direction: int, sign: int) -> str:
        """Get a name for the displacement.

        Args:
            atom_index: Index of atom being displaced
            direction: Direction of displacement (0=x, 1=y, 2=z)
            sign: Sign of displacement (1 or -1)

        Returns:
            str: Name for the displacement
        """
        direction_name = ["x", "y", "z"][direction]
        sign_name = "+" if sign > 0 else "-"
        return f"{self.name}.{atom_index}{direction_name}{sign_name}"

    def __call__(self, supercell: SimState) -> torch.Tensor:
        """Method to be overridden by derived classes.

        Args:
            supercell: SimState containing the supercell

        Returns:
            torch.Tensor: Calculated property for the supercell
        """
        raise NotImplementedError("Implement this method in derived classes")


class Phonons(AtomicDisplacement):
    """Class for calculating phonon modes using the finite displacement method.

    This class computes the force constants matrix from finite difference
    approximation to the forces. It can then calculate the dynamical matrix,
    phonon frequencies, and phonon band structure.

    Attributes:
        C_N: Force constants matrix in real space (N, 3*natoms, 3*natoms)
        D_N: Dynamical matrix in real space (N, 3*natoms, 3*natoms)
    """

    def __init__(
        self,
        state: SimState,
        calculator: Optional[Callable] = None,
        supercell: Tuple[int, int, int] = (3, 3, 3),
        name: Optional[str] = None,
        delta: float = 0.01,
        indices: Optional[torch.Tensor] = None,
        center_cell: bool = True,
    ):
        """Initialize the Phonons class.

        Args:
            state: SimState object containing the atomic configuration
            calculator: Function that takes a SimState and returns forces
            supercell: Size of supercell as (nx, ny, nz) repetitions of unit cell
            name: Base name for saved results
            delta: Magnitude of displacement in Angstrom
            indices: Indices of atoms to displace, if None all atoms are displaced
            center_cell: If True, use the center cell of the supercell as reference.
                This is recommended for better numerical stability as it ensures the
                displaced atom has a more symmetric environment, reducing finite-size
                effects in the computed force constants.
        """
        # Initialize parent class
        super().__init__(
            state=state,
            calculator=calculator,
            supercell=supercell,
            name=name if name is not None else "phonon",
            delta=delta,
            indices=indices,
            center_cell=center_cell,
        )

        self.C_N: Optional[torch.Tensor] = None  # Force constants
        self.D_N: Optional[torch.Tensor] = None  # Dynamical matrix
        self.m_inv_x: Optional[torch.Tensor] = None  # Inverse mass vector

    def __call__(self, supercell: SimState) -> torch.Tensor:
        """Calculate forces on atoms in the supercell.

        This method prepares the state and calls the calculator. It ensures
        the forces are properly extracted and returned as a tensor.

        Args:
            supercell: SimState containing the supercell

        Returns:
            torch.Tensor: Forces on atoms as a tensor
        """
        if self.calculator is None:
            raise ValueError("Calculator must be set.")

        # Clone state and reshape
        state_copy = supercell.clone()
        if state_copy.cell.dim() == 2:  # [3, 3] shape
            state_copy.cell = state_copy.cell.unsqueeze(0)  # [1, 3, 3]

        results = self.calculator(state_copy)

        # Extract forces as tensor
        if isinstance(results, dict) and "forces" in results:
            forces = results["forces"]
        else:
            forces = results  # Assumes forces only

        # Keep forces on correct device
        if not isinstance(forces, torch.Tensor):
            forces = torch.tensor(forces, dtype=self.dtype, device=self.device)
        elif forces.device != self.device or forces.dtype != self.dtype:
            forces = forces.to(device=self.device, dtype=self.dtype)

        return forces

    def _check_eq_forces(
        self,
    ) -> Tuple[float, float, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Check maximum size of forces in the equilibrium structure.

        Returns:
            Tuple containing:
                - Minimum force component
                - Maximum force component
                - Indices of minimum force component (atom, direction)
                - Indices of maximum force component (atom, direction)
        """
        # Get equilibrium forces
        eq_name = f"{self.name}.eq"
        if eq_name not in self.cache:
            raise ValueError(
                "Equilibrium forces not computed yet. Run the calculation first."
            )

        feq_av = self.cache[eq_name]["forces"]

        fmin = torch.min(feq_av)
        fmax = torch.max(feq_av)
        i_min = torch.where(feq_av == fmin)
        i_max = torch.where(feq_av == fmax)

        return fmin.item(), fmax.item(), i_min, i_max

    def _symmetrize(self, C_N: torch.Tensor) -> torch.Tensor:
        """Symmetrize force constant matrix.

        .. warning::
            The numerical implementation may not be correct. The current approach
            of flipping and permuting tensors needs verification.

        Makes the force constants symmetric in indices C_ij = C_ji, which
        is required by Newton's third law.
            1. Reshape to cell indices
            2. Shift tensor by flipping all spatial dimensions
            3. Compute half-sums
            4. Reshape back to original form

        Args:
            C_N: Force constants matrix (n_cells, 3*natoms, 3*natoms)

        Returns:
            torch.Tensor: Symmetrized force constants
        """
        # TODO: Verify this implementation
        shape = self._supercell + (3 * self._natoms, 3 * self._natoms)
        C_ijk = C_N.reshape(shape)
        C_ijk_flipped = C_ijk.flip(dims=(0, 1, 2)).permute(0, 1, 2, 4, 3)
        C_ijk_sym = 0.5 * (C_ijk + C_ijk_flipped)

        return C_ijk_sym.reshape((self._n_cells, 3 * self._natoms, 3 * self._natoms))

    def _apply_acoustic(self, C_N: torch.Tensor):
        """Apply acoustic sum rule on force constants.

        .. warning::
            The numerical implementation may not be correct. The current approach
            of zeroing the central cell and subtracting force constants from other
            cells needs verification against established methods.

        The acoustic sum rule requires that the sum of forces on all atoms vanishes
        for any rigid displacement of the crystal.
            1. Zero central cell
            2. Sum over all cells
            3. Subtract the force constants

        Args:
            C_N: Force constants matrix (n_cells, 3*natoms, 3*natoms)
        """
        # Safety copy
        C_N_temp = C_N.clone()

        # Loop through all atoms and directions
        for a in range(self._natoms):
            for i in range(3):  # x, y, z
                idx = 3 * a + i

                # TODO: Verify this implementation
                C_N[self.offset, idx, :] = 0.0
                for R in range(len(C_N)):
                    if R != self.offset:  # Skip reference cell
                        C_N[self.offset, idx, :] -= C_N_temp[R, idx, :]

    def _apply_cutoff(self, C_N: torch.Tensor, r_c: float):
        """Zero elements for interatomic distances larger than the cutoff.
        This helps to avoid spurious forces at large distances.

        Args:
            C_N: Force constants matrix (n_cells, 3*natoms, 3*natoms)
            r_c: Cutoff distance in Angstrom
        """
        n_cells = len(C_N)
        C_Navav = C_N.reshape((n_cells, self._natoms, 3, self._natoms, 3))

        # Cell vectors and atomic positions
        cell_vc = self.state.cell.squeeze().T
        pos_av = self.state.positions[self.indices]

        # Cell lattice vectors
        R_cN = self.lattice_vectors

        # Loop over all cells
        for n in range(n_cells):
            # Lattice vector to cell
            R_v = torch.matmul(cell_vc, R_cN[n].float())
            posn_av = pos_av + R_v

            for i, a in enumerate(self.indices):
                pos_a = self.state.positions[a]
                dist_a = torch.sqrt(torch.sum((pos_a - posn_av) ** 2, dim=1))

                mask = dist_a > r_c
                if torch.any(mask):
                    for j in torch.where(mask)[0]:
                        C_Navav[n, i, :, j, :] = 0.0

        C_N.copy_(C_Navav.reshape((n_cells, 3 * self._natoms, 3 * self._natoms)))

        return C_N

    def _calculate_force_constants(
        self,
        symm_factor: int = 10,
        acoustic: bool = True,
        cutoff: Optional[float] = None,
    ):
        """Calculate force constants.

        Args:
            symm_factor: Number of symmetrization iterations to perform
            acoustic: If True, restore acoustic sum rule
            cutoff: Distance cutoff for force constants (None for no cutoff)
        """

        # Initialize force constants tensor
        C_xNav = torch.zeros(
            (self._natoms * 3, self._n_cells, self._natoms, 3),
            dtype=self.dtype,
            device=self.device,
        )

        # Force constant calculation
        for i, a in enumerate(self.indices):
            for j, v in enumerate(["x", "y", "z"]):
                # Get forces for negative and positive displacements
                name_minus = self._get_displacement_name(a, j, -1)
                name_plus = self._get_displacement_name(a, j, 1)

                if name_minus not in self.cache or name_plus not in self.cache:
                    raise ValueError(
                        f"Missing displacement data for atom {a}, direction {j}"
                    )

                fminus_av = self.cache[name_minus]["forces"]
                fplus_av = self.cache[name_plus]["forces"]

                # Enforce momentum conservation by mean-field substraction
                n_atoms_unit = len(self.state.positions)

                # Identify the displaced atom in the supercell
                offset_index = self.offset * n_atoms_unit + a
                fminus_sum = torch.sum(fminus_av, dim=0)
                fplus_sum = torch.sum(fplus_av, dim=0)
                fminus_av[offset_index] -= fminus_sum
                fplus_av[offset_index] -= fplus_sum

                # Finite difference derivative
                C_av = (fminus_av - fplus_av) / (2 * self.delta)

                # Reshape
                n_atoms_unit = len(self.state.positions)
                C_Nav = C_av.reshape((self._n_cells, n_atoms_unit, 3))

                # Atom selection
                index_mask = torch.zeros(
                    n_atoms_unit, dtype=torch.bool, device=self.device
                )
                index_mask[self.indices] = True
                C_Nav = C_Nav[:, index_mask, :]

                index = 3 * i + j
                C_xNav[index] = C_Nav

        # Reshape: (n_cells, 3*natoms, 3*natoms)
        C_N = C_xNav.permute(1, 0, 2, 3).reshape(
            (self._n_cells, 3 * self._natoms, 3 * self._natoms)
        )

        if cutoff is not None:
            self._apply_cutoff(C_N, cutoff)

        # TODO: Check if methods work correctly
        for _ in range(symm_factor):
            C_N = self._symmetrize(C_N)
            if acoustic:
                self._apply_acoustic(C_N)

        self.C_N = C_N
        self.D_N = C_N.clone()

        self.m_inv_x = torch.repeat_interleave(
            torch.sqrt(1.0 / self.state.masses[self.indices]), 3
        )

        # Mass-scale dynamical matrix
        M_inv = torch.outer(self.m_inv_x, self.m_inv_x)
        for i in range(len(self.D_N)):
            self.D_N[i] *= M_inv

    def _compute_q_dynamical_matrix(self, q_scaled: torch.Tensor) -> torch.Tensor:
        """Compute the dynamical matrix for a given q-vector using precomputed D_N.

        This uses the precomputed mass-scaled dynamical matrix D_N.

        Args:
            q_scaled: q-vector in scaled coordinates (relative to reciprocal lattice)

        Returns:
            torch.Tensor: Dynamical matrix D(q) of shape (3*natoms, 3*natoms)
        """
        if self.D_N is None:
            self._calculate_force_constants()

        # Make sure q_scaled is a tensor
        if not isinstance(q_scaled, torch.Tensor):
            q_scaled = torch.tensor(q_scaled, dtype=self.dtype, device=self.device)

        if q_scaled.dim() == 1:
            q_scaled = q_scaled.unsqueeze(0)

        # Convert lattice vectors to complex dtype for better numerical precision
        # This is crucial for phase factor calculations
        R_cN = self.lattice_vectors.to(dtype=torch.float64, device=self.device)

        # Compute phases and transform
        # phase = exp(-i*2π*q·R)
        phase_factors = torch.exp(+2j * math.pi * torch.matmul(q_scaled, R_cN.T))

        # Extract the relevant part of the dynamical matrix
        # DOF: 3*natoms x 3*natoms per cell
        n_dof = 3 * self._natoms
        D_q = torch.zeros((n_dof, n_dof), dtype=torch.complex128, device=self.device)

        # Sum over all cells with phase factors
        for i, phase in enumerate(phase_factors[0]):
            if i < len(self.D_N):
                cell_matrix = self.D_N[i]
                D_q += phase * cell_matrix[:n_dof, :n_dof]

        return D_q

    def run(self, **kwargs):
        """Run the displacement calculations.

        This method creates a supercell, then displaces each atom in each direction,
        and computes forces for each displacement.

        Args:
            **kwargs: Additional keyword arguments for _calculate_force_constants
        """
        if self.calculator is None:
            raise ValueError("Calculator must be set before running calculations")

        supercell = self._create_supercell()

        # Equilibrium structure forces
        eq_name = f"{self.name}.eq"
        if eq_name not in self.cache:
            eq_forces = self(supercell)
            self.cache[eq_name] = {"forces": eq_forces}

        # Create workload for displacements
        displacements = []
        for atom_idx in self.indices:
            for direction in range(3):
                for sign in [-1, 1]:
                    disp_name = self._get_displacement_name(atom_idx, direction, sign)
                    if disp_name not in self.cache:
                        displacements.append((atom_idx, direction, sign, disp_name))

        # Process displacements
        for atom_idx, direction, sign, disp_name in displacements:
            displaced = self._displace_atom(supercell, atom_idx, direction, sign)

            # Keep on device to avoid CPU-GPU transfers
            forces = self(displaced)
            self.cache[disp_name] = {"forces": forces}

        # Calculate force constants with given parameters
        self._calculate_force_constants(**kwargs)

    def band_structure(
        self,
        path_kc: torch.Tensor,
        modes: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Calculate phonon dispersion along a path in the Brillouin zone.

        The dynamical matrix at arbitrary q-vectors is obtained by Fourier
        transforming the real-space force constants. For negative eigenvalues
        (imaginary frequencies), the negative frequency is returned to make
        visualization easier while maintaining the physical meaning.

        Args:
            path_kc: List of k-point coordinates defining the path
            modes: If True, return both frequencies and modes

        Returns:
            If modes=False: tensor of frequencies
            If modes=True: tuple of (frequencies, eigenvectors)
        """
        if self.D_N is None:
            raise ValueError("Dynamical matrix not computed yet.")

        # Convert path to tensor if needed
        if not isinstance(path_kc, torch.Tensor):
            path_kc = torch.tensor(path_kc, dtype=self.dtype, device=self.device)

        # Pre-allocate
        n_kpoints = path_kc.shape[0]
        n_modes = 3 * self._natoms  # This is the physically correct number of modes

        # Create tensors directly rather than lists for better performance
        omega_k = torch.zeros(
            (n_kpoints, n_modes), dtype=self.dtype, device=self.device
        )

        if modes:
            u_k = torch.zeros(
                (n_kpoints, n_modes, self._natoms, 3),
                dtype=(
                    torch.complex64 if self.dtype == torch.float32 else torch.complex128
                ),
                device=self.device,
            )

        for k, q_c in enumerate(path_kc):
            # Compute Hermitian dynamical matrix at q
            D_q = self._compute_q_dynamical_matrix(q_c)
            D_q = 0.5 * (D_q + D_q.conj().T)

            # Eigenvalues
            if modes:
                omega2_l, u_xl = torch.linalg.eigh(D_q)

                # Get sorting indices for eigenvalues (ascending order)
                # Ensure sorting is stable by using the real part
                sorted_idx = torch.argsort(omega2_l.real)
                omega2_l = omega2_l[sorted_idx]
                u_xl = u_xl[:, sorted_idx]

                # Mass-weight eigenvectors and reshape to (modes, atoms, xyz)
                # This ensures normalization: |u|^2 = 1/meff
                u_lx = (self.m_inv_x[:, None] * u_xl).T
                u_k[k] = u_lx.reshape(-1, self._natoms, 3)
            else:
                omega2_l = torch.linalg.eigvalsh(D_q)
                omega2_l = torch.sort(omega2_l.real)[0]

            # Complex valued eigenvalues as:
            # real(sqrt(ω²)) for ω² > 0, -abs(sqrt(ω²)) for ω² < 0
            omega_l = torch.zeros_like(omega2_l, dtype=self.dtype)

            # Handle very small values
            zero_mask = torch.abs(omega2_l) < 1e-8
            omega2_l[zero_mask] = 0.0

            pos_mask = omega2_l >= 0
            omega_l[pos_mask] = torch.sqrt(omega2_l[pos_mask].real)

            neg_mask = omega2_l < 0
            # Handle negative eigenvalues
            if torch.any(neg_mask):
                omega_l[neg_mask] = -torch.sqrt(torch.abs(omega2_l[neg_mask]))

            # Store frequencies
            omega_k[k] = omega_l

        if modes:
            return omega_k, u_k

        return omega_k
