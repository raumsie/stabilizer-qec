"""Stabilizer codes for arbitrary [[n, k]] parameters (Gottesman theory).

A code is defined by its stabilizer generators and logical operators; an
error is diagnosed by its syndrome (which generators it anticommutes
with) and corrected via brute-force minimum-weight lookup decoding.
"""

import itertools
from typing import Optional

import numpy as np

from . import pauli


class StabilizerCode:
    """A [[n, k]] stabilizer code together with a brute-force lookup decoder."""

    def __init__(self, name: str, n: int, k: int, stabilizer_labels: list,
                 logical_x_labels: list, logical_z_labels: list,
                 distance: Optional[int] = None):
        self.name = name
        self.n = n
        self.k = k
        self.distance = distance

        self.stabilizer_labels = list(stabilizer_labels)
        self.logical_x_labels = list(logical_x_labels)
        self.logical_z_labels = list(logical_z_labels)

        self.stabilizers = [pauli.pauli_from_label(lbl) for lbl in stabilizer_labels]
        self.logical_x = [pauli.pauli_from_label(lbl) for lbl in logical_x_labels]
        self.logical_z = [pauli.pauli_from_label(lbl) for lbl in logical_z_labels]

        if len(self.stabilizers) != n - k:
            raise ValueError(
                f"{name}: expected {n - k} stabilizer generators for [[{n},{k}]], "
                f"got {len(self.stabilizers)}"
            )
        if len(self.logical_x) != k or len(self.logical_z) != k:
            raise ValueError(
                f"{name}: expected {k} logical X and {k} logical Z operators, "
                f"got {len(self.logical_x)} X and {len(self.logical_z)} Z"
            )
        for lbl in stabilizer_labels + logical_x_labels + logical_z_labels:
            if len(lbl) != n:
                raise ValueError(
                    f"{name}: label {lbl!r} has length {len(lbl)}, expected n={n}"
                )

        self._validate()
        self.stabilizer_group = self._build_stabilizer_group()

        self._lookup = None
        self.build_decoder()

    # ------------------------------------------------------------------
    # Validation done when the code is created
    # ------------------------------------------------------------------
    def _validate(self):
        name = self.name
        # All stabilizer generators must pairwise commute.
        for i in range(len(self.stabilizers)):
            for j in range(i + 1, len(self.stabilizers)):
                if not pauli.commutes(self.stabilizers[i], self.stabilizers[j]):
                    raise ValueError(
                        f"{name}: stabilizer generators {self.stabilizer_labels[i]!r} "
                        f"and {self.stabilizer_labels[j]!r} do not commute"
                    )

        # Logical operators must commute with every stabilizer generator.
        for i in range(self.k):
            for s_idx, s in enumerate(self.stabilizers):
                if not pauli.commutes(self.logical_x[i], s):
                    raise ValueError(
                        f"{name}: logical X_{i} ({self.logical_x_labels[i]!r}) does not "
                        f"commute with stabilizer {self.stabilizer_labels[s_idx]!r}"
                    )
                if not pauli.commutes(self.logical_z[i], s):
                    raise ValueError(
                        f"{name}: logical Z_{i} ({self.logical_z_labels[i]!r}) does not "
                        f"commute with stabilizer {self.stabilizer_labels[s_idx]!r}"
                    )

        # Logical operator algebra: X_i anticommutes with Z_i, commutes with
        # Z_j (j != i); X_i commutes with X_j; Z_i commutes with Z_j.
        for i in range(self.k):
            for j in range(self.k):
                xz_commutes = pauli.commutes(self.logical_x[i], self.logical_z[j])
                if i == j and xz_commutes:
                    raise ValueError(
                        f"{name}: logical X_{i} must anticommute with logical Z_{i}"
                    )
                if i != j and not xz_commutes:
                    raise ValueError(
                        f"{name}: logical X_{i} must commute with logical Z_{j} (i != j)"
                    )
                if i != j:
                    if not pauli.commutes(self.logical_x[i], self.logical_x[j]):
                        raise ValueError(
                            f"{name}: logical X_{i} must commute with logical X_{j}"
                        )
                    if not pauli.commutes(self.logical_z[i], self.logical_z[j]):
                        raise ValueError(
                            f"{name}: logical Z_{i} must commute with logical Z_{j}"
                        )

    def _build_stabilizer_group(self):
        """All 2**(n-k) elements from XOR-combining subsets of the
        generators, as a set of length-2n tuples for O(1) membership tests."""
        group = set()
        m = len(self.stabilizers)
        identity = np.zeros(2 * self.n, dtype=np.uint8)
        for bits in itertools.product([0, 1], repeat=m):
            elem = identity.copy()
            for bit, gen in zip(bits, self.stabilizers):
                if bit:
                    elem = pauli.multiply(elem, gen)
            group.add(tuple(int(v) for v in elem))
        return group

    # ------------------------------------------------------------------
    # Syndrome / decoding
    # ------------------------------------------------------------------
    def syndrome(self, error: np.ndarray) -> tuple:
        """Bit j is 1 iff `error` anticommutes with stabilizer generator j."""
        return tuple(pauli.symplectic_product(error, s) for s in self.stabilizers)

    def build_decoder(self, max_weight: Optional[int] = None):
        """Build a syndrome -> minimum-weight correction lookup table by
        enumerating Pauli errors up to `max_weight` (default: all 4**n,
        tractable for n <= 9) and keeping the lowest-weight one per syndrome."""
        n = self.n
        m = len(self.stabilizers)
        if max_weight is None:
            max_weight = n

        # digit d in {0,1,2,3} per qubit encodes I,X,Z,Y: x=d&1, z=(d>>1)&1.
        digits = np.array(
            np.meshgrid(*([np.arange(4, dtype=np.uint8)] * n), indexing="ij")
        ).reshape(n, -1).T  # shape (4**n, n)

        x_part = (digits & 1).astype(np.uint8)
        z_part = ((digits >> 1) & 1).astype(np.uint8)
        vecs = np.concatenate([x_part, z_part], axis=1)  # shape (4**n, 2n)
        weights = np.count_nonzero(digits, axis=1)

        keep = weights <= max_weight
        vecs = vecs[keep]
        weights = weights[keep]
        x_part = x_part[keep]
        z_part = z_part[keep]

        # Syndrome bit j = (x_part . s_z_j + z_part . s_x_j) mod 2, vectorized
        # over all errors for each stabilizer generator.
        syn_bits = np.zeros((vecs.shape[0], m), dtype=np.uint8)
        for j, s in enumerate(self.stabilizers):
            sx, sz = s[:n], s[n:]
            col = (x_part @ sz + z_part @ sx) % 2
            syn_bits[:, j] = col.astype(np.uint8)

        # Pack each syndrome bitstring into a small integer key for grouping.
        keys = np.zeros(vecs.shape[0], dtype=np.int64)
        for j in range(m):
            keys += syn_bits[:, j].astype(np.int64) << j

        idx = np.arange(vecs.shape[0])
        # Sort by key, then weight, then original index, so the first row
        # per key is the lowest-weight, first-found coset leader.
        order = np.lexsort((idx, weights, keys))
        sorted_keys = keys[order]
        _, first_pos = np.unique(sorted_keys, return_index=True)
        leader_rows = order[first_pos]

        lookup = {}
        for row in leader_rows:
            synd = tuple(int(b) for b in syn_bits[row])
            lookup[synd] = vecs[row].copy()

        self._lookup = lookup

    def decode(self, syndrome: tuple) -> np.ndarray:
        """Lookup-table correction for `syndrome` (identity if never seen)."""
        if syndrome in self._lookup:
            return self._lookup[syndrome].copy()
        return np.zeros(2 * self.n, dtype=np.uint8)

    def is_logical_error(self, error: np.ndarray, correction: np.ndarray) -> bool:
        """True if `error * correction` is outside the stabilizer group,
        i.e. the correction leaves a nontrivial logical operator behind."""
        residual = pauli.multiply(error, correction)
        return tuple(int(v) for v in residual) not in self.stabilizer_group
