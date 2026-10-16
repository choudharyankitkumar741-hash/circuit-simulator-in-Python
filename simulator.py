"""Core statevector quantum circuit simulator.

Convention: qubit 0 is the least-significant bit of the basis-state index,
so the basis state |q_{n-1} ... q_1 q_0> has index sum(q_i * 2**i).
Bitstrings are printed with q_{n-1} on the left.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def phase(theta: float) -> np.ndarray:
    """Phase-shift gate diag(1, e^{i theta})."""
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)


class QuantumCircuit:
    """An n-qubit circuit simulated by tracking the full 2^n state vector."""

    def __init__(self, n_qubits: int, seed: int | None = None):
        if n_qubits < 1:
            raise ValueError("need at least one qubit")
        self.n = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=complex)
        self.state[0] = 1.0
        self.rng = np.random.default_rng(seed)
        self.history: list[str] = []

    # ------------------------------------------------------ applying gates
    def apply(self, gate: np.ndarray, target: int, controls: tuple[int, ...] = ()):
        """Apply a single-qubit `gate` to `target`, optionally controlled.

        Works by reshaping the state into an n-axis tensor and contracting
        the gate with the target axis; controls restrict the update to the
        sub-block where every control qubit is |1>.
        """
        self._check(target, *controls)
        if target in controls:
            raise ValueError("target cannot also be a control")
        psi = self.state.reshape([2] * self.n)
        # axis for qubit q is (n-1-q) because qubit 0 is the last axis
        t_ax = self.n - 1 - target
        index = [slice(None)] * self.n
        for c in controls:
            index[self.n - 1 - c] = 1
        sub = psi[tuple(index)]
        # target axis position inside the sub-block after control axes removed
        removed_before = sum(1 for c in controls if (self.n - 1 - c) < t_ax)
        ax = t_ax - removed_before
        sub = np.moveaxis(np.tensordot(gate, sub, axes=([1], [ax])), 0, ax)
        psi[tuple(index)] = sub
        self.state = psi.reshape(-1)
        name = "C" * len(controls) + "U"
        self.history.append(f"{name} target={target} controls={controls}")
        return self

    # convenience wrappers
    def h(self, q): return self.apply(H, q)
    def x(self, q): return self.apply(X, q)
    def y(self, q): return self.apply(Y, q)
    def z(self, q): return self.apply(Z, q)
    def s(self, q): return self.apply(S, q)
    def t(self, q): return self.apply(T, q)
    def rx(self, q, th): return self.apply(rx(th), q)
    def ry(self, q, th): return self.apply(ry(th), q)
    def rz(self, q, th): return self.apply(rz(th), q)
    def cx(self, c, t): return self.apply(X, t, (c,))
    def cz(self, c, t): return self.apply(Z, t, (c,))
    def ccx(self, c1, c2, t): return self.apply(X, t, (c1, c2))
    def cphase(self, c, t, th): return self.apply(phase(th), t, (c,))

    def swap(self, a, b):
        self.cx(a, b).cx(b, a).cx(a, b)
        return self

    def mcz(self, qubits: list[int]):
        """Multi-controlled Z across `qubits` (flips sign of |1...1>)."""
        *ctrls, tgt = qubits
        return self.apply(Z, tgt, tuple(ctrls))

    # ------------------------------------------------------ measurement
    def probabilities(self) -> np.ndarray:
        return np.abs(self.state) ** 2

    def prob_dict(self, threshold: float = 1e-9) -> dict[str, float]:
        p = self.probabilities()
        return {format(i, f"0{self.n}b"): float(p[i]) for i in range(len(p)) if p[i] > threshold}

    def sample(self, shots: int = 1024) -> dict[str, int]:
        """Sample measurement outcomes without collapsing the state."""
        outcomes = self.rng.choice(len(self.state), size=shots, p=self.probabilities())
        counts: dict[str, int] = {}
        for o in outcomes:
            key = format(int(o), f"0{self.n}b")
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def measure(self, qubit: int) -> int:
        """Projectively measure one qubit, collapsing the state."""
        self._check(qubit)
        psi = self.state.reshape([2] * self.n)
        ax = self.n - 1 - qubit
        p1 = float(np.sum(np.abs(np.take(psi, 1, axis=ax)) ** 2))
        result = int(self.rng.random() < p1)
        # collapse: zero out the other branch, renormalise
        idx = [slice(None)] * self.n
        idx[ax] = 1 - result
        psi = psi.copy()
        psi[tuple(idx)] = 0
        self.state = psi.reshape(-1)
        self.state /= np.linalg.norm(self.state)
        return result

    # ------------------------------------------------------ analysis
    def reduced_density_matrix(self, keep: list[int]) -> np.ndarray:
        psi = self.state.reshape([2] * self.n)
        keep_axes = [self.n - 1 - q for q in keep]
        other = [a for a in range(self.n) if a not in keep_axes]
        mat = np.transpose(psi, keep_axes + other).reshape(2 ** len(keep), -1)
        return mat @ mat.conj().T

    def entanglement_entropy(self, keep: list[int]) -> float:
        """Von Neumann entropy (bits) of the subsystem `keep`."""
        rho = self.reduced_density_matrix(keep)
        w = np.linalg.eigvalsh(rho)
        w = w[w > 1e-12]
        return float(-np.sum(w * np.log2(w)))

    def _check(self, *qs):
        for q in qs:
            if not 0 <= q < self.n:
                raise IndexError(f"qubit {q} out of range for {self.n}-qubit circuit")
