"""Quantum algorithms built on top of the simulator."""
from __future__ import annotations

import math

import numpy as np

from simulator import QuantumCircuit


def bell_pair(seed=None) -> QuantumCircuit:
    qc = QuantumCircuit(2, seed)
    qc.h(0).cx(0, 1)
    return qc


def ghz(n: int, seed=None) -> QuantumCircuit:
    qc = QuantumCircuit(n, seed)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def deutsch_jozsa(n: int, oracle: str = "balanced", seed=None) -> str:
    """Decide constant vs balanced with one oracle call.

    Qubits 0..n-1 are inputs, qubit n is the ancilla. Returns the measured
    input string: all zeros => constant, anything else => balanced.
    """
    qc = QuantumCircuit(n + 1, seed)
    qc.x(n)
    for q in range(n + 1):
        qc.h(q)
    if oracle == "balanced":      # f(x) = parity of x
        for q in range(n):
            qc.cx(q, n)
    elif oracle == "constant1":   # f(x) = 1
        qc.x(n)
    elif oracle == "constant0":   # f(x) = 0
        pass
    else:
        raise ValueError(oracle)
    for q in range(n):
        qc.h(q)
    probs = qc.prob_dict()
    # marginalise over ancilla (leftmost bit)
    marg: dict[str, float] = {}
    for bits, p in probs.items():
        marg[bits[1:]] = marg.get(bits[1:], 0) + p
    return max(marg, key=marg.get)


def grover(n: int, marked: int, iterations: int | None = None, seed=None):
    """Grover search for basis state `marked` among 2^n items.

    Returns (circuit, iterations_used, success_probability).
    """
    N = 2**n
    if not 0 <= marked < N:
        raise ValueError("marked item out of range")
    if iterations is None:
        iterations = max(1, round(math.pi / 4 * math.sqrt(N)))
    qc = QuantumCircuit(n, seed)
    for q in range(n):
        qc.h(q)
    qubits = list(range(n))
    for _ in range(iterations):
        # oracle: flip sign of |marked>
        zeros = [q for q in qubits if not (marked >> q) & 1]
        for q in zeros:
            qc.x(q)
        qc.mcz(qubits)
        for q in zeros:
            qc.x(q)
        # diffusion: 2|s><s| - I
        for q in qubits:
            qc.h(q)
            qc.x(q)
        qc.mcz(qubits)
        for q in qubits:
            qc.x(q)
            qc.h(q)
    return qc, iterations, float(qc.probabilities()[marked])


def qft(qc: QuantumCircuit, qubits: list[int] | None = None, inverse: bool = False):
    """Apply the quantum Fourier transform in place (qubits[0] = LSB)."""
    qubits = list(range(qc.n)) if qubits is None else qubits
    n = len(qubits)
    sign = -1 if inverse else 1
    if not inverse:
        for j in reversed(range(n)):
            qc.h(qubits[j])
            for k in reversed(range(j)):
                qc.cphase(qubits[k], qubits[j], sign * math.pi / 2 ** (j - k))
        for i in range(n // 2):
            qc.swap(qubits[i], qubits[n - 1 - i])
    else:
        for i in range(n // 2):
            qc.swap(qubits[i], qubits[n - 1 - i])
        for j in range(n):
            for k in range(j):
                qc.cphase(qubits[k], qubits[j], sign * math.pi / 2 ** (j - k))
            qc.h(qubits[j])
    return qc


def teleport(alpha: complex, beta: complex, seed=None):
    """Teleport state alpha|0>+beta|1> from qubit 0 to qubit 2.

    Returns (bob_state_vector, measurement_bits).
    """
    norm = math.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    alpha, beta = alpha / norm, beta / norm
    qc = QuantumCircuit(3, seed)
    qc.state = np.zeros(8, dtype=complex)
    qc.state[0b000] = alpha  # q0 = 0
    qc.state[0b001] = beta   # q0 = 1
    qc.h(1).cx(1, 2)         # Bell pair between Alice (1) and Bob (2)
    qc.cx(0, 1).h(0)         # Alice's Bell-basis rotation
    m0, m1 = qc.measure(0), qc.measure(1)
    if m1:
        qc.x(2)
    if m0:
        qc.z(2)
    # tensor axes are (q2, q1, q0): fix the measured q1, q0 to read Bob's qubit
    bob = qc.state.reshape([2] * 3)[:, m1, m0]
    return bob / np.linalg.norm(bob), (m0, m1)
