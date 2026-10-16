import math

import numpy as np
import pytest

from algorithms import bell_pair, deutsch_jozsa, ghz, grover, qft, teleport
from simulator import QuantumCircuit


def test_hadamard_superposition():
    qc = QuantumCircuit(1).h(0)
    assert np.allclose(qc.probabilities(), [0.5, 0.5])


def test_x_flips_and_qubit_ordering():
    qc = QuantumCircuit(3).x(0)
    assert qc.prob_dict() == {"001": 1.0}
    qc = QuantumCircuit(3).x(2)
    assert qc.prob_dict() == {"100": 1.0}


def test_cnot_control_target():
    qc = QuantumCircuit(2).x(1).cx(1, 0)
    assert qc.prob_dict() == {"11": 1.0}
    qc = QuantumCircuit(2).x(0).cx(1, 0)   # control off: no change
    assert qc.prob_dict() == {"01": 1.0}


def test_toffoli():
    qc = QuantumCircuit(3).x(0).x(1).ccx(0, 1, 2)
    assert qc.prob_dict() == {"111": 1.0}


def test_bell_state_and_entanglement():
    qc = bell_pair()
    p = qc.prob_dict()
    assert set(p) == {"00", "11"}
    assert all(math.isclose(v, 0.5) for v in p.values())
    assert math.isclose(qc.entanglement_entropy([0]), 1.0, abs_tol=1e-9)


def test_product_state_has_zero_entropy():
    qc = QuantumCircuit(2).h(0).h(1)
    assert math.isclose(qc.entanglement_entropy([0]), 0.0, abs_tol=1e-9)


def test_ghz():
    p = ghz(4).prob_dict()
    assert set(p) == {"0000", "1111"}


def test_unitarity_preserves_norm():
    qc = QuantumCircuit(4)
    qc.h(0).rx(1, 0.7).cx(0, 2).t(3).cphase(1, 3, 1.1).ry(2, 2.2)
    assert math.isclose(np.linalg.norm(qc.state), 1.0)


@pytest.mark.parametrize("oracle,expected_zero", [("constant0", True), ("constant1", True), ("balanced", False)])
def test_deutsch_jozsa(oracle, expected_zero):
    out = deutsch_jozsa(4, oracle)
    assert (out == "0000") == expected_zero


@pytest.mark.parametrize("n,marked", [(3, 5), (4, 11), (5, 0), (6, 42)])
def test_grover_finds_marked(n, marked):
    qc, _, prob = grover(n, marked)
    assert prob > 0.9
    assert int(np.argmax(qc.probabilities())) == marked


def test_qft_matches_dft_matrix():
    n = 3
    N = 2**n
    for basis in range(N):
        qc = QuantumCircuit(n)
        qc.state[:] = 0
        qc.state[basis] = 1
        qft(qc)
        expected = np.exp(2j * np.pi * basis * np.arange(N) / N) / math.sqrt(N)
        assert np.allclose(qc.state, expected)


def test_qft_inverse_roundtrip():
    qc = QuantumCircuit(4).h(0).x(2).ry(1, 0.4)
    before = qc.state.copy()
    qft(qc)
    qft(qc, inverse=True)
    assert np.allclose(qc.state, before)


def test_teleportation():
    alpha, beta = 0.6, 0.8j
    for seed in range(10):
        bob, _ = teleport(alpha, beta, seed=seed)
        target = np.array([alpha, beta])
        assert math.isclose(abs(np.vdot(target, bob)), 1.0, abs_tol=1e-9)


def test_measurement_collapse():
    qc = bell_pair(seed=3)
    a = qc.measure(0)
    b = qc.measure(1)
    assert a == b


def test_sampling_statistics():
    qc = QuantumCircuit(1, seed=1).h(0)
    counts = qc.sample(10000)
    assert abs(counts["0"] - 5000) < 300
