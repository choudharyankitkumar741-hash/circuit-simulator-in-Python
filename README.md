# Quantum Circuit Simulator

A from-scratch statevector simulator in Python (NumPy only) with implementations of classic quantum algorithms.

## Files

| File | Purpose |
|---|---|
| `simulator.py` | `QuantumCircuit` class: gates (H, X, Y, Z, S, T, Rx, Ry, Rz, CNOT, CZ, Toffoli, controlled-phase, SWAP, multi-controlled Z), measurement, sampling, entanglement entropy |
| `algorithms.py` | Bell and GHZ states, Deutsch-Jozsa, Grover search, Quantum Fourier Transform, quantum teleportation |
| `demo.py` | Runs every algorithm and prints results |
| `test_quantum.py` | 20 pytest tests (gate correctness, unitarity, entanglement, algorithm outputs, QFT vs. DFT matrix) |

## Run

```bash
pip install numpy pytest
python demo.py
pytest -q
```

## How it works

The state of n qubits is a vector of 2^n complex amplitudes. Qubit 0 is the least-significant bit of the index, so `|q2 q1 q0>` has index `4*q2 + 2*q1 + q0`.

A single-qubit gate (optionally controlled) is applied by reshaping the vector into an n-axis tensor of shape `(2, ..., 2)` and contracting the 2x2 gate matrix with the target qubit's axis. Control qubits restrict the update to the sub-block where all controls are |1>. This costs O(2^n) per gate instead of building a 2^n x 2^n matrix.

Measurement probabilities are `|amplitude|^2`. `measure()` collapses the state; `sample()` draws shots without collapsing. Entanglement is quantified by the von Neumann entropy of the reduced density matrix (1 bit for a Bell pair, 0 for a product state).

## Algorithms and results

- **Bell / GHZ**: `(|00> + |11>)/sqrt2` gives entropy of exactly 1 bit.
- **Deutsch-Jozsa**: decides constant vs. balanced with one oracle query (classically up to 2^(n-1)+1 are needed).
- **Grover**: finds a marked item among N in about (pi/4)*sqrt(N) iterations with >94% success (N=8 to 256), versus ~N/2 classical guesses.
- **QFT**: matches the DFT matrix exactly; a signal with period 8 over N=64 gives peaks at multiples of N/8.
- **Teleportation**: fidelity 1.0 for every measurement outcome, after Bob's X/Z corrections.

## Limits and extensions

Memory grows as 2^n (about 30 qubits is a practical ceiling on a laptop). Ideas to extend: noise models (depolarizing channel via density matrices), Shor's algorithm using the QFT, VQE for a small molecule, circuit drawing, a QASM parser.
