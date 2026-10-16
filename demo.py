"""Run all demos:  python demo.py"""
import math

import numpy as np

from algorithms import bell_pair, deutsch_jozsa, ghz, grover, qft, teleport
from simulator import QuantumCircuit


def bar(p, width=30):
    return "#" * round(p * width)


def show(title, probs):
    print(f"\n== {title} ==")
    for bits, p in probs.items():
        print(f"  |{bits}>  {p:6.3f}  {bar(p)}")


def main():
    # 1. Bell state
    qc = bell_pair(seed=7)
    show("Bell state (|00> + |11>)/sqrt2", qc.prob_dict())
    print("  1000 shots:", qc.sample(1000))
    print(f"  entanglement entropy of qubit 0: {qc.entanglement_entropy([0]):.3f} bit")

    # 2. GHZ
    show("4-qubit GHZ state", ghz(4).prob_dict())

    # 3. Deutsch-Jozsa
    print("\n== Deutsch-Jozsa (5 input qubits, ONE oracle query) ==")
    for o in ("constant0", "constant1", "balanced"):
        out = deutsch_jozsa(5, o)
        verdict = "constant" if set(out) == {"0"} else "balanced"
        print(f"  oracle={o:<9} measured={out} -> {verdict}")

    # 4. Grover
    print("\n== Grover search ==")
    print("  n  N     iters  P(success)  classical expected guesses")
    for n in range(3, 9):
        N = 2**n
        _, it, p = grover(n, marked=N // 3)
        print(f"  {n}  {N:<5} {it:<6} {p:9.4f}   ~{N // 2}")

    # 5. QFT of a periodic state reveals the period
    print("\n== QFT: finding the period of a signal ==")
    n, period = 6, 8
    N = 2**n
    qc = QuantumCircuit(n)
    qc.state[:] = 0
    qc.state[::period] = 1
    qc.state /= np.linalg.norm(qc.state)
    qft(qc)
    peaks = [i for i, p in enumerate(qc.probabilities()) if p > 1e-6]
    print(f"  state has period {period} over N={N}; QFT peaks at {peaks}")
    print(f"  spacing N/period = {N // period}  (peaks at multiples of N/r)")

    # 6. Teleportation
    print("\n== Quantum teleportation ==")
    alpha, beta = 0.6, 0.8j
    for seed in range(3):
        bob, (m0, m1) = teleport(alpha, beta, seed=seed)
        fid = abs(np.vdot([alpha, beta], bob)) ** 2
        print(f"  Alice measured ({m0},{m1}); fidelity of Bob's qubit = {fid:.6f}")


if __name__ == "__main__":
    main()
