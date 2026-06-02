#!/usr/bin/env python3
"""Prototipo TP3: Hopfield para denoise y detección de centro de aro en imagen 10x10."""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np

Pattern = np.ndarray


def make_blank_pattern(size: int = 10) -> Pattern:
    return -np.ones((size, size), dtype=int)


def make_ring_pattern(center: Tuple[int, int], ref_block: List[Tuple[int, int]], size: int = 10) -> Pattern:
    pattern = make_blank_pattern(size)
    cx, cy = center
    radius = 2
    for r in range(size):
        for c in range(size):
            if (abs(r - cy) == radius and abs(c - cx) <= radius) or (abs(c - cx) == radius and abs(r - cy) <= radius):
                pattern[r, c] = 1
    for (r, c) in ref_block:
        pattern[r, c] = 1
    return pattern


def flatten(pattern: Pattern) -> np.ndarray:
    return pattern.reshape(-1)


def unflatten(vector: np.ndarray, size: int = 10) -> Pattern:
    return vector.reshape(size, size)


def display_pattern(pattern: Pattern) -> str:
    chars = {1: '#', -1: '.'}
    return '\n'.join(''.join(chars[int(v)] for v in row) for row in pattern)


def add_noise(pattern: Pattern, noise_prob: float) -> Pattern:
    noisy = pattern.copy()
    mask = np.random.rand(*pattern.shape) < noise_prob
    noisy[mask] *= -1
    return noisy


def center_of_mass(pattern: Pattern, value: int = 1) -> Tuple[float, float]:
    coords = np.argwhere(pattern == value)
    if coords.size == 0:
        raise ValueError('No pattern pixels found for center computation.')
    avg_row = np.mean(coords[:, 0])
    avg_col = np.mean(coords[:, 1])
    return avg_col, avg_row


def relative_coordinates(center: Tuple[float, float], ref_point: Tuple[int, int]) -> Tuple[float, float]:
    ref_x, ref_y = ref_point
    cx, cy = center
    x = cx - ref_x
    y = ref_y - cy
    return x, y


class HopfieldNetwork:
    def __init__(self, size: int = 100, weights: np.ndarray | None = None) -> None:
        self.size = size
        self.weights = np.zeros((size, size), dtype=float) if weights is None else weights

    @classmethod
    def from_hebb(cls, patterns: List[Pattern]) -> HopfieldNetwork:
        n = patterns[0].size
        net = cls(size=n)
        for pattern in patterns:
            vector = flatten(pattern).astype(float)
            net.weights += np.outer(vector, vector)
        np.fill_diagonal(net.weights, 0)
        net.weights /= n
        return net

    @classmethod
    def from_pseudoinverse(cls, patterns: List[Pattern]) -> HopfieldNetwork:
        n = patterns[0].size
        patterns_matrix = np.stack([flatten(p).astype(float) for p in patterns], axis=1)
        pseudoinv = np.linalg.pinv(patterns_matrix.T.dot(patterns_matrix)).dot(patterns_matrix.T)
        weights = patterns_matrix.dot(pseudoinv)
        np.fill_diagonal(weights, 0)
        net = cls(size=n, weights=weights)
        return net

    def update(self, state: np.ndarray, synchronous: bool = False) -> np.ndarray:
        if synchronous:
            raw = self.weights.dot(state)
            return np.where(raw >= 0, 1, -1)
        result = state.copy()
        for i in np.random.permutation(self.size):
            raw = self.weights[i].dot(result)
            result[i] = 1 if raw >= 0 else -1
        return result

    def run_until_stable(self, state: np.ndarray, max_steps: int = 1000) -> np.ndarray:
        current = state.copy()
        for _ in range(max_steps):
            next_state = self.update(current)
            if np.array_equal(next_state, current):
                break
            current = next_state
        return current


def example_patterns() -> Tuple[List[Pattern], Tuple[float, float]]:
    ref_block = [(8, 0), (8, 1), (9, 0), (9, 1)]
    ref_center = (
        np.mean([c for _, c in ref_block]),
        np.mean([r for r, _ in ref_block]),
    )
    pattern_a = make_ring_pattern(center=(4, 4), ref_block=ref_block)
    pattern_b = make_ring_pattern(center=(5, 5), ref_block=ref_block)
    return [pattern_a, pattern_b], ref_center


def evaluate_network(name: str, net: HopfieldNetwork, patterns: List[Pattern], ref_point: Tuple[float, float], noise_prob: float = 0.15) -> str:
    lines = [f'=== {name} ===']
    for idx, pattern in enumerate(patterns, start=1):
        noisy = add_noise(pattern, noise_prob)
        lines.append(f'Pattern {idx}: original')
        lines.append(display_pattern(pattern))
        lines.append('Noisy input:')
        lines.append(display_pattern(noisy))
        recovered = net.run_until_stable(flatten(noisy))
        recovered_pattern = unflatten(recovered)
        lines.append('Recovered output:')
        lines.append(display_pattern(recovered_pattern))
        ring_center = center_of_mass(recovered_pattern, value=1)
        x_rel, y_rel = relative_coordinates(ring_center, ref_point)
        lines.append(f'Center pixel coordinates (col, row): ({ring_center[0]:.2f}, {ring_center[1]:.2f})')
        lines.append(f'Relative coordinates X,Y: ({x_rel:.2f}, {y_rel:.2f})')
        lines.append('-' * 40)
    return '\n'.join(lines)


def main() -> None:
    patterns, ref_center = example_patterns()
    hebb_net = HopfieldNetwork.from_hebb(patterns)
    pseudo_net = HopfieldNetwork.from_pseudoinverse(patterns)

    report = []
    report.append('TP3 Prototipo Hopfield - Hebb vs Pseudoinversa')
    report.append('Se definen dos patrones de referencia 10x10 con una figura de aro y un bloque de referencia fijo.')
    report.append('Cada red se entrena con ambos patrones y se prueba la recuperación desde entradas ruidosas.')
    report.append('')
    report.append(evaluate_network('Hebb', hebb_net, patterns, ref_center, noise_prob=0.15))
    report.append('')
    report.append(evaluate_network('Pseudoinversa', pseudo_net, patterns, ref_center, noise_prob=0.15))

    output = '\n'.join(report)
    print(output)
    out_path = Path(__file__).resolve().parent / 'resultado_tp3.txt'
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(output)


if __name__ == '__main__':
    main()
