"""Category tree: paths, ancestors and probability roll-up to inner nodes."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np


class Taxonomy:
    def __init__(self, tree: dict):
        self.parent: dict[str, str | None] = {}
        self._walk(tree, None)

    def _walk(self, nodes: dict, parent: str | None) -> None:
        for node_id, node in nodes.items():
            self.parent[node_id] = parent
            self._walk(node.get("children", {}), node_id)

    @classmethod
    def load(cls, path: Path) -> Taxonomy:
        return cls(json.loads(Path(path).read_text()))

    @classmethod
    def from_paths(cls, paths: list[str]) -> Taxonomy:
        """Build the tree from 'a/b/c' category paths (used in tests and as a fallback)."""
        tree: dict = {}
        for p in paths:
            level = tree
            for node in p.split("/"):
                level = level.setdefault(node, {"children": {}})["children"]
        return cls(tree)

    def path(self, node: str) -> list[str]:
        out = []
        while node is not None:
            out.append(node)
            node = self.parent[node]
        return out[::-1]

    def depth(self, node: str) -> int:
        return len(self.path(node))

    @cached_property
    def children(self) -> dict[str, list[str]]:
        kids: dict[str, list[str]] = {n: [] for n in self.parent}
        for node, parent in self.parent.items():
            if parent is not None:
                kids[parent].append(node)
        return kids

    @cached_property
    def leaves(self) -> list[str]:
        return sorted(n for n, k in self.children.items() if not k)

    def rollup(self, leaf_probs: np.ndarray, leaf_labels: list[str]) -> dict[str, float]:
        """Probability of every node = sum of the probabilities of the leaves below it."""
        probs: dict[str, float] = {}
        for p, leaf in zip(leaf_probs, leaf_labels, strict=True):
            for node in self.path(leaf):
                probs[node] = probs.get(node, 0.0) + float(p)
        return probs

    def deepest_confident_node(self, leaf_probs: np.ndarray, leaf_labels: list[str],
                               threshold: float) -> tuple[str, float]:
        """Most specific category whose rolled-up probability reaches `threshold`.

        When the model hesitates between two sibling leaves ("LED TV" vs "OLED TV"), it can still
        assert the parent ("TV") with high confidence: useful for partial auto-categorisation.
        """
        probs = self.rollup(leaf_probs, leaf_labels)
        best = max((n for n, p in probs.items() if p >= threshold), key=lambda n: (self.depth(n), probs[n]),
                   default=None)
        if best is None:  # root always reaches 1.0, but guard anyway
            best = self.path(leaf_labels[int(np.argmax(leaf_probs))])[0]
        return best, probs[best]
