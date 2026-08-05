"""Topology graph representation and confidence-scored module classification for InterpLens."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import torch.nn as nn


@dataclass
class DiscoveredModule:
    """Discovered PyTorch submodule with confidence score and discovery rationale."""
    module_path: str
    module: Optional[nn.Module] = field(default=None, repr=False)
    confidence: float = 1.0  # 0.0 to 1.0 confidence score
    reason: str = "Exact match"
    param_count: int = 0
    sublayer_type: str = "unknown"  # embedding, mhsa, mlp, norm, lm_head, block

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_path": self.module_path,
            "confidence": self.confidence,
            "reason": self.reason,
            "param_count": self.param_count,
            "sublayer_type": self.sublayer_type,
        }


@dataclass
class GraphNode:
    """Computational topology node in the ModelGraph hierarchy."""
    node_id: str
    name: str
    node_type: str  # embedding, block, attention, mlp, norm, lm_head
    module_path: str = ""
    layer_idx: Optional[int] = None
    children: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type,
            "module_path": self.module_path,
            "layer_idx": self.layer_idx,
            "children": self.children,
            "confidence": self.confidence,
        }


@dataclass
class ModelGraph:
    """Hierarchical computational topology graph for a neural network model."""
    root_node_id: str = "model"
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    discovered_modules: Dict[str, DiscoveredModule] = field(default_factory=dict)
    overall_confidence: float = 1.0

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node

    def add_discovered_module(self, key: str, module: DiscoveredModule) -> None:
        self.discovered_modules[key] = module

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_node_id": self.root_node_id,
            "overall_confidence": self.overall_confidence,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "discovered_modules": {k: v.to_dict() for k, v in self.discovered_modules.items()},
        }
