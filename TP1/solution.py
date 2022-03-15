

import json
from typing import TextIO
from config import Config

from cube import Cube
from tree import Node


class Solution:
    def __init__(self, config: Config, solution_node: Node, expanded_nodes: int, border_nodes: int, processing_time: float) -> None:
        self.solution_node = solution_node
        self.config = config
        self.expanded_nodes = expanded_nodes
        self.border_nodes = border_nodes
        self.processing_time = processing_time

    def save(self, file: TextIO, file_for_visualization: TextIO):
        seq = self.solution_node.get_branch()
        file_for_visualization.write(seq[0].state.get_faces())
        file_for_visualization.write("===\n")
        for state in seq[1:]:
            file_for_visualization.write(f'{state.action}\n')

        success = self.solution_node is not None

        success_data = {
            "solution_depth": self.solution_node.get_depth() + 1,
            "solution_cost": self.solution_node.get_depth(),
            "expanded_nodes": self.expanded_nodes,
            "border_nodes": self.border_nodes,
            "solution": {
                "initial_state": repr(self.config.state),
                "intermediate_states": [repr(node.state) for node in self.solution_node.get_branch()[1:-1]],
                "final_state": repr(self.solution_node.state)
            },
        } if success else {}

        json.dump({
            "result": "success" if success else "failed",
            "processing_time": self.processing_time,
            "search_config": self.config.to_dict(),
        } | success_data, file, indent=2)
