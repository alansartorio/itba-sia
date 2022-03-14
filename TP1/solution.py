

from typing import TextIO

from cube import Cube
from tree import Node


class Solution:
    def __init__(self, final_node: Node) -> None:
        self.final_node = final_node
        
    def save(self, file: TextIO):
        seq = self.final_node.get_branch()
        file.write(seq[0].state.get_faces())
        file.write("===\n")
        for state in seq[1:]:
            file.write(f'{state.action}\n')
