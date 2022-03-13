from typing import Optional
from cube import \
    Action, Cube, solved_cubes, turns, cube_rotations, \
    apply_action, apply_algorythm, parse_algorythm

actions = "R R' L L' U U' D D' F F' B B'".split()


class Node:
    def __init__(self, state: Cube, action: Optional[str] = None, parent: Optional['Node'] = None):
        self.state = state
        self.action = action
        self.explored = False
        self.child_nodes: list[Node] = []
        self.parent = parent

    def add_child(self, child: 'Node'):
        self.child_nodes.append(child)

    def calculate_children(self):
        for action in actions:
            yield Node(apply_action(self.state, turns[action]), action, self)

    def get_branch(self) -> list[tuple[Cube, Optional[str]]]:
        branch = self.parent.get_branch() if self.parent is not None else []
        branch.append((self.state, self.action))
        return branch

    def get_depth(self) -> int:
        return self.parent.get_depth() + 1 if self.parent is not None else 0


class Tree:
    def __init__(self, root: Node):
        self.root = root
        self.visited: set[Cube] = set()
        self.queue: list[Node] = []

    def bpa(self) -> Optional[Node]:
        self.visited.add(self.root.state)
        self.queue.append(self.root)
        while self.queue:
            s = self.queue.pop(0)
            if s.state.is_solved():
                return s
            # print(f'\r{s.get_depth()}', end="")
            for node in s.calculate_children():
                if node.state not in self.visited:
                    s.add_child(node)
            for n in s.child_nodes:
                self.visited.add(n.state)
                self.queue.append(n)
        self.visited.clear()

        return None

    def _bpp(self, node: Node, depth: int = 0, max_depth: Optional[int] = None) -> Optional[Node]:
        if max_depth is not None and depth > max_depth:
            return None
        # print(depth)
        if node not in self.visited:
            if node.state.is_solved():
                return node
            for n in node.calculate_children():
                if n.state not in self.visited:
                    node.add_child(n)
            self.visited.add(node.state)
            for child in node.child_nodes:
                sol = self._bpp(child, depth + 1, max_depth)
                if sol is not None:
                    return sol
        return None

    def bpp(self, max_depth: Optional[int] = None):
        sol = self._bpp(self.root, max_depth=max_depth)
        self.visited.clear()
        return sol

    def bppv(self, max_depth: int):
        sol = self.bpp(max_depth)
        self.visited.clear()
        return sol
