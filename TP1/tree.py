from typing import Optional
from cube import \
    Action, Cube, solved_cubes, turns, cube_rotations, \
    apply_action, apply_algorythm, parse_algorythm

actions = "R R' L L' U U' D D' F F' B B'".split()


class Node:
    def calculate_heuristic(self):
        #TODO: Implementar metodos de heuristica
        if self.state.is_solved():
            return 0
        return 1

    def __init__(self, state: Cube, action: Optional[str] = None, parent: Optional['Node'] = None):
        self.state = state
        self.action = action
        self.explored = False
        self.child_nodes: list[Node] = []
        self.parent = parent
        self.heuristic = self.calculate_heuristic()    

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
        self.border: list[Node] = []

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

    def global_heuristic(self) -> Optional[Node]:
        self.border.append(self.root)
        while self.border:
            s = self.border.pop(0)
            if s.state.is_solved():
                return s
            for n in s.calculate_children():
                if n.state not in self.visited:
                    s.add_child(n)
            for n in s.child_nodes:
                self.visited.add(n.state)
                self.border.append(n)
            #TODO: Reordenar border segun la heuristica del estado que etiqueta cada nodo
        self.visited.clear()    
        return None

    def get_min_heuristic(self, L: list[Node]) -> Optional[Node]:
        if not L:
            return None
        min = L[0]
        for n in L:
            if n.heuristic < min.heuristic:
                min.heuristic = n.heuristic
        return min

    def _local_heuristic(self, L: list[Node]) -> Optional[Node]:
        s = self.get_min_heuristic(L)
        print(s.heuristic)
        if s.heuristic == 0:             
            return s
        L.remove(s)
        for n in s.calculate_children():
            L.append(n)
            self._local_heuristic(L)


    def local_heuristic(self) -> Optional[Node]:
        L: list[Node] = []
        L.append(self.root)
        return self._local_heuristic(L)
