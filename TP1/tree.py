from typing import Callable, Generic, Iterable, Optional, TypeVar
from typing_extensions import Self
from cube import \
    Action, Cube, solved_cubes, turns, cube_rotations, \
    apply_action, apply_algorythm, parse_algorythm
from bisect import insort

actions = "R R' L L' U U' D D' F F' B B'".split()

HeuristicFunction = Callable[[Cube], float]

class Node:
    def __init__(self, state: Cube, action: Optional[str] = None, parent: Optional[Self] = None):
        self.state = state
        self.action = action
        self.explored = False
        self.child_nodes: list[Self] = []
        self.parent = parent

    def add_child(self, child: Self):
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

class HeuristicNode(Node):
    def __init__(self, state: Cube, heuristic_function: HeuristicFunction, hasCost: bool, action: Optional[str] = None, parent: Optional[Self] = None):
        super().__init__(state, action=action, parent=parent)
        self.heuristic_function = heuristic_function
        self.heuristic = self.cost(hasCost) + self.calculate_heuristic()

    def calculate_heuristic(self):
        return self.heuristic_function(self.state)

    def cost(self, hasCost: bool):
        if hasCost:
            return self.get_depth()
        else:
            return 0

    def calculate_children(self, hasCost: bool):
        for action in actions:
            yield HeuristicNode(apply_action(self.state, turns[action]), self.heuristic_function, hasCost, action, self)

N = TypeVar('N', bound=Node)
class Tree(Generic[N]):
    def __init__(self, root: N):
        self.root = root
        self.visited: set[Cube] = set()
        self.queue: list[N] = []
        self.border: list[N] = []

    def bpa(self) -> Optional[N]:
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

    def _bpp(self, node: N, depth: int = 0, max_depth: Optional[int] = None) -> Optional[N]:
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

class HeuristicTree(Tree[HeuristicNode]):
    def __init__(self, root: HeuristicNode, hasCost: bool):
        super().__init__(root)
        self.hasCost = hasCost

    def global_heuristic(self) -> Optional[HeuristicNode]:
        self.border.append(self.root)
        while self.border:
            print(len(self.border))
            s = self.border.pop(0)
            assert s is not None
            if s.state.is_solved():
                return s
            for n in s.calculate_children(self.hasCost):
                if n.state not in self.visited:
                    s.add_child(n)
            for n in s.child_nodes:
                print(n.heuristic)
                self.visited.add(n.state)
                self.border.append(n)

                insort(self.border, n, key=lambda n:n.heuristic)
            #TODO: Reordenar border segun la heuristica del estado que etiqueta cada nodo
        self.visited.clear()  
        self.border.clear()  
        return None

    def get_min_heuristic(self, L: Iterable[HeuristicNode]) -> Optional[HeuristicNode]:
        return min(L, key=lambda n:n.heuristic)

    def _local_heuristic(self, L: list[HeuristicNode]) -> Optional[HeuristicNode]:
        s = self.get_min_heuristic(L)
        if s is None: return None
        print(s.heuristic)
        if s.heuristic == 0:             
            return s
        L.remove(s)
        for n in s.calculate_children(False):
            L.append(n)
        return self._local_heuristic(L)

    def local_heuristic(self) -> Optional[HeuristicNode]:
        L: list[HeuristicNode] = [self.root]
        return self._local_heuristic(L)
