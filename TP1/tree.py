from typing import Any, Callable, Generic, Iterable, Optional, TypeVar
from typing_extensions import Self
from cube import \
    Action, Cube, solved_cubes, turns, cube_rotations, \
    apply_action, apply_algorythm, parse_algorythm
from bisect import insort

# actions = "L L' L2 U U' U2 B B' B2".split()
actions = "R R' R2 U U' U2 F F' F2".split()
# actions = "R R' L L' U U' D D' F F' B B'".split()

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

    def get_branch(self) -> list[Self]:
        branch = self.parent.get_branch() if self.parent is not None else []
        branch.append(self)
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
    def __init__(self, root: N, is_solved: Callable[[Cube], bool] = Cube.is_solved, map_to_hashable: Callable[[Cube], Any] = lambda x:x):
        self.root = root
        self.visited: set[Cube] = set()
        self.queue: list[N] = []
        self.border: list[N] = []
        self.is_solved = is_solved
        self.map_to_hashable = map_to_hashable

    def bpa(self) -> Optional[N]:
        self.visited.add(self.map_to_hashable(self.root.state))
        self.queue.append(self.root)
        while self.queue:
            s = self.queue.pop(0)
            if self.is_solved(s.state):
                return s
            print(f'\r{s.get_depth()}', end="")
            for node in s.calculate_children():
                if self.map_to_hashable(node.state) not in self.visited:
                    s.add_child(node)
            for n in s.child_nodes:
                self.visited.add(self.map_to_hashable(n.state))
                self.queue.append(n)
        self.visited.clear()

        return None

    def bpp_non_recursive(self, max_depth: int = None):
        self.visited.add(self.map_to_hashable(self.root.state))
        stack = [self.root]
        while stack:
            s = stack.pop()
            if self.is_solved(s.state):
                return s
            if max_depth is not None and s.get_depth() > max_depth:
                continue
            print(f'\r{s.get_depth()}', end="")
            for node in s.calculate_children():
                if self.map_to_hashable(node.state) not in self.visited:
                    s.add_child(node)
            for n in s.child_nodes:
                self.visited.add(self.map_to_hashable(n.state))
                stack.append(n)
        self.visited.clear()

        return None

    def _bpp(self, node: N, depth: int = 0, max_depth: Optional[int] = None) -> Optional[N]:
        if max_depth is not None and depth > max_depth:
            return None
        # print(depth)
        if node not in self.visited:
            if self.is_solved(node.state):
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
    def __init__(self, root: HeuristicNode, hasCost: bool, is_solved: Callable[[Cube], bool]):
        super().__init__(root, is_solved)
        self.hasCost = hasCost

    def global_heuristic(self) -> Optional[HeuristicNode]:
        self.border.append(self.root)
        while self.border:
            # print(len(self.border))
            s = self.border.pop(0)
            print(s.heuristic)
            assert s is not None
            if self.is_solved(s.state):
                return s
            for n in s.calculate_children(self.hasCost):
                if n.state not in self.visited:
                    s.add_child(n)
            for n in s.child_nodes:
                self.visited.add(n.state)

                insort(self.border, n, key=lambda n:n.heuristic)
            #TODO: Reordenar border segun la heuristica del estado que etiqueta cada nodo
        self.visited.clear()  
        self.border.clear()  
        return None

    def get_min_heuristic(self, L: Iterable[HeuristicNode]) -> Optional[HeuristicNode]:
        return min(L, key=lambda n:n.heuristic)

    def _local_heuristic(self, L: list[HeuristicNode]) -> Optional[HeuristicNode]:
        while L:
            s = self.get_min_heuristic(L)
            assert s is not None
            if s.heuristic == 0 and not self.is_solved(s.state):
                print(s.heuristic, self.is_solved(s.state))
                print(s.state)
            if s.state.is_solved():
                return s
            Lsuccessors: set[HeuristicNode] = set()
            for child in s.calculate_children(False):
                if child.state not in self.visited:
                    self.visited.add(child.state)
                    Lsuccessors.add(child)
            sol = self._local_heuristic(list(Lsuccessors))
            if sol is not None:return sol
            L.remove(s)

    def local_heuristic(self) -> Optional[HeuristicNode]:
        L: list[HeuristicNode] = [self.root]
        sol = self._local_heuristic(L)
        self.visited.clear()
        return sol
