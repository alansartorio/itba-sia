from typing import Any, Callable, Generic, Iterable, Optional, TypeVar
from typing_extensions import Self
from cube import \
    Action, Cube, solved_cubes, turns, cube_rotations, \
    apply_action, apply_algorythm, parse_algorythm
from bisect import insort
import math

# actions = "L L' L2 U U' U2 B B' B2".split()
actions = "R R' U U' F F'".split()
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
        node = self
        branch = []
        while node:
            branch.insert(0, node)
            node = node.parent
        return branch

    def get_depth(self) -> int:
        return self.parent.get_depth() + 1 if self.parent is not None else 0

    def get_state_string(self) -> str:
        return self.state.get_state_string() + " " + str(self.get_depth())

    def __repr__(self) -> str:
        return self.state.get_state_string()

    def clone(self) -> 'Cube':
        return Node.parse(repr(self))

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, str):
            return self.get_state_string() == __o
        if isinstance(__o, Node):
            return self == __o.get_state_string()
        return False

    def __hash__(self) -> int:
        return hash(repr(self))

class HeuristicNode(Node):
    def __init__(self, state: Cube, heuristic_function: HeuristicFunction, hasCost: bool, action: Optional[str] = None, parent: Optional[Self] = None):
        super().__init__(state, action=action, parent=parent)
        self.heuristic_function = heuristic_function
        heuristic = self.calculate_heuristic()
        self.heuristic = self.cost(hasCost) + heuristic

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
        self.bpp_visited: set[Node] = set()
        self.visited: set[Cube] = set()
        self.queue: list[N] = []
        self.border: list[N] = []
        self.stack: list[N] = []
        self.border_count = 0
        self.is_solved = is_solved
        self.map_to_hashable = map_to_hashable


    def bpa(self) -> tuple[Optional[N], int]:
        self.visited.add(self.map_to_hashable(self.root.state))
        self.queue.append(self.root)
        while self.queue:
            s = self.queue.pop(0)
            if self.is_solved(s.state):
                self.border_count = len(self.queue)
                return s, len(self.visited)
            for node in s.calculate_children():
                if self.map_to_hashable(node.state) not in self.visited:
                    s.add_child(node) # type: ignore
            for n in s.child_nodes:
                self.visited.add(self.map_to_hashable(n.state))
                self.queue.append(n) # type: ignore
        visited_count = len(self.visited)
        self.visited.clear()

        self.border_count = len(self.queue)
        return None, visited_count

    def bpp(self, max_depth: int = None):
        if not self.stack:
            self.stack = [self.root]
            self.bpp_visited.add(self.map_to_hashable(self.root))
        while self.stack:
            s = self.stack.pop()
            
            if self.is_solved(s.state):
                self.border_count = len(self.stack)
                return s, len(self.bpp_visited)
            if max_depth is not None and s.get_depth() >= max_depth:
                continue
            
            for node in s.calculate_children():
                if self.map_to_hashable(node) not in self.bpp_visited:
                    self.bpp_visited.add(self.map_to_hashable(node))
                    s.add_child(node)
                

            for node in reversed(s.child_nodes):
                self.stack.append(node)
        
        self.border_count = len(self.stack)
        visited_count = len(self.bpp_visited) #TODO Revisar que se este haciendo bien en casos como bppv
        self.bpp_visited.clear()
        self.root.child_nodes.clear()
        #No solution found -> Stack is empty, no need to clear

        return None, visited_count

    def bppv(self, max_depth: int):
        improving = False
        last_solution = None
        last_visited_count = 0
        while max_depth >= 0:
            sol, visited_count = self.bpp(max_depth)
            if sol:
                last_solution = sol
                last_visited_count = visited_count
                max_depth -= 1
                improving = True
            elif not improving:
                left = max_depth
                right = 14
                max_depth = math.ceil((left + right)/2)
            else:
                break

        
        self.visited.clear()
        return last_solution, last_visited_count

class HeuristicTree(Tree[HeuristicNode]):
    def __init__(self, root: HeuristicNode, hasCost: bool, is_solved: Callable[[Cube], bool] = Cube.is_solved):
        super().__init__(root, is_solved)
        self.hasCost = hasCost

    def global_heuristic(self) -> tuple[Optional[HeuristicNode], int]:
        self.border.append(self.root)
        while self.border:
            # print(len(self.border))
            s = self.border.pop(0)
            # print(s.heuristic)
            assert s is not None
            if self.is_solved(s.state):
                self.border_count = len(self.border)
                return s, len(self.visited)
            for n in s.calculate_children(self.hasCost):
                if n.state not in self.visited:
                    s.add_child(n)
            for n in s.child_nodes:
                self.visited.add(n.state)

                insort(self.border, n, key=lambda n:n.heuristic)
            #TODO: Reordenar border segun la heuristica del estado que etiqueta cada nodo
        visited_count = len(self.visited)
        self.visited.clear()
        self.border_count = len(self.border)
        self.border.clear()
        return None, visited_count

    def get_min_heuristic(self, L: Iterable[HeuristicNode]) -> Optional[HeuristicNode]:
        return min(L, key=lambda n:n.heuristic)

    def _local_heuristic(self, L: list[HeuristicNode]) -> tuple[Optional[HeuristicNode], int]:
        while L:
            s = self.get_min_heuristic(L)
            self.border_count -= 1
            assert s is not None
            if s.heuristic == 0 and not self.is_solved(s.state):
                # print(s.heuristic, self.is_solved(s.state))
                # print(s.state)
                pass
            if s.state.is_solved():
                return s, len(self.visited)
            for child in s.calculate_children(False):
                if child.state not in self.visited:
                    self.visited.add(child.state)
                    s.add_child(child)
                    self.border_count += 1
            sol, visited_count = self._local_heuristic(s.child_nodes)
            if sol is not None:return sol, visited_count
            L.remove(s)
        return None, len(self.visited)

    def local_heuristic(self) -> tuple[Optional[HeuristicNode], int]:
        self.border_count = 1
        L: list[HeuristicNode] = [self.root]
        sol = self._local_heuristic(L)
        self.visited.clear()
        return sol
