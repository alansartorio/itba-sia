
from dataclasses import dataclass
from cube import Cube
from tree import *

N = TypeVar('N', bound=Node)

@dataclass
class Output(Generic[N]):
    solution: Optional[N]
    expanded_count: int
    border_count: int

def test_bpp(cube: Cube):
    tree = Tree(Node(cube))
    sol, expanded = tree.bpp()
    return Output(sol, expanded, tree.border_count)

def test_bpa(cube: Cube):
    tree = Tree(Node(cube))
    sol, expanded = tree.bpa()
    return Output(sol, expanded, tree.border_count)

def test_bppv(cube: Cube):
    tree = Tree(Node(cube))
    sol, expanded = tree.bppv(7)
    return Output(sol, expanded, tree.border_count)

def test_local_heuristics(cube: Cube, heuristic_function: Callable[[Cube], float]):
    tree = HeuristicTree(HeuristicNode(cube, heuristic_function, False), False)
    sol, expanded = tree.local_heuristic()
    return Output(sol, expanded, tree.border_count)

def test_global_heuristics(cube: Cube, heuristic_function: Callable[[Cube], float]):
    tree = HeuristicTree(HeuristicNode(cube, heuristic_function, False), False)
    sol, expanded = tree.global_heuristic()
    return Output(sol, expanded, tree.border_count)

def test_global_heuristics_cost(cube: Cube, heuristic_function: Callable[[Cube], float]):
    tree = HeuristicTree(HeuristicNode(cube, heuristic_function, True), True)
    sol, expanded = tree.global_heuristic()
    return Output(sol, expanded, tree.border_count)


methods = {
        "bpp": test_bpp,
        "bppv": test_bppv,
        "bpa": test_bpa
}

heuristic_methods = {
        "local_heuristic": test_local_heuristics,
        "global_heuristic": test_global_heuristics,
        "global_heuristic_cost": test_global_heuristics_cost
}
