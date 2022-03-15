
from cube import Cube
from tree import *


def test_bpp(cube: Cube):
    tree = Tree(Node(cube))
    sol, expanded = tree.bpp_non_recursive()
    return sol, expanded, tree.border_count

def test_bpa(cube: Cube):
    tree = Tree(Node(cube))
    sol, expanded = tree.bpa()
    return sol, expanded, tree.border_count

def test_bppv(cube: Cube):
    tree = Tree(Node(cube))
    sol, expanded = tree.bppv(5)
    return sol, expanded, tree.border_count

def test_local_heuristics(cube: Cube, heuristic_function: Callable[[Cube], float]):
    tree = HeuristicTree(HeuristicNode(cube, heuristic_function, False), False)
    sol, expanded = tree.local_heuristic()
    return sol, expanded, tree.border_count

def test_global_heuristics(cube: Cube, heuristic_function: Callable[[Cube], float]):
    tree = HeuristicTree(HeuristicNode(cube, heuristic_function, False), False)
    sol, expanded = tree.global_heuristic()
    return sol, expanded, tree.border_count

def test_global_heuristics_cost(cube: Cube, heuristic_function: Callable[[Cube], float]):
    tree = HeuristicTree(HeuristicNode(cube, heuristic_function, True), True)
    sol, expanded = tree.global_heuristic()
    return sol, expanded, tree.border_count


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
