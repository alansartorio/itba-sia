
from cube import Cube
from tree import *


def test_bpp(cube: Cube):
    tree = Tree(Node(cube))
    sol = tree.bpp_non_recursive()
    return sol

def test_bpa(cube: Cube):
    tree = Tree(Node(cube))
    sol = tree.bpa()
    return sol

def test_bppv(cube: Cube):
    tree = Tree(Node(cube))
    sol = tree.bppv(5)
    return sol

def test_local_heuristics(cube: Cube, heuristic_function: Callable[[Cube], float]):
    tree = HeuristicTree(HeuristicNode(cube, heuristic_function, False), False)
    sol = tree.local_heuristic()
    return sol

def test_global_heuristics(cube: Cube, heuristic_function: Callable[[Cube], float]):
    tree = HeuristicTree(HeuristicNode(cube, heuristic_function, False), False)
    sol = tree.global_heuristic()
    return sol

def test_global_heuristics_cost(cube: Cube, heuristic_function: Callable[[Cube], float]):
    tree = HeuristicTree(HeuristicNode(cube, heuristic_function, True), True)
    sol = tree.global_heuristic()
    return sol


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