
from dataclasses import dataclass
from functools import partial
from typing import Union
from timing import time_method
from cube import Cube
from tree import *

N = TypeVar('N', bound=Node)


@dataclass
class SolveData:
    depth: int

@dataclass
class Output(Generic[N]):
    solution: Optional[Union[N, SolveData]]
    expanded_count: int
    border_count: int

@dataclass
class ExecutionData(Generic[N]):
    time: float
    output: Output[N]


FullExecutionData = dict[str, dict[int, list[ExecutionData]]]


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
    sol, expanded = tree.bppv(4)
    return Output(sol, expanded, tree.border_count)

def test_local_heuristics(heuristic_function: Callable[[Cube], float], cube: Cube):
    tree = HeuristicTree(HeuristicNode(cube, heuristic_function, False), False)
    sol, expanded = tree.local_heuristic()
    return Output(sol, expanded, tree.border_count)

def test_global_heuristics(heuristic_function: Callable[[Cube], float], cube: Cube):
    tree = HeuristicTree(HeuristicNode(cube, heuristic_function, False), False)
    sol, expanded = tree.global_heuristic()
    return Output(sol, expanded, tree.border_count)

def test_global_heuristics_cost(heuristic_function: Callable[[Cube], float], cube: Cube):
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


def time_solve(method: Callable[[Cube], Output], cube: Cube, timeout: float = None) -> ExecutionData[Node]:
    t, output = time_method(partial(method, cube), timeout)
    assert output is not None
    assert output is TimeoutError or output.solution.state.is_solved()
    return ExecutionData(t, output)

def time_solve_reduced(method: Callable[[Cube], Output], cube: Cube, timeout: float = None) -> ExecutionData:
    execution = time_solve(method, cube, timeout)

    if execution.output is not TimeoutError:
        depth = execution.output.solution.get_depth()
        # if depth > 10:
        execution.output.solution = SolveData(depth)

    return execution

