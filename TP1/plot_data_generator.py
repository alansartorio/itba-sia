from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from itertools import cycle, product, zip_longest
from math import ceil
from multiprocessing import cpu_count, get_context
from multiprocessing.pool import Pool
import signal
import time
from typing import Any, Callable

from more_itertools import take
from tqdm.std import tqdm
from timing import time_method
from tree import Node

from cube import Cube
from heuristics import manhattan_distance, move_count_combination, sticker_groups
import methods as search_methods
from methods import ExecutionData, Output, SolveData, time_solve_reduced
import plot_data_loader
import scramble_file

# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# sys.setrecursionlimit(1000)



SelectedMethods = dict[str, Callable[[Cube], Output]]


# def test_scramble(methods: SelectedMethods, cube: Cube) -> dict[str, ExecutionData]:
    # executions = {}
    # for method_name, method in methods.items():
        # executions[method_name] = time_solve_reduced(method, cube)
    # return executions


scrambles = scramble_file.load()


# def get_scramble(depth: int):
    # return scrambles[depth].pop()


# chosen_scrambles = [cube for cube, *_ in scrambles.values()]


# def test_sampled(methods: SelectedMethods, samples: list[Cube], pool: Pool):
    # method_executions = defaultdict(lambda: [])
    # # sample_count = 5
    # # samples = [get_scramble(scramble_depth) for _ in range(sample_count)]
    # for executions in tqdm(pool.imap_unordered(partial(test_scramble, methods), samples), leave=False, total=len(samples)):
        # # print(f'Sample: {sample}')
        # # executions = test_scramble(chosen_scrambles[scramble_depth])
        # # executions = test_scramble(sample)
        # for method_name, e in executions.items():
            # method_executions[method_name].append(e)

    # return method_executions

# def test(lst):
# print(type(lst[0]))
# if type(lst[0]) is list:
# return list
# else:
# return ExecutionData


# def deserialize_list(lst):
# if type(lst[0]) is list:
# return list_deserialization(list[list], lst, deserialization_func=JSONSerializer.deserialize)
# else:
# return list_deserialization(list[ExecutionData], lst, deserialization_func=JSONSerializer.deserialize)


def test(p):
    (method_name, method), (depth, cube) = p
    return (p, time_solve_reduced(method, cube))

if __name__ == '__main__':

    methods = {
        # "Breadth first (BPA)": search_methods.test_bpa,
        # "Depth first (BPP)": search_methods.test_bpp,
        # "BPPV": search_methods.test_bppv,
        # "Local Heuristic (Move Count Combination)": partial(search_methods.test_local_heuristics, move_count_combination),
        # "Global Heuristic (Move Count Combination)": partial(search_methods.test_global_heuristics, move_count_combination),
        # "A* (Move Count Combination)": partial(search_methods.test_global_heuristics_cost, move_count_combination),
        # "Local Heuristic (Sticker Groups)": partial(search_methods.test_local_heuristics, sticker_groups),
        # "Global Heuristic (Sticker Groups)": partial(search_methods.test_global_heuristics, sticker_groups),
        # "A* (Sticker Groups)": partial(search_methods.test_global_heuristics_cost, sticker_groups),
        # "Local Heuristic (Manhattan Distance)": partial(search_methods.test_local_heuristics, manhattan_distance),
        "Global Heuristic (Manhattan Distance)": partial(search_methods.test_global_heuristics, manhattan_distance),
        # "A* (Manhattan Distance)": partial(search_methods.test_global_heuristics_cost, manhattan_distance),
    }

    depths = list(range(0, 10))
    choose_scrambles = lambda depth:take(5, cycle(scrambles[depth]))

    all_tests = list(product(methods.items(), ((depth, cube) for depth in depths for cube in choose_scrambles(depth))))

    # print(all_tests)

    executions_by_method: defaultdict[str, dict[int, list[ExecutionData]]] = defaultdict(lambda: defaultdict(lambda: []))  # type: ignore
    try:
        with get_context('spawn').Pool(processes=cpu_count()) as pool:
            i = pool.imap_unordered(test, all_tests)
            i = tqdm(i, total=len(all_tests), mininterval=0, )
            for ((method_name, method), (depth, cube)), execution in i:
                # print(method_name, depth, execution.time)
                executions_by_method[method_name][depth].append(execution)
            # for count in tqdm(counts, leave=False):
                # # print(f'Scramble depth: {count}')
                # depth = count
                # samples = take(10, cycle(scrambles[depth]))
                # results = test_sampled(methods, samples, pool)
                # for method_name, e in results.items():
                    # executions_by_method[method_name][depth] = e
    except KeyboardInterrupt:
        print("Terminated by user.")
    finally:
        executions_by_method: dict[str, dict[int, list[ExecutionData]]] = dict(executions_by_method)  # type: ignore

        # Uncomment for saving results to json file
        plot_data_loader.save(executions_by_method)
