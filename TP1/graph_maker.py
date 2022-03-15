from collections import defaultdict
from typing import Callable
import matplotlib.pyplot as plt
from tqdm.std import tqdm
from cube import Cube, generate_scrambled, solved_cubes
from tree import Tree, Node, HeuristicTree, HeuristicNode
from heuristics import move_count_combination, sticker_groups
import time
from methods import *

import sys

sys.setrecursionlimit(20000)

def test_local_heuristics_0(cube: Cube):
    return test_local_heuristics(cube, move_count_combination)
def test_global_heuristics_0(cube: Cube):
    return test_global_heuristics(cube, move_count_combination)
def test_global_heuristics_cost_0(cube: Cube):
    return test_global_heuristics_cost(cube, move_count_combination)

methods = {
        # "bpp": test_bpp,
        # "bppv": test_bppv,
        # "bpa": test_bpa,
        "local_heuristic": test_local_heuristics_0,
        "global_heuristic": test_global_heuristics_0,
        # "global_heuristic_cost": test_global_heuristics_cost_0
}

def thread_time():
    return time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)

def time_method(method):
    start = thread_time()
    output = method()
    end = thread_time()
    return end - start, output

def test_scramble(cube: Cube):
    times = {}
    for method_name, method in tqdm(methods.items(), leave=False):
        # print(f'Testing: {method_name}')
        t, output = time_method(lambda:method(cube))
        if output is None or not output.state.is_solved():
            output = None
            raise Exception("??")
        times[method_name] = t
    return times


def test_sampled(scramble_count: int):
    method_sums = defaultdict(lambda:0)
    sample_count = 20
    for sample in tqdm(range(sample_count), leave=False):
        # print(f'Sample: {sample}')
        times = test_scramble(generate_scrambled(scramble_count))
        for method_name, t in times.items():
            method_sums[method_name] += t


    method_averages = {}
    for method_name, method_sum in method_sums.items():
        method_averages[method_name] = method_sum / sample_count

    return method_averages

times_by_method = defaultdict(lambda:[])
counts = list(range(0, 10))
for count in tqdm(counts, leave=False):
    # print(f'Scramble depth: {count}')
    results = test_sampled(count)
    for method_name, average in results.items():
        times_by_method[method_name].append(average)

for method_name, line in times_by_method.items():
    plt.plot(line, label=method_name)
# plt.legend(list(times_by_method.keys()), list(times_by_method.values()))
plt.legend(loc="upper left")
plt.xticks(counts)
plt.show()

