import json
import scramble_file
from collections import defaultdict
import pandas as pd
import seaborn as sns
from typing import Callable
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
from tqdm.std import tqdm
from cube import Cube, solved_cubes
from scramble import generate_scramble_by_optimum_solve_depth, generate_scrambled, generate_scrambles_by_solve_depth
from tree import Tree, Node, HeuristicTree, HeuristicNode
from heuristics import move_count_combination, sticker_groups, manhattan_distance
import time
from methods import *
from multiprocessing import Process
from more_itertools import take

import sys

sys.setrecursionlimit(20000)

methods = {
    # "bpp": test_bpp,
    # "bppv": test_bppv,
    # "bpa": test_bpa,
    "Local Heuristic (Move Count Combination)": lambda cube: test_local_heuristics(cube, move_count_combination),
    # "Global Heuristic (Move Count Combination)": lambda cube: test_global_heuristics(cube, move_count_combination),
    # "A* (Move Count Combination)": lambda cube: test_global_heuristics_cost(cube, move_count_combination),
    "Local Heuristic (Sticker Groups)": lambda cube: test_local_heuristics(cube, sticker_groups),
    # "Global Heuristic (Sticker Groups)": lambda cube: test_global_heuristics(cube, sticker_groups),
    # "A* (Sticker Groups)": lambda cube: test_global_heuristics_cost(cube, sticker_groups),
    "Local Heuristic (Manhattan Distance)": lambda cube: test_local_heuristics(cube, manhattan_distance),
    # "Global Heuristic (Manhattan Distance)": lambda cube: test_global_heuristics(cube, manhattan_distance),
    # "A* (Manhattan Distance)": lambda cube: test_global_heuristics_cost(cube, manhattan_distance),
}


def thread_time():
    return time.clock_gettime(time.CLOCK_REALTIME)


def timeout(func, timeout: float):
    p = Process(target=func)
    p.start()
    p.join(timeout=timeout)
    p.terminate()


def time_method(method):
    start = thread_time()
    output = timeout(method, 0.1)
    end = thread_time()
    return end - start, output


def test_scramble(cube: Cube):
    times = {}
    for method_name, method in tqdm(methods.items(), leave=False):
        # print(f'Testing: {method_name}')
        t, output = time_method(lambda: method(cube))
        # if output is None or not output.state.is_solved():
        # output = None
        # raise Exception("??")
        times[method_name] = t
    return times


scrambles = scramble_file.load()


def get_scramble(depth: int):
    return scrambles[depth].pop()


chosen_scrambles = [cube for cube, *_ in scrambles.values()]


def test_sampled(scramble_depth: int):
    method_times = defaultdict(lambda: [])
    sample_count = 20
    for sample in tqdm(range(sample_count), leave=False):
        # print(f'Sample: {sample}')
        times = test_scramble(chosen_scrambles[scramble_depth])
        for method_name, t in times.items():
            method_times[method_name].append(t)

    return method_times


counts = list(range(0, 13))
times_by_method: defaultdict[str, list[list[float]]] = defaultdict(lambda: [])
for count in tqdm(counts, leave=False):
    # print(f'Scramble depth: {count}')
    results = test_sampled(count)
    for method_name, times in results.items():
        times_by_method[method_name].append(times)

json.dump(times_by_method, open('times_by_method.json', 'w'))
times_by_method = json.load(open('times_by_method.json'))

def plot_boxes(values_by_cathegory: dict[str, list[list[float]]]):
    df = pd.DataFrame({'Solve Depth': np.repeat(counts, len(list(values_by_cathegory.values())
                                                            [0][0])),
                       } | {method_name: [time for time_group in time_groups for time in time_group] for method_name, time_groups in values_by_cathegory.items()}, )

    dd=pd.melt(df,id_vars=['Solve Depth'],value_vars=values_by_cathegory.keys(),var_name='Methods')
    ax=sns.violinplot(x='Solve Depth',y='value',data=dd,hue='Methods', linewidth=1)

    # for i,box in enumerate(ax.artists):
        # box.set_edgecolor('transparent')
    plt.show()

# def plot_boxes(values_by_cathegory: dict[str, Iterable[Iterable[float]]]):
    # def set_box_colors(box, color):
    # for prop in ['boxes', 'caps', 'whiskers', 'fliers', 'medians']:
    # plt.setp(box[prop], color=color)

    # ax = plt.axes()
    # colors = ['red', 'blue', 'green']
    # for i, ((method_name, line), color) in enumerate(zip(values_by_cathegory.items(), colors)):
    # box = plt.boxplot(line, positions=[i + o * (len(values_by_cathegory) + 2) for o in counts], widths=0.6, labels=counts)
    # set_box_colors(box, color)

    # for method_name, color in zip(values_by_cathegory.keys(), colors):
    # plt.plot([], c=color, label=method_name)
    # # ax.set_xticklabels(counts)
    # # ax.set_xticks(np.linspace(1.5, 7.5, len(counts)))
    # # plt.legend(list(values_by_cathegory.keys()), list(values_by_cathegory.values()))
    # plt.legend(loc="upper left")
    # plt.xticks(counts)
    # plt.show()


plot_boxes(dict(times_by_method))
