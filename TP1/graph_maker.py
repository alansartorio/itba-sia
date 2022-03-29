from dataclasses_serialization.serializer_base import noop_serialization, noop_deserialization, dict_serialization, dict_deserialization, list_deserialization, Serializer
from dataclasses import dataclass
from functools import partialmethod
import signal
from contextlib import contextmanager
import json
import scramble_file
from collections import defaultdict
import pandas as pd
import seaborn as sns
from typing import Callable, NewType
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
from methods import Output

import sys

# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

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


def raise_timeout(signum, frame):
    raise TimeoutError


@contextmanager
def timeout_block(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGPROF, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.setitimer(signal.ITIMER_PROF, time)
    # signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGPROF, signal.SIG_IGN)


def timeout(func, timeout: float):
    with timeout_block(timeout):
        return func()
    return TimeoutError


def time_method(method: Callable[[], Any]):
    start = thread_time()
    output = timeout(method, 0.1)
    end = thread_time()
    return end - start, output


N = TypeVar('N', bound=Node)


@dataclass
class ExecutionData(Generic[N]):
    time: float
    output: Output[N]


def test_scramble(cube: Cube) -> dict[str, ExecutionData]:
    executions = {}
    for method_name, method in tqdm(methods.items(), leave=False):
        # print(f'Testing: {method_name}')
        t, output = time_method(lambda: method(cube))

        assert output is not None
        assert output is TimeoutError or output.solution.state.is_solved()
        # output = None
        # raise Exception("??")
        executions[method_name] = ExecutionData(t, output)
    return executions


scrambles = scramble_file.load()


def get_scramble(depth: int):
    return scrambles[depth].pop()


chosen_scrambles = [cube for cube, *_ in scrambles.values()]


def test_sampled(scramble_depth: int):
    method_executions = defaultdict(lambda: [])
    sample_count = 20
    for sample in tqdm(range(sample_count), leave=False):
        # print(f'Sample: {sample}')
        executions = test_scramble(chosen_scrambles[scramble_depth])
        for method_name, e in executions.items():
            method_executions[method_name].append(e)

    return method_executions


counts = list(range(0, 13))
# executions_by_method: defaultdict[str, list[list[ExecutionData]]] = defaultdict(lambda: [])  # type: ignore
# for count in tqdm(counts, leave=False):
    # # print(f'Scramble depth: {count}')
    # results = test_sampled(count)
    # for method_name, e in results.items():
        # executions_by_method[method_name].append(e)
# executions_by_method: dict[str, list[list[ExecutionData]]] = dict(
    # executions_by_method)  # type: ignore

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


JSONSerializer = Serializer(
    serialization_functions={
        dict: lambda dct: dict_serialization(dct, key_serialization_func=JSONSerializer.serialize, value_serialization_func=JSONSerializer.serialize),
        list: lambda lst: list(map(JSONSerializer.serialize, lst)),
        Node: lambda node: {"depth": node.get_depth()},
        type(TimeoutError): lambda _: "timeout",
        Cube: lambda cube: repr(cube),
        (str, int, float, bool, type(None)): noop_serialization
    },
    deserialization_functions={
        # # dict: lambda cls, dct: None,
        # dict: lambda cls, dct: dict_deserialization(cls, dct, key_deserialization_func=JSONSerializer.deserialize, value_deserialization_func=JSONSerializer.deserialize),
        # # list: lambda cls, lst: list_deserialization(cls, lst, deserialization_func=JSONSerializer.deserialize(list if type(lst[0]) is list else ExecutionData)),
        # list: lambda cls, lst: deserialize_list(lst),
        # Cube: lambda cls, cube: Cube.parse(cube),
        # # Node: lambda cls, node: Node(JSONSerializer.deserialize(Cube, node['state']), node['action'], JSONSerializer.deserialize(Node.__init__, node['parent'])),
        Output: lambda cls, output: Output(output['solution'], output['expanded_count'], output['border_count']) if output != "timeout" else TimeoutError,
        ExecutionData: lambda cls, execution_data: ExecutionData(execution_data['time'], JSONSerializer.deserialize(Output, execution_data['output'])),
        (str, int, float, bool, type(None)): noop_deserialization
    }
)

# pickle.dump(executions_by_method, open('executions_by_method.pickle', 'wb'))
# executions_by_method = json.load(open('executions_by_method.pickle', 'rb'))

# print(JSONSerializer.serialize(executions_by_method))

# Uncomment for saving results to json file
# json.dump(JSONSerializer.serialize(executions_by_method),
          # open('executions_by_method.json', 'w'), indent=2)
data = json.load(open('executions_by_method.json'))
executions_by_method = {method_name: [[JSONSerializer.deserialize(ExecutionData, execution_data) for execution_data in execution_data_group]
                                      for execution_data_group in execution_data_groups] for method_name, execution_data_groups in data.items()}
# executions_by_method = JSONSerializer.deserialize(
# dict, json.load(open('executions_by_method.json')))


def plot_boxes(values_by_cathegory: dict[str, list[list[ExecutionData]]]):
    df = pd.DataFrame({'Solve Depth': np.repeat(np.array(counts), len(list(values_by_cathegory.values())
                                                                      [0][0])),
                       } | {method_name: [execution.time for execution_group in execution_groups for execution in execution_group] for method_name, execution_groups in values_by_cathegory.items()}, )

    dd = pd.melt(df, id_vars=[
                 'Solve Depth'], value_vars=values_by_cathegory.keys(), var_name='Methods')
    ax=sns.violinplot(x='Solve Depth',y='value',data=dd,hue='Methods', linewidth=1)
    # ax = sns.pointplot(x='Solve Depth', y='value',
                       # data=dd, hue='Methods', join=False)

    # for i,box in enumerate(ax.artists):
    # box.set_edgecolor('transparent')
    plt.show()


def plot_points(values_by_cathegory: dict[str, list[list[ExecutionData]]]):
    df = pd.DataFrame({'Solve Depth': np.repeat(np.array(counts), len(list(values_by_cathegory.values())
                                                                      [0][0])),
                       } | {method_name: [execution.output.expanded_count if type(execution.output) is Output else None for execution_group in execution_groups for execution in execution_group] for method_name, execution_groups in values_by_cathegory.items()}, )

    dd = pd.melt(df, id_vars=[
                 'Solve Depth'], value_vars=values_by_cathegory.keys(), var_name='Methods')
    # ax=sns.violinplot(x='Solve Depth',y='value',data=dd,hue='Methods', linewidth=1)
    ax = sns.pointplot(x='Solve Depth', y='value',
                       data=dd, hue='Methods', join=False)
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


plot_boxes(dict(executions_by_method))
plot_points(dict(executions_by_method))
