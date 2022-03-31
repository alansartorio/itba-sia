import sys
from typing import Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tree import Node
from methods import Output

from plot_data_loader import ExecutionData, FullExecutionData
import plot_data_loader


def getDataFrame(values_by_cathegory: FullExecutionData, execution_map: Callable[[ExecutionData[Node]], Any]):
    depths = list(next(iter(values_by_cathegory.values())).keys())
    df = pd.DataFrame({
        'Solve Cost': np.repeat(np.array(depths), len(next(iter(next(iter(values_by_cathegory.values())).values())))),
    } | {
        method_name: [
            execution_map(execution) for execution_group in execution_groups.values() for execution in execution_group
        ] for method_name, execution_groups in values_by_cathegory.items()
    })

    dd = pd.melt(df, id_vars=[
        'Solve Cost'], value_vars=values_by_cathegory.keys(), var_name='Methods')

    return dd


def plot_boxes(values_by_cathegory: FullExecutionData):
    dd = getDataFrame(values_by_cathegory, lambda ex: ex.time)
    ax = sns.violinplot(x='Solve Cost', y='value',
                        data=dd, hue='Methods', linewidth=1)
    # ax = sns.pointplot(x='Solve Cost', y='value',
    # data=dd, hue='Methods', join=False)

    # for i,box in enumerate(ax.artists):
    # box.set_edgecolor('transparent')
    plt.show()


def plot_points(values_by_cathegory: FullExecutionData):
    dd = getDataFrame(values_by_cathegory, lambda ex: ex.output.border_count if type(
        ex.output) is Output else None)
    print(dd)

    def plot_all():
        # ax=sns.violinplot(x='Solve Cost',y='value',data=dd,hue='Methods', linewidth=1)
        ax = sns.pointplot(x='Solve Cost', y='value',
                           data=dd, hue='Methods')
        plt.yscale("log")
        plt.show()
    # ax = sns.boxplot(x='Solve Cost', y='value',

    def plot_single_depth(depth: int):
        ax = sns.violinplot(x='Solve Cost', y='value',
                            data=dd.loc[dd['Solve Cost'] == depth], hue='Methods')
        # ax = sns.boxplot(x='Solve Cost', y='value',
        # data=dd.loc[dd['Solve Cost'] == depth], hue='Methods')
        # plt.yscale("log")
        plt.show()

    plot_all()
    # plot_single_depth(2)


def plot_methods(values_by_cathegory: FullExecutionData, method_names: list[str], execution_map: Callable[[ExecutionData[Node]], Any], imagename: str = None, ylabel: str = 'value'):
    dd = getDataFrame(values_by_cathegory, execution_map)
    dd = dd.loc[dd['Methods'].map(method_names.__contains__)]
    dd = dd[['Solve Cost', 'Methods', 'value']]

    sns.set(rc={'figure.figsize':(9,5)})
    # ax = sns.boxplot(x='Solve Cost', y='value',
                       # data=dd, hue='Methods')
    ax = sns.pointplot(x='Solve Cost', y='value',
                       data=dd, hue='Methods', fmt='d')
    ax.ticklabel_format(style='plain', axis='y')
    plt.ylim(0)
    ax.grid(True)
    ax.set_ylabel(ylabel)
    ticks = list(dd['value'])
    # ax.set_yticks(ticks)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.ticklabel_format(useOffset=False)
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.15)
    plt.tight_layout()
    if imagename is not None:
        plt.savefig(imagename)
    plt.show()



def plot_many(values_by_cathegory: FullExecutionData, method_names: list[str], execution_maps: dict[str, Callable[[ExecutionData[Node]], Any]]):
    for plot, ex_map in execution_maps.items():
        plot_methods(values_by_cathegory, method_names, ex_map,
                     f'plots/{"_".join(method_names)}-{plot}.png', plot)


def plot_single_method(values_by_cathegory: FullExecutionData, method_name: str, execution_map: Callable[[ExecutionData[Node]], Any]):
    plot_methods(values_by_cathegory, [method_name], execution_map)

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


if __name__ == "__main__":
    sns.set_theme()
    filename = None
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    executions_by_method = plot_data_loader.load(filename)
    # plot_boxes(executions_by_method)
    # plot_points(executions_by_method)
    plot_many(executions_by_method, list(executions_by_method.keys()), {
              'Time': lambda ex: ex.time,
              'Expanded Nodes': lambda ex: ex.output.expanded_count,
              'Border Nodes': lambda ex: ex.output.border_count,
              'Found Solution Cost': lambda ex: ex.output.solution['depth'],
              })
    # plot_single_method(executions_by_method, 'A* (Move Count Combination)', lambda ex:ex.time)
