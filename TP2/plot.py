from functools import partial
from typing import TypedDict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import sys
from os.path import splitext, basename
from more_itertools import padnone, take
from selection import SelectionDict
from crossover import CrossoverDict

from algorythm import GeneticAlgorythmDict

sns.set_theme()
sns.set(rc={'figure.figsize': (12, 5)})


def plot(base_name: str):
    Result = TypedDict('Result', fitnesses=list[int], stop_reason=str)
    OutputDataItem = TypedDict(
        'OutputDataItem', algorythm=GeneticAlgorythmDict, result=Result)
    OutputData = TypedDict(
        'OutputData', show_in_legend=list[str], results=list[OutputDataItem])

    with open(f'data/{base_name}.json', 'r') as file:
        data: OutputData = json.load(file)

    def crossover_string(data: CrossoverDict):
        type = data['type']
        params = data['params']
        if params:
            return f'{type} {params}'
        else:
            return type

    def selection_string(data: SelectionDict):
        type = data['type']
        params = data['params']
        if params:
            return f'{type} {params}'
        else:
            return type

    def convert_data():
        results = data['results']
        max_generations = max(len(d['result']['fitnesses']) for d in results)
        for d in results:
            for i, v in enumerate(take(max_generations, padnone(d['result']['fitnesses']))):
                yield i, d['algorythm']['population_count'], d['algorythm']['mutation']['probability'], crossover_string(d['algorythm']['crossover']), selection_string(d['algorythm']['selection']), v

    key_columns = ('Generation', 'Population Size', 'Mutation Probability',
                   'Crossover', 'Selection')
    value_columns = ('Fitness', )
    df = pd.DataFrame(convert_data(), columns=key_columns + value_columns)
    # df.index
    df.set_index(list(key_columns[1:] + ('Generation', )), inplace=True)
    # df.set_index('Generation', inplace=True)
    # df = df.explode('Fitness')
    # df = df.melt(id_vars='index', value_vars=value_columns)
    # df.plot()
    # sns.pointplot(data=df)
    # df.plot.(y='Fitness')#, x=['Generation', 'Population Size'])
    # df['Fitness'].plot()
    # df = df.drop('Crossover Type', axis=1)
    # df = df.drop('Selection Type', axis=1)
    groups = ['Population Size', 'Mutation Probability',
              'Crossover', 'Selection']
    legend_fields = data['show_in_legend']
    # groups
    df.groupby(level=legend_fields + ['Generation']) \
        .agg({'Fitness': 'mean'}) \
        .groupby(level=legend_fields)['Fitness'].plot(legend=True, use_index=False)
    plt.legend(title=', '.join(legend_fields),
               loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('Fitness')
    plt.xlabel('Generations')
    plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)
    # plt.yscale("log")
    fixed_fields = set(groups) - set(legend_fields)
    values = next(iter(df.to_dict()['Fitness'].keys()))
    d = dict(zip(key_columns[1:] + key_columns[:1], values))
    d = {k: str(d[k]) for k in fixed_fields}
    plt.title(' | '.join(f'{k}={v}' for k, v in d.items()))
    # df.groupby(groups)['Fitness'].plot.line(ylabel='Fitness', legend=True)

    plt.tight_layout()
    plt.savefig(f'plots/{base_name}.png', dpi=400)

    # sns.pointplot(df.groupby(list(key_columns[1:]))['Fitness'])
    plt.show()
    # # df = df.pivot(columns=value_columns)
    # # df = df.set_index(list(key_columns)).mean()
    # # df.set_index()
    # # df.columns.names = key_columns

    # # df.columns = pd.MultiIndex.from_tuples(df.columns, names=key_columns)
    # # sns.pointplot(data=df)
    # # df.plot()
    # plt.show()


files = sys.argv[1:]
if not files:
    files = ['plot_data.json']
for file in files:
    plot(splitext(basename(file))[0])
