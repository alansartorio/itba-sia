from functools import partial
from typing import TypedDict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from more_itertools import padnone, take

from algorythm import GeneticAlgorythmDict

sns.set_theme()

Result = TypedDict('Result', fitnesses=list[int], stop_reason=str)
OutputData = TypedDict(
    'OutputData', algorythm=GeneticAlgorythmDict, result=Result)

with open('plot_data.json', 'r') as file:
    data: list[OutputData] = json.load(file)


def convert_data():
    max_generations = max(len(d['result']['fitnesses']) for d in data)
    for d in data:
        for i, v in enumerate(take(max_generations, padnone(d['result']['fitnesses']))):
            yield i, d['algorythm']['population_count'], d['algorythm']['mutation']['probability'], d['algorythm']['crossover']['type'], d['algorythm']['selection']['type'], v


key_columns = ('Generation', 'Population Size', 'Mutation Probability',
               'Crossover Type', 'Selection Type')
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
groups = ['Population Size', 'Mutation Probability', 'Crossover Type', 'Selection Type']
df.groupby(level=groups)['Fitness'].plot(legend=True, use_index=False)
plt.legend(title=', '.join(groups), loc='center left', bbox_to_anchor=(1, 0.5))
plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)
plt.tight_layout()
# df.groupby(groups)['Fitness'].plot.line(ylabel='Fitness', legend=True)

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
