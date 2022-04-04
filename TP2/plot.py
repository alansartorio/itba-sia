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
        yield (d['algorythm']['population_count'], d['algorythm']['mutation']['probability'], d['algorythm']['crossover']['type'], d['algorythm']['selection']['type']), take(max_generations, padnone(d['result']['fitnesses']))


key_columns = ('Population Size', 'Mutation Probability',
               'Crossover Type', 'Selection Type')
value_columns = ('Fitness', )
df = pd.DataFrame(dict(convert_data()))
df.columns.names = key_columns
# df['Keys'] = df[df.columns[:-1]].apply(
# lambda x: ','.join(x.dropna().astype(str)),
# axis=1
# )
# df = df[['Keys', 'Fitness']]
# print(df)
# print(type(df.index))
# df = df.melt(value_vars=df.columns)
# print(df.columns.names)
df.columns = pd.MultiIndex.from_tuples(df.columns, names=key_columns)
# df.index = pd.MultiIndex.from_frame(df)
# df = df.melt(id_vars=[(0, 1, 2, 3)])
# print(df)
df.plot()
# df.rename(columns=lambda tup:','.join(map(str, tup)), inplace=True)
# print(df.columns.groupby(lambda v:v[0] == 10))
# df.columns = df.columns.map(partial(map, str)).map('|'.join).str.strip('|')
# print(df.columns)
# sns.lineplot(data=df.columns)
plt.show()
