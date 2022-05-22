import numpy as np
import pandas as pd
import csv

import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

data= []
with open('europe.csv', newline='\n') as File:
    reader = csv.reader(File)
    header = next(reader)
    print(header)
    for row in reader:
        data.append(row)


countries = [row[0] for row in data]
data =np.delete(data, 0, axis=1)

"""
print(countries)
for i in data:
    print(i)
"""

pca_pipe = make_pipeline(StandardScaler(), PCA())
pca_pipe.fit(data)
model_pca = pca_pipe.named_steps['pca']

df = pd.DataFrame(
    data = model_pca.components_,
    columns = header[1:],
    index = ['PC1','PC2','PC3','PC4', 'PC5','PC6','PC7']
)

print(df)

projections = pca_pipe.transform(X=data)
projections = pd.DataFrame(
    projections,
    columns = ['PC1','PC2','PC3','PC4', 'PC5','PC6','PC7'],
    index = countries
)
projections.head()

print(projections)


"""
rebuilt = pca_pipe.inverse_transform(X=projections)
rebuilt = pd.DataFrame(
    rebuilt,
    columns = header[1:],
    index = countries
)

print('Original values')
display(rebuilt.head())

print('Rebuilt values')
display(data.head())

"""
