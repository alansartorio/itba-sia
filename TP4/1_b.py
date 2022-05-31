from pca import pca
import seaborn as sns
import numpy as np
import pandas as pd
import csv
import statistics

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
sns.set_theme()


colors = sns.color_palette('pastel')


class Plot:
    def __init__(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        bars = ax.barh(range(7), np.zeros(7), color=colors)
        ax.set_yticks(range(7))
        ax.set_yticklabels(original_df.columns.values)
        ax.set_xlim((-0.7, 0.7))
        plt.tight_layout()
        plt.show(block=False)

        self.bars = bars
        self.fig = fig
        self.ax = ax

    def update(self, first):
        print(first)
        first = first / (first.max() - first.min())

        for rect, h in zip(self.bars, first):
            rect.set_width(h)
        self.fig.canvas.flush_events()
        self.fig.canvas.draw_idle()

    def finish(self, first):
        self.update(first)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.pie(abs(first), labels=original_df.columns.values,
               autopct='%.0f%%', colors=colors)
        plt.tight_layout()


class Oja:
    def __init__(self, data, registers, learning_rate: float, size: int) -> None:
        self.data = np.array(data)
        self.registers = registers
        self.size = size
        self.learning_rate = learning_rate
        self.weights = np.zeros([self.size])

    def randomize_weights(self, amplitude: float = 0.01):
        self.weights[:] = (np.random.rand(
            *self.weights.shape) * 2 - 1) * amplitude

    def train(self, data):
        for i in range(self.size):
            # s = np.sum(data[i] * self.weights)
            s = np.inner(data[i], self.weights)
            self.weights += self.learning_rate * \
                s * (data[i] - s * self.weights)


def standarize(data):
    from sklearn.preprocessing import StandardScaler

    data = np.array(data)
    # data -= data.mean(0)
    # data /= data.std(0)
    data = StandardScaler().fit_transform(data)
    return data


def other_library():
    from pca import pca

    _, vec, _ = pca(data)

    return np.array([v[0] for v in vec])


data = []
with open('europe.csv', newline='\n') as File:
    reader = csv.reader(File)
    header = next(reader)
    for country, *row in reader:
        data.append([country] + list(map(float, row)))

countries = [row[0] for row in data]
data = [row[1:] for row in data]
original_df = pd.DataFrame(data, columns=header[1:], index=countries)

# real_order = ['Luxembourg', 'Switzerland', 'Norway', 'Netherlands', 'Ireland', 'Iceland',
# 'Austria', 'Denmark', 'Sweden', 'Italy', 'Belgium', 'Germany', 'United Kingdom'
# ',Finland' ',Czech Republic', 'Spain', 'Slovenia', 'Portugal', 'Slovakia',
# 'Greece', 'Croatia', 'Hungary', 'Poland', 'Lithuania', 'Latvia', 'Estonia',
# 'Bulgaria', 'Ukraine']
# print(real_order)
# print(original_df.index.values)
# original_df.sort_index(key=lambda r:pd.Index([real_order.index(r.values[i]) for i in r]), inplace=True)
# print(original_df)

# print(original_df.columns)
plot = Plot()


data = standarize(data)
standarized = pd.DataFrame(
    data, columns=original_df.columns, index=original_df.index)
real_component = other_library()
standarized['real_PC1'] = standarized.apply(
    lambda row: np.inner(row, real_component), axis=1)
standarized.sort_values(by='real_PC1', inplace=True)
with_real = standarized.copy()
print(standarized)
standarized.drop('real_PC1', inplace=True, axis=1)
net = Oja(data, countries, 0.01, 7)
net.randomize_weights(1)
_, vec, _ = pca(data)
# net.weights[:] = vec[0]
errors = []
for i in range(50):
    net.train(data)
    print(net.weights)
    plot.update(net.weights)
    errors.append(np.sum((net.weights - real_component)**2))
    # plt.pause(0.1)
print(net.weights)
plot.finish(net.weights)

standarized['calculated_PC1'] = standarized.apply(
    lambda row: np.inner(row, net.weights), axis=1)

# plot_first_PCA(net.weights)
def other_library_sk():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    pca_pipe = make_pipeline(StandardScaler(), PCA())
    pca_pipe.fit(data)
    model_pca = pca_pipe.named_steps['pca']
    vec = model_pca.components_[0]
    Plot().finish(vec)

    # plot_first_PCA(model_pca.components_[0, :])
# Plot().finish(real_component)

# other_library()

# fig = plt.figure()
# ax = fig.add_subplot()
# ax.plot(errors)
# plt.show()




def plot_var(data, var: str):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    r = np.arange(len(data))
    ax.set_yticks(r, labels=data.index)
    # ax.invert_yaxis()

    ax.set_title(var)
    ax.set_xlabel(f'{var} value')
    ax.barh(r, np.array(data[var]))
    # ax2 = ax.twiny()
    # gdp = np.array(merged['Area'])
    # ax2.plot(gdp, r, marker="D", alpha=0.6, color='red', linestyle="", markersize=3)
    # plt.grid(None)
    plt.tight_layout()


plot_var(standarized, 'calculated_PC1')
plot_var(with_real, 'real_PC1')
plt.show()
