
import numpy as np
import numpy.typing as npt

def pca(data: npt.NDArray[np.float64]):
    # data = data.T
    data -= data.mean(axis=0)
    data /= data.std(axis=0)
    covariance_matrix = np.dot(data.T, data) / data.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    ids = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, ids]
    eigenvalues = eigenvalues[ids]

    ratios = eigenvalues / eigenvalues.sum()
    return ratios, eigenvectors, np.dot(data, eigenvectors)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import seaborn as sns

    
    fig, (ax0, ax1, ax2) = plt.subplots(1,3,figsize=(20,4))


    test_data = np.array([[0, 1, 2, 3, 4, 5], [3, 4, 5, 6, 7, 8], [5, 4, 3, 2, 1, 0], [1, 6, 2, 5, 2, 3]], dtype=np.float64).T
    sns.heatmap(test_data, annot=True, ax=ax0, center=0)
    ratios, _, matrix = pca(test_data)
    print(*ratios.cumsum())
    sns.heatmap(matrix, annot=True, ax=ax1, center=0)

    pca_pipe = make_pipeline(StandardScaler(), PCA())
    pca_pipe.fit(test_data)
    model_pca = pca_pipe.named_steps['pca']

    print(pca_pipe.transform(test_data))

    sns.heatmap(pca_pipe.transform(test_data), annot=True, ax=ax2, center=0)
    plt.show()
