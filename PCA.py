import numpy as np 

def covariance_matrix(data: np.ndarray) -> np.ndarray:
    counter = 0
    x_mean = 0
    y_mean = 0
    for i in data:
        x_mean += i[0]
        y_mean += i[1]
        counter += 1
    x_mean = x_mean / counter
    y_mean = y_mean / counter

    variance_x = 0
    variance_y = 0
    for i in data:
        variance_x += (i[0] - x_mean) ** 2
        variance_y += (i[1] - y_mean) ** 2
    variance_x = variance_x / (counter - 1)
    variance_y = variance_y / (counter - 1)

    covariance_xy = 0
    for i in data:
        covariance_xy += (i[0] - x_mean) * (i[1] - y_mean)
    covariance_xy = covariance_xy / (counter - 1)

    covariance_matrix = [[variance_x, covariance_xy], [covariance_xy, variance_y]]
    return np.array(covariance_matrix)

def eigen_values_and_vectors(covariance_matrix: np.ndarray):
    eigenval, eigenvec = np.linalg.eig(covariance_matrix)
    return eigenval, eigenvec

def pca(data: np.ndarray, k: int) -> list[list[float]]:
    data_standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    cov = np.cov(data_standardized, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, idx]
    
    principal_components = eigenvectors_sorted[:, :k]
    
    return np.round(principal_components, 4).tolist()

