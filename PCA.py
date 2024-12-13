import numpy as np 

def covariance_matrix(data):
    counter = 0
    x_mean = 0
    y_mean = 0
    for i in data:
        x_mean += i[0]
        y_mean += i[1]
        counter += 1
    x_mean = x_mean / counter
    y_mean = y_mean / counter
    print(x_mean)
    print(y_mean)

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
    return covariance_matrix


    
# def pca(data: np.ndarray, k: int) -> list[list[int|float]]:
# 	# Your code here
	
# 	return principal_components
data = np.array([[80, 70], [63, 20], [100, 50]])
eigenval , eigenvector = np.linalg.eig(covariance_matrix(data))
print(f"here is the eigenvalues :{eigenval}")

k = 1

covariance_matrix(data)