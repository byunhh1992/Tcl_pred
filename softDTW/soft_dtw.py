import numpy as np
from numba import njit



@njit
def squared_euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n: int = x.shape[0]
    m: int = y.shape[0]
    D = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            D[i, j] = (x[i] - y[j])**2
    
    return D



@njit
def soft_dtw_distance(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    gamma: float = 0.1
    D: np.ndarray = squared_euclidean_distance(x, y)
    n, m = D.shape
    R = np.full((n+1, m+1), np.inf)
    R[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            r = np.array([R[i-1, j], R[i, j-1], R[i-1, j-1]])
            # Following statements are replacing this equation: R[i, j] = D[i-1, j-1] + (-gamma*np.log(np.sum(np.exp(-r/gamma))))
            # These statements solved the log(0) problem
            max_r = np.max(-r/gamma)
            log_sum_exp = max_r + np.log(np.sum(np.exp(-r/gamma-max_r)))
            R[i, j] = D[i-1, j-1] + (-gamma*log_sum_exp)
    
    return R[n, m], R



@njit
def soft_dtw_gradient(x: np.ndarray, y: np.ndarray, R: np.ndarray) -> np.ndarray:
    gamma: float = 0.1
    n, m = x.shape[0], y.shape[0]
    E: np.ndarray = np.zeros_like(R)
    E[n, m] = 1

    for i in range(n, 0, -1):
        for j in range(m, 0, -1):
            r: np.ndarray = np.array([R[i-1, j], R[i, j-1], R[i-1, j-1]])
            max_r = np.max(-r/gamma); exp_r = np.exp((-r/gamma)-max_r); softmin = exp_r/np.sum(exp_r)
            # softmin = np.exp(-r/gamma) / np.sum(np.exp(-r/gamma))
            E[i-1, j-1] += softmin[2]*E[i, j]
            E[i-1, j] += softmin[0]*E[i, j]
            E[i, j-1] += softmin[1]*E[i, j]
    
    return E[1:, 1:]



@njit
def soft_dtw(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    distance, R = soft_dtw_distance(x, y)
    gradient = soft_dtw_gradient(x, y, R)
    return distance, gradient



def soft_dtw_test() -> None:
    Tcl: np.ndarray = np.load("Tcl_sampled.npz")["Tcl_sampled"]
    random_indices = np.random.choice(Tcl.shape[0], size=10, replace=False)
    Tcl_sorted: np.ndarray = Tcl[random_indices, :]
    n: int = Tcl_sorted.shape[0]
    
    # Distance & Gradient Test
    for i in range(n):
        for j in range(n):
            soft_dtw(Tcl_sorted[i], Tcl_sorted[j])



def main() -> None:
    soft_dtw_test()



if __name__ == "__main__":
    main()