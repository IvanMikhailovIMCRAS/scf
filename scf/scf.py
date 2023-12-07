import matplotlib.pyplot as plt
import numpy as np


def flat_brush(
    N: int, sigma: float, n_layers: int, eta: float = 0.1, precision: float = 1e-7
) -> np.ndarray:
    """_summary_

    Args:
        N (int): polymerization degree
        sigma (float): grafting density
        n_layers (int): number of layers
        eta (float, optional): covergence size step. Defaults to 0.1.
        precision (float, optional): precision of calculation. Defaults to 1e-7.

    Returns:
        phi: np.ndarray: volume fraction profile
    """
    alpha = np.zeros(n_layers + 2, dtype=float)
    W_b = np.zeros(n_layers + 2, dtype=float)
    phi = np.zeros(n_layers + 2, dtype=float)
    phi_s = np.zeros(n_layers + 2, dtype=float)
    Gf = np.zeros(shape=(n_layers + 2, N), dtype=float)
    Gb = np.zeros(shape=(n_layers + 2, N), dtype=float)

    deviation = 1.0
    lambda_f = 1 / 6
    lambda_b = 1 / 6
    lambda_0 = 1.0 - (lambda_f + lambda_b)

    count = 0

    while deviation > precision**2:
        count += 1
        W_b = np.exp(-alpha)
        # print(W_b)
        phi_s = W_b
        Gf[:, :] = 0.0
        Gb[:, :] = 0.0
        Gf[1, 0] = 1.0
        Gb[:, N - 1] = 1.0

        for s in range(1, N):
            for z in range(1, n_layers + 1):
                Gf[z, s] = W_b[z] * (
                    lambda_f * Gf[z - 1, s - 1]
                    + lambda_0 * Gf[z, s - 1]
                    + lambda_b * Gf[z + 1, s - 1]
                )
        for s in range(N - 2, -1, -1):
            for z in range(1, n_layers + 1):
                Gb[z, s] = W_b[z] * (
                    lambda_b * Gb[z - 1, s + 1]
                    + lambda_0 * Gb[z, s + 1]
                    + lambda_f * Gb[z + 1, s + 1]
                )
        phi[:] = 0.0
        for s in range(N):
            for z in range(1, n_layers + 1):
                phi[z] += Gb[z, s] * Gf[z, s] / W_b[z]

        phi = phi / np.sum(phi) * N * sigma
        alpha -= eta * (1 - phi - phi_s)

        deviation = np.sum((1 - phi - phi_s) ** 2)
        # deviation = 0.0
    print(f"itteration number = {count}")
    return phi


if __name__ == "__main__":
    phi = flat_brush(N=100, sigma=0.1, n_layers=70)
    plt.plot(phi)
    plt.show()
