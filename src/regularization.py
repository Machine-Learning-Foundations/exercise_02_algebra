"""Regularization proof of concept."""
import matplotlib.pyplot as plt
import numpy as np
import pandas


def set_up_point_matrix(points: int, degree: int) -> tuple:
    """Set up a point matrix to fit a polynomial.

    The matrix should have to following form:
    [x_1**0       x_1**1      ...  x_1**(degree-1)
     x_2**0       x_2**1      ...  x_1**(degree-1)
     x_3**0       x_3**1      ...  x_1**(degree-1)
     ...          ...         ...  ...
     x_points**0  x_points**1 ...  x_points**(degree-1)]

    Args:
        points (int): The number of points to evaluate the polynomial on.
        degree (int): The degree of the polynomial.

    Returns:
        tuple: The polynomial point matrix A.
    """
    x_axis = np.linspace(0, 1, num=points)
    mat_a = np.zeros((points, degree))
    # TODO: implement me.
    return x_axis, mat_a


if __name__ == "__main__":
    b_noise = pandas.read_csv('./data/noisy_signal.tab')
    plt.plot(b_noise)
    plt.show()