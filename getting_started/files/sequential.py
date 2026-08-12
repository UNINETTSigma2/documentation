# assumed to be sequential.py
import numpy as np

def approximate_pi():
    """
    Approximate pi by throwing random points in a unit square and 
    counting how many fall inside the unit circle.
    Returns:
        float: Approximation of pi.
    """

    # Number of random points to generate
    num_points = 2000000

    # Generate random points in the square [-1, 1] x [-1, 1]
    x = np.random.uniform(-1.0, 1.0, num_points)
    y = np.random.uniform(-1.0, 1.0, num_points)

    # Count how many points fall inside the unit circle
    inside_circle = np.sum(x**2 + y**2 < 1)

    # Approximate pi using the ratio of points inside the circle to total points
    pi_approximation = (inside_circle / num_points) * 4

    return pi_approximation

if __name__ == "__main__":
    results = [approximate_pi() for _ in range(100)]
