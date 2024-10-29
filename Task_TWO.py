"""
Task Description:
               Solve a system of linear equations using numpy's linear algebra functions.

Output:

Solution for x and y: [1.75 1.5 ]


"""

import numpy as np

# Coefficient matrix A
A = np.array([[2, 1], [4, -6]])

# Constant vector b
b = np.array([5, -2])

# Solve the system of equations
solution = np.linalg.solve(A, b)

# Display the solution
print("Solution for x and y:", solution)
