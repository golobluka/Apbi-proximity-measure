import numpy as np
from scipy.optimize import minimize, approx_fprime
import warnings



def pbi_kktpm(x, phi_func, g_functions, epsilon=1e-8):
    """
    Solve the PBI-KKT proximity measure subproblem:
    
        Minimize: 
            eps_k + sum_{j=1}^J [ (u_j * g_j(x))^2 ]

        subject to:
            1) || grad_phi(x) + sum_j [u_j * grad_g_j(x)] ||_2 <= eps_k
            2) sum_j [u_j*g_j(x)] >= - eps_k
            3) u_j >= 0, for j = 1..J

    Parameters
    ----------
    x : np.ndarray
        The point x in R^n.
    phi_func : callable
        A function phi_func(x) -> float.
    g_functions : list of callables
        List of constraint functions g_j(x) -> float for j in 1..J.
    epsilon : float, optional
        Step size for finite-difference gradient approximation (default = 1e-8).

    Returns
    -------
    float
        The optimal eps_k (epsilon_k^*) if successful.
        -1 if the optimizer fails to find a solution.
    """

    ############################################################################
    # 1. Finite-Difference Gradient Approximation
    ############################################################################


    # Compute grad_phi(x)


    grad_phi = approx_fprime(x, phi_func, epsilon)


    # Compute g_j(x) and grad_g_j(x) for j=1..J
    J = len(g_functions)
    g_vals = [g_func(x) for g_func in g_functions]
    grad_g_list = [approx_fprime(x, g_func, epsilon) for g_func in g_functions]

    ############################################################################
    # 2. Define the Subproblem in scipy.optimize form
    ############################################################################
    # Variables: vars_ = [eps_k, u_1, ..., u_J]
    # We want to minimize:
    #   eps_k + sum_j (u_j*g_j(x))^2
    def objective(vars_):
        eps_k = vars_[0]
        u = vars_[1:]
        penalty = 0.0
        for j in range(J):
            penalty += (u[j] * g_vals[j])**2
        return eps_k

    # Constraint 1 (ineq in SLSQP means >= 0):
    #   eps_k - || grad_phi + sum_j [u_j * grad_g_j] || >= 0
    def c1_norm(vars_):
        eps_k = vars_[0]
        u = vars_[1:]
        grad_sum = np.copy(grad_phi)
        for j in range(J):
            grad_sum += u[j] * grad_g_list[j]
        return eps_k - np.linalg.norm(grad_sum, 2) ** 2

    # Constraint 2:
    #   sum_j [u_j*g_j] + eps_k >= 0
    def c2_linear(vars_):
        eps_k = vars_[0]
        u = vars_[1:]
        return sum(u[j] * g_vals[j] for j in range(J)) + eps_k

    # Constraints 3 (J of them): u_j >= 0
    def make_constraint_u(j):
        return {'type': 'ineq', 
                'fun': lambda vars_, j=j: vars_[j]}

    # Collect them all
    constraints = [
        {'type': 'ineq', 'fun': c1_norm},
        {'type': 'ineq', 'fun': c2_linear}
    ]
    # u_j >= 0 constraints
    for j in range(J+1):
        constraints.append(make_constraint_u(j))

    ############################################################################
    # 3. Solve the Subproblem
    ############################################################################
    # Initial guess: eps_k=1.0, all u_j=1.0
    init_guess = np.ones(J + 1, dtype=float)
    init_guess[0] = 1

    res = minimize(
        fun=objective,
        x0=init_guess,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-7}
    )

    # Check for success
    if not res.success:
        print("Optimization failed:", res.message)
        return None  # Return None if the solver fails

    # If successful, print and return the optimized eps_k
    eps_star = res.x[0]
    if eps_star < -1e-9:
        warnings.warn(f"Optimal epsilon_k is negative: {eps_star:.6g}. We changed its value to 10.")
        eps_star = 10
    
    return eps_star

