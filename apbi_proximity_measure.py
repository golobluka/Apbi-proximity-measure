from ortools.linear_solver import pywraplp
import numpy as np
from scipy.optimize import approx_fprime

import cvxpy as cp
import numpy as np
import itertools
import warnings



def optimize_u(A, B, G):
    """
    Solve the QP:
        minimize  || A*u1 + B^T * uJ ||^2 + (1 - u1)^2 + uJ^T (G G^T) uJ
        subject to u1 >= 0,  uJ >= 0

    Parameters
    ----------
    A : np.ndarray of shape (n,)
        Gradient of phi(x). (If you have A as (1,n), flatten it to (n,) first.)
    B : np.ndarray of shape (J, n)
        Gradients of g_j(x). Each row B[j, :] is grad(g_j).
    G : np.ndarray of shape (J,)
        Vector of constraint values g_j(x).

    Returns
    -------
    (u1_val, uJ_val, obj_val) : (float, np.ndarray, float)
        u1_val : solution for u1 (scalar)
        uJ_val : solution for uJ (1D array of length J)
        obj_val: final value of the objective
                 If infeasible or unbounded, returns None for all.
    """
    # Dimensions
    n = A.shape[0]
    J = B.shape[0]

    # Define decision variables
    u1 = cp.Variable(nonneg=True)     # scalar, must be >= 0
    uJ = cp.Variable(J, nonneg=True)  # J-dimensional vector, each >= 0

    # Expression for (A^T * u1 + B^T * uJ)
    # shape: (n,) once evaluated
    x_expr = A * u1 + B.T @ uJ

    # (1) term:  || x_expr ||^2
    cost1 = cp.sum_squares(x_expr)

    # (2) term:  (1 - u1)^2
    cost2 = cp.square(1 - u1)

    # (3) term:  uJ^T (G G^T) uJ
    # G G^T is (J x J), so use quad_form:
    GGt = np.outer(G, G)  # shape (J, J)
    cost3 = cp.quad_form(uJ, GGt)

    # Full objective = cost1 + cost2 + cost3
    objective = cost1 + cost2 + cost3

    # Set up and solve
    problem = cp.Problem(cp.Minimize(objective))
    try:
        problem.solve()  # By default uses an appropriate QP solver
    except cp.error.SolverError:
        # Could happen if no solver is installed or solver fails
        return (None, None, None)

    # Check status
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return (None, None, None)

    # Extract solution
    u1_val = u1.value
    uJ_val = uJ.value
    obj_val = problem.value

    return (u1_val, uJ_val)





def solve_linear_system(LHS, RHS):
    """
    Solves the linear system LHS * x = RHS using OR-Tools.

    Parameters:
        LHS: numpy array, the left-hand side matrix
        RHS: numpy array, the right-hand side vector

    Returns:
        x: numpy array, the solution vector
    """
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return None

    num_vars = LHS.shape[1]
    x_vars = [solver.NumVar(0.0, solver.infinity(), f'x{i}') for i in range(num_vars)]

    # Add equations
    for i in range(LHS.shape[0]):
        constraint_expr = solver.Sum([LHS[i, j] * x_vars[j] for j in range(num_vars)])
        solver.Add(constraint_expr == RHS[i])

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        solution = np.array([x_vars[i].solution_value() for i in range(num_vars)])
        return solution
    else:
        print('The solver could not find an optimal solution.')
        return None
    


def solve_subset_case(A1, BJ, G, active_set):
    """
    Builds the "full" stationarity matrix (LHS_full) and vector (RHS_full),
    then extracts only the rows/columns for the given `active_set`.
    Solves that smaller linear system using `solve_linear_system()`.
    Returns a (1+J)-dimensional vector x_full that is zero outside active_set.

    """
    J = BJ.shape[0]  # number of g_i constraints
    # 1. Build the full (1+J) x (1+J) system
    # Compute matrices
    # A1^T A1: scalar
    A1A1T = A1 @ A1.T

    A1BJT = A1 @ BJ.T      # shape (1, J)

    BJA1T = A1BJT.T               # shape (J, 1)

    BJBJT = BJ @ (BJ.T)     # shape (J, J)

    # G G^T: shape (J, J)
    GGT = G @ G.T  # shape (J, J)

    # Construct LHS matrix
    LHS_top_left = A1A1T + 1     # scalar
    LHS_top_right = A1BJT        # shape (1, J)
    LHS_bottom_left = BJA1T      # shape (J, 1)
    LHS_bottom_right = BJBJT + GGT  # shape (J, J)

    LHS = np.block([
        [LHS_top_left, LHS_top_right],
        [LHS_bottom_left, LHS_bottom_right]
    ])  # shape (1+J, 1+J)
    #Print the shape of LHS
    print("The shape of LHS is: ",LHS.shape)


    RHS = np.concatenate(([1], np.zeros(J)))  # shape (1+J,)



    # 2. Sort the active_set so we have consistent indexing
    S = sorted(active_set)

    # 3. Extract submatrix/subvector: we keep only
    #    - rows i in S
    #    - columns j in S
    #    With NumPy, we can do:
    print("The value of S is: ",S)
    print("The value of LHS is: ",LHS)
    print("The value of RHS is: ",RHS)
    LHS_sub = LHS[np.ix_(S, S)]  # shape = (|S| x |S|)
    RHS_sub = RHS[S]            # shape = (|S|,)

    # 4. Solve the reduced system
    x_S = solve_linear_system(LHS_sub, RHS_sub)
    if x_S is None:
        return None

    # 5. Embed solution back into a (1+J)-vector
    x_full = np.zeros(1 + J)
    x_full[S] = x_S  # This assigns x_S[i] to x_full[S[i]] elementwise

    return x_full


def calculate_epsilon_from_u(u, g_values):
    """
    Calculate the proximity measure ε_A from the solution u1, uJ.
    Inputs:
        u = [u1, uJ]: list of scalars, the solution to the optimization problem
        g_values: numpy array, the values of g(x)
    Returns:
        epsilon_A: scalar, the proximity measure or None if infeasible
    """
    # Verify the condition in Eq. (47)
    u1 = u[0]       # scalar
    uJ = np.array(u[1:]).reshape(-1, 1)     # shape (J,)

    if u1 is None or uJ[0] is None:
        print('Failed to solve the optimization problem.')
        return None

    GTuJ = (g_values.T @ uJ)[0, 0]  # scalar


    condition = u1 - GTuJ * (1 - GTuJ) <= 1

    # Determine feasibility
    feasible = np.all(g_values.flatten() <= 1e-9)

    #Print the solutions


    if feasible:
        if condition:
            print("Condition is satisfied")
            # First scenario, calculate ε_FSk using Eq. (45)
            epsilon_FSk = 1 - u1 - (GTuJ)**2
            epsilon_A = epsilon_FSk

            print("The value of epsilon_FSk is: ",epsilon_FSk)
        else:
            print("Condition is not satisfied")
            # Second scenario
            # Calculate ε_FSk using Eq. (45)
            epsilon_FSk = 1 - u1 - (GTuJ)**2
            # Calculate ε_Adjk using Eq. (48)
            epsilon_Adjk = - GTuJ
            # Calculate ε_Pk using Eq. (49)
            GTG = (g_values.T @ g_values)[0, 0]  # scalar
            numerator = np.dot(g_values.T, epsilon_FSk * g_values - uJ.reshape(-1,1))[0,0]
            denominator = 1 + GTG
            epsilon_Pk = numerator / denominator
            # Calculate ε_SSk using Eq. (50)
            epsilon_A = (epsilon_FSk + epsilon_Adjk + epsilon_Pk) / 3

    else:
        # x is infeasible, calculate ε_A using Eq. (51)
        epsilon_A = 1 + np.sum(np.maximum(g_values.flatten(), 0)**2)


    return epsilon_A



def solve_kkt_all_combinations(A1, BJ, G):
    """
    Enumerate all possible subsets of indices that might be positive.
    Return a list of (u, fval) for each feasible KKT solution.
    """
    n_u = 1 + BJ.shape[0]  # u1 plus the J-dim vector (J+1 total)

    # We'll store (best_u, best_fval) as we go
    # or keep all feasible solutions in a list
    all_solutions = []

    # Each subset S of {0, 1, ..., n_u-1} indicates which u_i are "strictly positive".
    # We let 0 correspond to u1, 1..J correspond to the components of uJ.
    full_index_set = range(n_u)

    for k in range(n_u+1):
        # generate all subsets of size k
        for S in itertools.combinations(full_index_set, k):
            S = set(S)  # to allow easy membership checks
            if len(S) == 0 or 0 not in S:
                continue  # skip the empty set

            # 1) Solve for u_i in S using stationarity with lambda_i=0
            #    for i in S, and fix u_i=0 for i not in S.

            # We'll create a function that does that.
            u_candidate = solve_subset_case(A1, BJ, G, S)

            # 2) If solve_subset_case returns None, it means no solution was found
            if u_candidate is None:
                continue

            # 3) Check feasibility:
            #    - u_candidate[i] == 0 for i not in S
            #    - u_candidate[i] >  0 for i in S
            #    - all entries must be >= 0 (just in case of floating epsilon)
            feasibility_ok = True
            tol = 1e-12
            for i in range(n_u):
                if i in S:
                    # strictly positive
                    if not (u_candidate[i] > -tol):
                        feasibility_ok = False
                        break


            if not feasibility_ok:
                continue



            # 5) Store it
            all_solutions.append(u_candidate)

    # After enumerating everything, pick the best or return them all
    # E.g., to get the best:
    if len(all_solutions) == 0:
        return None  # no feasible solutions found
    
    return all_solutions

def apbi_kktpm(x, phi_func, g_funcs, implementation = "paper_version"):
    """
    Calculates the APBI-KKTPM value for a given x. 

    Parameters:
        x: numpy array, the point at which to evaluate
        phi_func: function that returns phi(x)
        g_funcs: list of functions, g_j(x)
        implementation: string, the type of implementation to use. There are three options: 
            - "paper_version": the implementation as described in the paper
            - "quadratic_optimization": the implementation using quadratic optimization, avoiding linear system
            - "complicate_linear_system": the implementation using a more involved linear system solver

    Returns:
        epsilon_A: the APBI-KKTPM value
    """
    n = len(x)
    J = len(g_funcs)

    # Calculate gradients ∇φ(x) and ∇g_j(x)

    grad_phi = approx_fprime(x, phi_func, 1e-6)


    g_values = []
    grad_g = []
    if len(g_funcs) == 0: # If there is no constraints make a trivial constaraint g(x) = -1
        g_values = np.array([[-1]])
        grad_g = np.array([0] * len(x)).reshape(1, -1) 
    else:
        for g_func in g_funcs:
            g_value = g_func(x)   # grad_g_j: n x 1
            grad_g_j = approx_fprime(x, g_func, 1e-6)
            g_values.append(g_value)
            grad_g.append(grad_g_j.reshape(-1))
        g_values = np.array(g_values).reshape(J, 1)  # shape (J, 1)
    grad_g = np.array(grad_g)                 # shape (J, n)

    # Convert grad_phi to column vector
    A1 = grad_phi.reshape(1, n)   # shape (1, n)

    # BJ is grad_g
    BJ = grad_g                   # shape (J, n)
    
    # Compute matrices
    # A1^T A1: scalar
    A1A1T = A1 @ A1.T

    A1BJT = A1 @ BJ.T      # shape (1, J)

    BJA1T = A1BJT.T               # shape (J, 1)

    BJBJT = BJ @ (BJ.T)     # shape (J, J)

    # G G^T: shape (J, J)
    GGT = g_values @ g_values.T  # shape (J, J)


    # Construct LHS matrix
    LHS_top_left = A1A1T + 1     # scalar
    LHS_top_right = A1BJT        # shape (1, J)
    LHS_bottom_left = BJA1T      # shape (J, 1)
    LHS_bottom_right = BJBJT + GGT  # shape (J, J)

    LHS = np.block([
        [LHS_top_left, LHS_top_right],
        [LHS_bottom_left, LHS_bottom_right]
    ])  # shape (1+J, 1+J)

    # RHS vector
    if len(g_funcs) == 0:
        RHS = np.concatenate(([1], np.zeros(1)))  # special case when there are no constraints
    else:
        RHS = np.concatenate(([1], np.zeros(J)))  # shape (1+J,)

    
    # Here we have devide into different implementations
    #_____________________________________________________________________________

    if implementation == "paper_version":
        u = solve_linear_system(LHS, RHS)
        if u is None:
            print('Failed to solve the linear system. We now try with quadratic system.')
            return None
        return calculate_epsilon_from_u(u, g_values)

   #_____________________________________________________________________________ 
    
    elif implementation == "quadratic_optimization":
        u = optimize_u(A1, BJ, g_values) 
        if u is None:
            print('Failed to solve the linear system. We now try with quadratic system.')
            return None
        return calculate_epsilon_from_u(u, g_values)


    #_____________________________________________________________________________


    elif implementation == "complicate_linear_system":
        all_solutions = solve_kkt_all_combinations(A1, BJ, g_values)

        if all_solutions is None:
            print('Failed to solve the linear system.') 
            return None
    
        best_solution = 1 # just to initialize
        for u in all_solutions:
            epsilon_A = calculate_epsilon_from_u(u, g_values)
            if epsilon_A < -1e-9:
                warnings.warn(f"Negative proximity measure: {epsilon_A}. We ignored this value")
            elif epsilon_A < best_solution:
                best_solution = np.abs(epsilon_A)
        return best_solution
    
    else:
        raise ValueError("Invalid implementation string. Choose one of 'paper_version', 'quadratic_optimization', 'complicate_linear_system'")


