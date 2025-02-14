# I have two lists of problem ly customly defined problems and pymoo problems.
#____________________________________________________________________________


from pymoo.problems import get_problem
from pymoo.core.problem import Problem
import numpy as np

# Here I define my costum problems
#___________________________________________________________________________-


class MyProblem0(Problem):
    def __init__(self):
        super().__init__(n_var=1,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array([-5]),
                         xu=np.array([5]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0] ** 2  # Objective 1
        f2 = (x[:, 0] - 2) ** 2

        g1 = 0.5 - x[:, 0]  # Constraint g1(x) < 0
        out["F"] = np.column_stack([f1, f2])

        out["G"] = np.column_stack([g1])

    def pareto_front(self):
        return np.array([[0.5**2]])
    def pareto_set(self):
        return np.array([[0.5]])
    




class MyProblem1(Problem):
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=2,
                         xl=np.array([-5, -5]),
                         xu=np.array([5, 5]))

    def _evaluate(self, x, out, *args, **kwargs):

        f1 = x[:, 0] ** 2  # Objective 1
        f2 = x[:, 1] ** 2  # Objective 2
        g1 = 1 - x[:, 0]  # Constraint g1(x) < 0
        g2 = 1 - x[:, 1]  
        out["F"] = np.column_stack([f1, f2])

        out["G"] = np.column_stack([g1, g2])

    def pareto_front(self):
        return np.array([[1,1]])
    def pareto_set(self):
        return np.array([[1,1]])
    

    
    
class MyProblem4(Problem):
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([-5, -5]),
                         xu=np.array([5, 5]))

    def _evaluate(self, x, out, *args, **kwargs):


        
        f1 = x[:, 0] ** 2 + (x[:,1] - 2) ** 2  # Objective 1
        f2 = x[:, 1] ** 2  # Objective 2


        out["F"] = np.column_stack([f1, f2])
    
    
    def pareto_front(self):
        # Define list of points from (0,0) to (0,2)

        points = np.array([[0, x/50] for x in range(101)])
        #Evaluate f1 and f2 of each point
        pareto_front = np.array([[x[0]**2 + (x[1] - 2)**2, x[1]**2] for x in points])
        return np.array(pareto_front)
    def pareto_set(self):
        # Define list of points from (0,0) to (0,2)

        pareto_set = np.array([[0, x/50] for x in range(101)])
        return pareto_set

class MyProblem2(Problem):
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([-5, -5]),
                         xu=np.array([5, 5]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 =  x[:, 0] ** 2 # Objective 1
        f2 = (x[:, 0] - 2) ** 2 # Objective 2 (all zeros, same shape as f1)
        out["F"] = np.column_stack([f1, f2])

    def pareto_set(self):
        return np.array([[i/100,0] for i in range(200)])
    
    def pareto_front(self):
        pareto_set = [[i/100,0] for i in range(200)]
        return np.array([[x[0] ** 2, (x[0] - 2) ** 2] for x in pareto_set])
    
class MyProblem3(Problem):
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array([-0.5, -0.5]),
                         xu=np.array([3,3]))

    def _evaluate(self, x, out, *args, **kwargs):
        # Objective 1: Sum of squares
        f1 = x[:, 0]**2 + x[:, 1]**2  
        # Objective 2: Difference of squares
        f2 = (x[:, 0] - 2.5)**2 + (x[:, 1] - 2.5)**2  
        # Constraint: Ensure sum of variables does not exceed a value
        g1 = x[:, 0] + x[:, 1] - 4

        # Output results
        out["F"] = np.column_stack([f1, f2])
        out["G"] = g1.reshape(-1, 1)

    def pareto_set(self):
        # Example Pareto front (e.g., two extremes of the range)
        pareto_set =  [[x/100,  x/100] for x in range(201)]

        return np.array(pareto_set)
    def pareto_front(self):
        # Example Pareto front (e.g., two extremes of the range)
        pareto_set =  [[x/100, x/100] for x in range(201)]
        pareto_front = [[x**2 + y**2, (x - 2.5)**2 + (y - 2.5)**2] for x, y in pareto_set]
        return np.array(pareto_front)


    
class MyProblem3_2(Problem):
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array([-0.5, -0.5]),
                         xu=np.array([3,3]))

    def _evaluate(self, x, out, *args, **kwargs):
        # Objective 1: Sum of squares
        f1 = x[:, 0]**2 + x[:, 1]**2  
        # Objective 2: Difference of squares
        f2 = (x[:, 0] - 2)**2 + (x[:, 1])**2  
        # Constraint: Ensure sum of variables does not exceed a value
        g1 = x[:, 0]  - 1

        # Output results
        out["F"] = np.column_stack([f1, f2])
        out["G"] = g1.reshape(-1, 1)

    def pareto_set(self):
        # Example Pareto front (e.g., two extremes of the range)
        pareto_set =  [[x/20, 0] for x in range(21)]

        return np.array(pareto_set)
    


    def pareto_front(self):
        # Example Pareto front (e.g., two extremes of the range)
        pareto_set =  [[x/20, 0] for x in range(21)]
        pareto_front = [[x**2 + y**2, (x - 2)**2 + (y)**2] for x, y in pareto_set]
        return np.array(pareto_front)
    

class MyCircleProblem(Problem):

    def __init__(self):
        # n_var=2 (x and y), n_obj=2, n_eq_constr=1 for the circle equation
        # n_ieq_constr=0 since we have no inequality constraints
        # Lower and upper bounds: [0,1] for both x and y
        super().__init__(
            n_var=2,
            n_obj=2,
            n_ieq_constr=0,
            n_eq_constr=1,
            xl=np.array([0, 0]),
            xu=np.array([1, 1])
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # Objectives:
        f1 = x[:, 0]  # x
        f2 = x[:, 1]  # y

        # Equality constraint: x^2 + y^2 = 1
        # Usually in pymoo, we specify constraints in the form "h = 0" for equality constraints
        h1 = x[:, 0]**2 + x[:, 1]**2 - 1

        out["F"] = np.column_stack([f1, f2])  # shape (n_solutions, 2)
        out["H"] = np.column_stack([h1])      # shape (n_solutions, 1)

    def pareto_set(self, n_points=100):
        """
        Return a discretized version of the quarter circle in decision space.
        """
        t = np.linspace(0, np.pi/2, n_points)
        xs = np.cos(t)
        ys = np.sin(t)
        return np.column_stack([xs, ys])

    def pareto_front(self, n_points=100):
        """
        Because f1 = x and f2 = y, the Pareto front is identical
        to the decision space arc for this problem.
        """
        return self.pareto_set(n_points)

    


class MyProblem5(Problem):
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array([-2, -2]),  # lower bounds
                         xu=np.array([4, 4]))   # upper bounds

    def _evaluate(self, x, out, *args, **kwargs):
        # Objective 1: A "bowl"-shaped function offset in one dimension
        #    f1 = (x1 - 1)^2 + x2^2
        # This shifts the minimum of the function in the x1-direction.
        f1 = (x[:, 0] - 1.0)**2 + x[:, 1]**2

        # Objective 2: Combined trigonometric and polynomial behavior
        #    f2 = sin(x1) + (x2 - 2)^2
        # This mixes a sinusoidal term with a parabola in x2.
        f2 = np.sin(x[:, 0]) + (x[:, 1] - 2.0)**2

        # Constraint: Some circle-like boundary (or ring) to restrict feasible region
        #    g1 = x1^2 + x2^2 - 8 <= 0
        # This forces (x1, x2) to lie within or on a circle of radius sqrt(8).
        g1 = x[:, 0]**2 + x[:, 1]**2 - 8

        out["F"] = np.column_stack([f1, f2])
        out["G"] = g1.reshape(-1, 1)

    def pareto_set(self):
        return np.array([[0,0]])


    def pareto_front(self):
        
        pareto_front = [[ 1.06085495e+01 ,-9.99999817e-01],
 [ 5.03869671e-08,  4.84233346e+00],
 [ 1.99236355e-01,  3.25291543e+00],
 [ 6.57412048e-02,  3.88193540e+00],
 [ 3.21662281e+00,  2.50579563e-01],
 [ 1.01780188e+00,  1.74335320e+00],
 [ 1.92857864e-03,  4.67519405e+00],
 [ 3.31722187e-01,  2.87017444e+00],
 [ 8.35528889e+00, -9.16964575e-01],
 [ 3.69466139e+00,  4.79576150e-02],
 [ 2.03843214e-02,  4.28505960e+00],
 [ 2.35390311e+00,  6.87471985e-01],
 [ 8.47548930e-01,  1.92340789e+00],
 [ 2.86308576e+00,  5.19212322e-01],
 [ 7.87811884e+00, -8.73909973e-01],
 [ 3.96847748e+00, -3.25105153e-02],
 [ 3.40597952e+00,  1.75623833e-01],
 [ 2.54169233e+00,  5.72539224e-01],
 [ 7.83076239e-01,  2.09230590e+00],
 [ 1.39366105e+00,  1.36271218e+00],
 [ 5.79625468e+00, -5.61471800e-01],
 [ 4.56545481e-01,  2.56312470e+00],
 [ 5.58212544e+00, -4.65719701e-01],
 [ 9.95084834e+00, -9.92921118e-01],
 [ 4.21197368e+00, -1.39471904e-01],
 [ 4.65090219e+00, -2.76139597e-01],
 [ 6.81189199e+00, -7.22514005e-01],
 [ 4.41315720e+00, -1.97078260e-01],
 [ 1.03355370e+01, -9.97862174e-01],
 [ 6.08358925e+00, -6.16761389e-01],
 [ 6.48000344e+00, -6.86744478e-01],
 [ 1.80519415e+00,  1.02355067e+00],
 [ 6.12186233e-02,  4.01084407e+00],
 [ 1.19353157e+00,  1.53220655e+00],
 [ 9.60430308e+00, -9.82047033e-01],
 [ 7.46773893e+00, -7.75273113e-01],
 [ 9.16473487e+00, -9.64777931e-01],
 [ 1.87266314e+00,  9.74713660e-01],
 [ 2.10853274e+00,  8.19171557e-01],
 [ 5.25907724e+00, -4.38116829e-01],
 [ 1.64228393e+00,  1.20371444e+00],
 [ 3.11108552e-02,  4.20576120e+00],
 [ 8.73802322e+00, -9.33020059e-01],
 [ 4.89067441e+00, -3.42826582e-01],
 [ 6.29223941e+00, -6.57479227e-01],
 [ 5.07324313e-01,  2.45043084e+00],
 [ 6.20326434e-01,  2.25046698e+00],
 [ 1.64882693e+00,  1.20035917e+00],
 [ 8.83578478e+00, -9.42567337e-01],
 [ 9.36590805e+00, -9.76053812e-01]]
        pareto_front = sorted(pareto_front, key=lambda row: row[0])
        return np.array(pareto_front)
    





#___________________________________________________________________________-
# Define the lists of problems


my_problems = [
    MyProblem3(),
    MyProblem2(),
    MyProblem4(),
    MyProblem1(),
    MyCircleProblem(),


    
    
]
pymoo_problems = [
    "g1",
    "g2",
    "g3",
    "g4",
    "zdt1",             # ZDT1 Problem (2 objectives)
#    "zdt2",             # ZDT2 Problem (2 objectives)
#    "zdt3",             # ZDT3 Problem (2 objectives)
#    "zdt4",             # ZDT4 Problem (2 objectives)
#    "zdt6",             # ZDT6 Problem (2 objectives)
#    "dtlz1",            # DTLZ1 Problem (Scalable objectives)
#    "dtlz2",            # DTLZ2 Problem (Scalable objectives)
#    "dtlz3",            # DTLZ3 Problem (Scalable objectives)
#    "dtlz4",            # DTLZ4 Problem (Scalable objectives)
#    "dtlz5",            # DTLZ5 Problem (Scalable objectives)
#    "dtlz6",            # DTLZ6 Problem (Scalable objectives)
#    "dtlz7",            # DTLZ7 Problem (Scalable objectives)
#    "wfg3",             # WFG3 Problem (Scalable objectives)
#    "wfg4",             # WFG4 Problem (Scalable objectives)
#    "wfg5",             # WFG5 Problem (Scalable objectives)
#    "wfg6",             # WFG6 Problem (Scalable objectives)
#    "wfg7",             # WFG7 Problem (Scalable objectives)
#    "wfg8",             # WFG8 Problem (Scalable objectives)
#    "wfg9",             # WFG9 Problem (Scalable objectives)
#    "tnk",              # TNK Problem (2 objectives, constrained)
#    "bnh",              # Binh and Korn Problem (2 objectives, constrained)
#    "srn",              # Srinivas Problem (2 objectives, constrained)
#    "c1dtlz1",          # Constrained DTLZ1 Problem
#    "c2dtlz2",          # Constrained DTLZ2 Problem
#    "kursawe",          # Kursawe Problem (2 objectives, non-convex Pareto front)
#    "viennet2",         # Viennet Problem (3 objectives)
#    "pressure_vessel",  # Pressure Vessel Design Problem (constrained)
#    "welded_beam"       # Welded Beam Design Problem (constrained)
]

