from pymoo.core.problem import ElementwiseProblem
from reproblems.reproblem_python_ver.reproblem import RE23
import numpy as np

class RE23_pymoo(ElementwiseProblem):
    """Pymoo wrapper for the RE23 problem from the Reproblems suite."""
    def __init__(self, re23_instance):
        self.re = re23_instance
        super().__init__(
            n_var = self.re.n_variables,    # Number of decision variables
            n_obj = self.re.n_objectives,   # Number of objectives
            n_ieq_constr = 0,               # Number of inequality constraints
            xl = self.re.lbound,            # Lower bounds
            xu = self.re.ubound             # Upper bounds   
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f = self.re.evaluate(x)             # Evaluate objectives
        out["F"] = f

def initialize_re23():
    re23_instance = RE23()
    problem = RE23_pymoo(re23_instance)
    return problem