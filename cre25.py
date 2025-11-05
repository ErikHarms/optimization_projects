from pymoo.core.problem import ElementwiseProblem
from reproblems.reproblem_python_ver.reproblem import CRE25

class CRE25_pymoo(ElementwiseProblem):
    """Pymoo wrapper for the CRE25 problem from the Reproblems suite."""
    def __init__(self, cre25_instance):
        self.cre = cre25_instance
        super().__init__(
            n_var = self.cre.n_variables,    # Number of decision variables
            n_obj = self.cre.n_objectives,   # Number of objectives
            n_ieq_constr = self.cre.n_constraints, # Number of inequality constraints
            xl = self.cre.lbound,            # Lower bounds
            xu = self.cre.ubound             # Upper bounds   
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f, g = self.cre.evaluate(x)         # Evaluate objectives and constraints
        out["F"] = f                        # Safe objectives in pymoo format
        out["G"] = g                        # Safe constraints in pymoo format

def initialize_cre25():
    cre25_instance = CRE25()
    problem = CRE25_pymoo(cre25_instance)
    return problem