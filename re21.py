from pymoo.core.problem import ElementwiseProblem
from reproblems.reproblem_python_ver.reproblem import RE21

class RE21_pymoo(ElementwiseProblem):
    def __init__(self, re21_instance):
        self.re = re21_instance
        super().__init__(
            n_var = self.re.n_variables,    # Number of decision variables
            n_obj = self.re.n_objectives,   # Number of objectives
            n_ieq_constr = 0,               # Number of inequality constraints
            xl = self.re.lbound,            # Lower bounds
            xu = self.re.ubound             # Upper bounds   
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f = self.re.evaluate(x)             # Evaluate objectives
        out["F"] = f                        # Safe objectives in pymoo format

def initialize_re21():
    re21_instance = RE21()
    problem = RE21_pymoo(re21_instance)
    return problem