import numpy as np
from pymoo.core.problem import ElementwiseProblem

class TRE01():
    """
    Beispielproblem T01: Kalibrierung einer gekoppelten Thermo-Fluid-Simulation
    Ziel: Optimierung von Simulationsparametern anhand realer Messpunkte.
    """

    def __init__(self):
        self.problem_name = 'T01'
        self.n_objectives = 2   # thermodynamisch, fluiddynamisch
        self.n_variables = 4    # z.B. Parameter der Simulation
        self.n_constraints = 0
        self.n_original_constraints = 0

        # Parametergrenzen: [Wärmeleitfähigkeit, Reibungskoeff., Dichtefaktor, Wärmetauscher-Wirkung]
        self.lbound = np.array([0.1, 0.001, 0.8, 0.5])
        self.ubound = np.array([2.0, 0.02, 1.2, 1.5])

        # Fiktive reale Messwerte (z. B. Temperatur- und Druckprofile)
        self.real_temp_profile = np.array([350, 355, 360, 362, 365])   # [K]
        self.real_press_profile = np.array([2.0, 2.3, 2.5, 2.6, 2.8])  # [bar]

    def simulate_thermal(self, params):
        """Einfache thermodynamische Modellgleichung."""
        k, f, rho, eta = params
        # simulierte Temperaturverteilung
        temp_sim = 340 + 10*k - 5*f + np.linspace(-1, 2, 5)*eta
        return temp_sim

    def simulate_fluid(self, params):
        """Einfache fluiddynamische Modellgleichung."""
        k, f, rho, eta = params
        # simulierte Druckverteilung
        press_sim = 2.0 + 0.4*rho - 10*f + np.linspace(0, 0.5, 5)*k
        return press_sim

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)

        # Simulationsergebnisse
        temp_sim = self.simulate_thermal(x)
        press_sim = self.simulate_fluid(x)

        # Ziel 1: thermodynamischer Fehler (RMSE zwischen Simulation und Messung)
        f[0] = np.sqrt(np.mean((temp_sim - self.real_temp_profile)**2))

        # Ziel 2: fluiddynamischer Fehler (RMSE)
        f[1] = np.sqrt(np.mean((press_sim - self.real_press_profile)**2))

        return f

class TRE01_pymoo(ElementwiseProblem):
    """Pymoo wrapper for the T01 problem."""
    def __init__(self, t01_instance):
        self.t01 = t01_instance
        super().__init__(
            n_var = self.t01.n_variables,    # Number of decision variables
            n_obj = self.t01.n_objectives,   # Number of objectives
            n_ieq_constr = self.t01.n_constraints, # Number of inequality constraints
            xl = self.t01.lbound,            # Lower bounds
            xu = self.t01.ubound             # Upper bounds   
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f = self.t01.evaluate(x)           # Evaluate objectives
        out["F"] = f                       # Safe objectives in pymoo format

def initialize_tre01():
    t01_instance = TRE01()
    problem = TRE01_pymoo(t01_instance)
    return problem