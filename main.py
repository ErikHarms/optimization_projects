import numpy as np
import os
import json
from datetime import datetime
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter
from pymoo.core.callback import Callback
import inspect

from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.core.variable import Real


from re21 import initialize_re21

class ProgressCallback(Callback):
    def __init__(self, interval=10):
        super().__init__()
        self.interval = interval
        self.history = []

    def notify(self, algorithm):
        gen = algorithm.n_gen
        pop_F = algorithm.pop.get("F")
        
        # Automatisch alle Objectives behandeln
        best_F = np.min(pop_F, axis=0)  # Minimum jeder Objective-Spalte
        entry = {"generation": gen}
        for i, val in enumerate(best_F):
            entry[f"best_f{i+1}"] = float(val)
        self.history.append(entry)

        if gen % self.interval == 0 or gen == 1:
            best_str = ", ".join([f"f{i+1}={best_F[i]:10.4f}" for i in range(best_F.shape[0])])
            print(f"Generation {gen:4d}: {best_str}")

def create_solution_folder(base_path="problem_solution", project_name="project"):
    # Projektordner
    project_path = os.path.join(base_path, project_name)
    os.makedirs(project_path, exist_ok=True)

    # Finde nächste freie Nummer
    existing = [d for d in os.listdir(project_path) if d.isdigit()]
    next_num = 1
    if existing:
        next_num = max(int(d) for d in existing) + 1

    solution_path = os.path.join(project_path, f"{next_num:03d}")
    os.makedirs(solution_path)
    return solution_path

def save_log(res, algorithm, algorithm_params, callback, save_dir, seed):
    """
    Speichert die Ergebnisse eines pymoo-Laufs in einer JSON-Datei.
    
    Parameter:
        res: Result-Objekt von pymoo
        algorithm: eingesetzter Algorithmus (z.B. NSGA2)
        callback: Callback-Objekt mit Progress-Historie
        save_dir: Pfad, in dem die Log-Datei gespeichert werden soll
        seed: verwendeter Random Seed
    """
    
    # Dynamisch Anzahl der Objectives und Variablen
    n_obj = res.F.shape[1] if hasattr(res, "F") else None
    n_var = res.X.shape[1] if hasattr(res, "X") else None

    # Log-Daten zusammenstellen
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": type(algorithm).__name__,
        "algorithm_parameters": algorithm_params,
        "n_objectives": n_obj,
        "n_variables": n_var,
        "generations": getattr(res, "n_gen", None),
        "seed": seed,
        "pareto_solutions": int(res.F.shape[0]) if hasattr(res, "F") else None,
        "F_first_10": res.F[:10].tolist() if hasattr(res, "F") else None,
        "X_first_10": res.X[:10].tolist() if hasattr(res, "X") else None,
        "progress": callback.history if hasattr(callback, "history") else None
    }

    # Datei speichern
    log_path = os.path.join(save_dir, "log.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)
    
    print("Log Datei gespeichert:", log_path)
    return log_path

def obj_to_str(v):
        # Primitive Typen
        if isinstance(v, (bool, int, float)):
            return str(v)
        elif isinstance(v, str):
            return f"'{v}'"
        
        # Real → nur value
        elif isinstance(v, Real):
            return str(v.value)
        
        # Funktionen → nur Name
        elif callable(v) and hasattr(v, "__name__"):
            return v.__name__
        
        else:
            cls = v.__class__.__name__
            attrs = {}
            for k in ['eta', 'prob', 'func_comp']:
                if hasattr(v, k):
                    val = getattr(v, k)
                    if isinstance(val, Real):
                        val = val.value
                    if callable(val) and hasattr(val, "__name__"):
                        val = val.__name__
                    attrs[k] = val
            if attrs:
                attr_str = ", ".join(f"{k}={obj_to_str(val)}" for k, val in attrs.items())
                return f"{cls}({attr_str})"
            else:
                return f"{cls}()"

def main():
    # Set random seed for reproducibility
    seed = 1
    np.random.seed(seed)

    problem_name = "re21"
    save_dir = create_solution_folder("problem_solutions", problem_name)

    init_func_name = f"initialize_{problem_name}"
    problem = globals()[init_func_name]()

    # Configure NSGA-II algorithm
    params = {
        'pop_size': 100,                                                # Number of individuals per generation → coverage & speed
        'eliminate_duplicates': True,                                   # Removes identical solutions → prevents stagnation
        'sampling': FloatRandomSampling(),                              # Initial population → affects starting distribution
        'selection': TournamentSelection(func_comp=binary_tournament),  # Parent selection → determines selection pressure
        'crossover': SBX(eta=15, prob=0.9),                             # Combines parents → affects exploration vs. exploitation
        'mutation': PM(eta=20, prob=0.1),                               # Modifies individuals → exploration & diversity
        'survival': RankAndCrowding()                                   # Survival selection based on rank and crowding
    }
    algorithm = NSGA2(**params)
    
    params_str = {k: obj_to_str(v) for k, v in params.items()}
    print(params_str)
    
    # Termination criterion
    termination = get_termination("n_gen", 100)
    
    # Progress Callback to get feedback during optimization
    callback = ProgressCallback(interval=10)

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=seed,
                   callback=callback,
                   save_history=True,
                   verbose=False)

    # Save Scatter Plot
    scatter = Scatter(title="RE21 - NSGA-II (approximierte Pareto-Front)")
    scatter.add(res.F)
    scatter_path = os.path.join(save_dir, "scatter.png")
    scatter.save(scatter_path)

    log_path = save_log(res, algorithm, params_str, callback, save_dir, seed)

    print(f"\nErgebnisse gespeichert in: {save_dir}")
    print("Scatter Plot:", scatter_path)
    print("Log Datei:", log_path)

if __name__ == "__main__":
    main()