import numpy as np
import os
import json
from datetime import datetime
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter
from pymoo.core.callback import Callback
import matplotlib.pyplot as plt

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.core.variable import Real
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import Hypervolume


from re21 import initialize_re21
from re23 import initialize_re23
from cre25 import initialize_cre25
from cre51 import initialize_cre51

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

def save_log(res, algorithm, algorithm_params, callback, save_dir, seed, termination=None, ref_point=None):
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

    termination_info = None
    if termination is not None:
        termination_info = {}
        # Prüfe, ob n_gen gesetzt ist
        if hasattr(termination, "n_max_gen"):
            termination_info["n_gen"] = termination.n_max_gen
        # Falls andere Typen, hier erweitern
        # z.B. tol, time_limit, etc.
        # termination_info["tol"] = getattr(termination, "tol", None)
    
    # Hypervolume berechnen, falls Referenzpunkt gegeben
    hv_value = None
    if ref_point is not None and hasattr(res, "F") and res.F.size > 0:
        hv = Hypervolume(ref_point=np.array(ref_point))
        hv_value = float(hv.calc(res.F))

    # Log-Daten zusammenstellen
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": type(algorithm).__name__,
        "algorithm_parameters": algorithm_params,
        "termination": termination_info,
        "n_objectives": n_obj,
        "n_variables": n_var,
        "generations": getattr(res, "n_gen", None),
        "seed": seed,
        "pareto_solutions": int(res.F.shape[0]) if hasattr(res, "F") else None,
        "F_first_10": res.F[:10].tolist() if hasattr(res, "F") else None,
        "X_first_10": res.X[:10].tolist() if hasattr(res, "X") else None,
        "hypervolume": hv_value,
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

def plot_progress(callback, save_dir):
    """
    Plottet die Entwicklung der Objectives über die Generationen
    mit logarithmischer Skala, um kleine Differenzen sichtbar zu machen.
    """
    history = callback.history
    generations = [entry["generation"] for entry in history]

    # Alle Objectives extrahieren
    n_obj = len([k for k in history[0] if k.startswith("best_f")])
    all_values = np.array([[entry[f"best_f{i+1}"] for i in range(n_obj)] for entry in history])

    # Differenzen berechnen
    diffs = np.diff(all_values, axis=0)

    # Absolutwerte für Log-Skala
    diffs_abs = np.abs(diffs)
    diffs_abs[diffs_abs == 0] = 1e-6  # kleine Werte vermeiden log(0)

    # Mittelwert der Differenzen pro Generation
    mean_diffs = np.mean(diffs_abs, axis=1)

    # Plot für jede Funktion
    plt.figure(figsize=(10, 6))
    for i in range(n_obj):
        plt.plot(generations[1:], diffs_abs[:, i], label=f"Δf{i+1}")
    plt.plot(generations[1:], mean_diffs, label="Mittelwert Δf", color="black", linewidth=2, linestyle="--")

    plt.xlabel("Generation")
    plt.ylabel("Absolute Differenz der Objectives")
    plt.yscale("log")
    plt.yticks([1e-3, 1e-2, 1e-1], ["0.001", "0.01", "0.1"])
    plt.title("Fortschritt der Objectives über die Generationen (log-Skala)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plot_path = os.path.join(save_dir, "progress.png")
    plt.savefig(plot_path)
    plt.close()
    print("Progress Plot gespeichert:", plot_path)
    return plot_path

def save_adaptive_scatter_plot(res, save_dir):
    n_obj = res.F.shape[1]
    scatter_path = os.path.join(save_dir, "scatter.png")
    width = min(6 + n_obj * 3, 20)
    height = min(5 + n_obj * 2, 15)

    # Pareto-Front bestimmen (erste Front)
    fronts = NonDominatedSorting().do(res.F, return_rank=True)
    ranks = fronts[1]  # Rückgabe: (fronts, ranks)
    pareto_indices = np.where(ranks == 0)[0]  # Front 0 = Pareto-Front

    # Ausgewählte Punkte: nur die ersten 10
    selected_indices = pareto_indices[:10]
    remaining_indices = np.setdiff1d(np.arange(res.F.shape[0]), selected_indices)

    if n_obj <= 3:
        from pymoo.visualization.scatter import Scatter
        scatter = Scatter(figsize=(width, height))

        # Alle Punkte hinzufügen
        scatter.add(res.F[remaining_indices], s=20, color="blue")  # Alle anderen Punkte
        scatter.add(res.F[selected_indices], s=50, color="red", marker="o", edgecolor="black", label="Top 10 Pareto")
        scatter.do()

        # Nummerierung der Top 10
        for i, idx in enumerate(selected_indices):
            point = res.F[idx]
            if n_obj == 2:
                scatter.ax.text(point[0], point[1], str(i + 1), color="black", fontsize=10, fontweight="bold")
            elif n_obj == 3:
                scatter.ax.text(point[0], point[1], point[2], str(i + 1), color="black", fontsize=10, fontweight="bold")

        # Achsenbeschriftungen
        scatter.ax.set_xlabel("F 1")
        scatter.ax.set_ylabel("F 2")
        if n_obj == 3:
            scatter.ax.set_zlabel("F 3")
        plt.tight_layout()
        scatter.fig.savefig(scatter_path, dpi=300, bbox_inches="tight")

    else:
        # Scatter-Matrix für n_obj > 3
        fig, axes = plt.subplots(n_obj, n_obj, figsize=(width, height))
        for i in range(n_obj):
            for j in range(n_obj):
                ax = axes[i, j]
                if i != j:
                    # Alle Punkte plotten
                    ax.scatter(res.F[remaining_indices, j], res.F[remaining_indices, i], s=20, color="blue")
                    ax.scatter(res.F[selected_indices, j], res.F[selected_indices, i], s=50, color="red", edgecolor="black")
                    # Top-10 nummerieren
                    for k, idx in enumerate(selected_indices):
                        ax.text(res.F[idx, j], res.F[idx, i], str(k + 1), color="black", fontsize=8, fontweight="bold")
                    ax.set_xlabel(f"F {j+1}")
                    ax.set_ylabel(f"F {i+1}")
                else:
                    ax.axis("off")  # Diagonalplots ausblenden

        plt.tight_layout()
        fig.savefig(scatter_path, dpi=300, bbox_inches="tight")

    print(f"Scatter Plot gespeichert unter: {scatter_path}")
    return scatter_path

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
    termination = get_termination("n_gen", 200)
    
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
    save_adaptive_scatter_plot(res, save_dir)

    log_path = save_log(res, algorithm, params_str, callback, save_dir, seed, termination)

    progress_plot_path = plot_progress(callback, save_dir)

    print(f"\nErgebnisse gespeichert in: {save_dir}")
    print("Konvergenz Plot:", progress_plot_path)
    print("Log Datei:", log_path)

if __name__ == "__main__":
    main()