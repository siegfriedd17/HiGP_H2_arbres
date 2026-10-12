"""
Regression par processus gaussien sur un jeu de donnees reel avec HiGP_H2_arbres :
entrainement des hyperparametres puis prediction, avec la matrice de covariance
    - dense (reference),
    - H2 sur arbre PCA,
    - H2 sur arbre 2-means.
Pour chaque mode : RMSE sur les points de test, temps d'entrainement, temps de prediction.

Le protocole par defaut est celui du benchmark publie par HiGP contre GPyTorch
(HiGP/py-demos/uci_benchmark/configs/test10_bike.json) : Bike Sharing, 3000 points
d'entrainement, 200 de test, noyau gaussien, simple precision, Adam (lr 0.01, 500
iterations), donnees centrees-reduites avec les statistiques d'entrainement.

Lancer (depuis le dossier "Hmat dossier code") :
    hmat_env/bin/python HiGP_H2_arbres/benchmark_dataset.py                       # Bike Sharing
    hmat_env/bin/python HiGP_H2_arbres/benchmark_dataset.py --donnees road3d --n-train 20000
    hmat_env/bin/python HiGP_H2_arbres/benchmark_dataset.py --donnees mon_fichier.csv
    hmat_env/bin/python HiGP_H2_arbres/benchmark_dataset.py --aide

Fichier CSV : une ligne par point, colonnes numeriques, la DERNIERE colonne est la
valeur a predire. Une premiere ligne d'en-tete est acceptee.
"""
import argparse
import contextlib
import ctypes
import gc
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

# Nombre de threads : a fixer AVANT de charger numpy / torch / higp. Sur cette machine 
# (20 coeurs, OpenBLAS a threads), laisser la valeur par defaut de 20 threads rend le 
# calcul jusqu'a 100 fois plus lent (trop de threads en concurrence). 10 est un bon reglage ;
# pour en changer :  OMP_NUM_THREADS=8 hmat_env/bin/python ...
os.environ.setdefault("OMP_NUM_THREADS", "10")

import numpy as np
import torch

# Utiliser cette bibliotheque, et non le HiGP installe dans l'environnement
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "py-interface"))
import higp  # noqa: E402

CACHE = Path("~/.higp_cache/datasets").expanduser()     # meme dossier que le benchmark de HiGP
URL_BIKE = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
URL_ROAD3D = "https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt"
NOYAUX = {"gaussien": higp.GaussianKernel, "matern32": higp.Matern32Kernel, "matern52": higp.Matern52Kernel}


# ============================================================
# Donnees
# ============================================================
def charger_bike(sans_fuite):
    """UCI Bike Sharing, 17 379 releves horaires. Meme preparation que le benchmark de HiGP."""
    import pandas as pd
    dossier = CACHE / "bike"
    if not (dossier / "hour.csv").exists():
        dossier.mkdir(parents=True, exist_ok=True)
        print("Telechargement de Bike Sharing...", flush=True)
        urllib.request.urlretrieve(URL_BIKE, dossier / "bike.zip")
        zipfile.ZipFile(dossier / "bike.zip").extractall(dossier)
        (dossier / "bike.zip").unlink()
    df = pd.read_csv(dossier / "hour.csv")
    df["dteday"] = pd.to_datetime(df["dteday"]).dt.dayofyear
    df = df.drop(columns=["instant"])
    if sans_fuite:
        # casual + registered == cnt (la cible) sur toutes les lignes
        df = df.drop(columns=["casual", "registered"])
    data = df.values.astype(np.float64)
    return data[:, :-1], data[:, -1], "Bike Sharing (UCI)" + (", sans casual/registered" if sans_fuite else "")


def charger_road3d():
    """UCI 3D Road Network, 434 874 points : (longitude, latitude) -> altitude."""
    import pandas as pd
    dossier = CACHE / "road3d"
    fichier = dossier / "3D_spatial_network.txt"
    if not fichier.exists():
        dossier.mkdir(parents=True, exist_ok=True)
        print("Telechargement de 3D Road Network...", flush=True)
        urllib.request.urlretrieve(URL_ROAD3D, fichier)
    data = pd.read_csv(fichier, sep=",", header=None).values
    return data[:, 1:3].astype(np.float64), data[:, 3].astype(np.float64), "3D Road Network (UCI)"


def charger_csv(chemin):
    """Fichier CSV numerique, derniere colonne = cible."""
    import pandas as pd
    df = pd.read_csv(chemin, sep=None, engine="python")
    try:
        float(df.columns[0])        # premiere ligne numerique : ce n'etait pas un en-tete
        df = pd.read_csv(chemin, sep=None, engine="python", header=None)
    except ValueError:
        pass
    df = df.select_dtypes(include=[np.number]).dropna()
    data = df.values.astype(np.float64)
    return data[:, :-1], data[:, -1], os.path.basename(chemin)


def preparer(X, y, n_train, n_test, graine, dtype):
    """Tirage aleatoire train / test, puis centrage-reduction avec les statistiques d'entrainement."""
    n = X.shape[0]
    if n_train + n_test > n:
        raise SystemExit(f"Le jeu de donnees n'a que {n} points : n_train + n_test = {n_train + n_test} est trop grand.")
    perm = np.random.default_rng(graine).permutation(n)
    i_tr, i_te = perm[:n_train], perm[n_train:n_train + n_test]
    mx, sx = X[i_tr].mean(0, keepdims=True), X[i_tr].std(0, keepdims=True) + 1e-6
    my, sy = y[i_tr].mean(), y[i_tr].std() + 1e-6
    # HiGP attend des tableaux (dimension, nombre de points) contigus
    x_tr = np.ascontiguousarray(((X[i_tr] - mx) / sx).T.astype(dtype))
    x_te = np.ascontiguousarray(((X[i_te] - mx) / sx).T.astype(dtype))
    y_tr = np.ascontiguousarray(((y[i_tr] - my) / sy).astype(dtype))
    y_te = (y[i_te] - my) / sy
    return x_tr, y_tr, x_te, y_te, sy


# ============================================================
# Calcul
# ============================================================
@contextlib.contextmanager
def silence(actif=True):
    """Masque ce que HiGP ecrit depuis le C++ (stdout et stderr du processus)."""
    if not actif:
        yield
        return
    # Le C++ a ses propres tampons d'ecriture : il faut les vider avant chaque bascule
    vider_c = ctypes.CDLL(None).fflush
    sys.stdout.flush(); sys.stderr.flush(); vider_c(None)
    sauve = os.dup(1), os.dup(2)
    nul = os.open(os.devnull, os.O_WRONLY)
    os.dup2(nul, 1); os.dup2(nul, 2)
    try:
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush(); vider_c(None)
        os.dup2(sauve[0], 1); os.dup2(sauve[1], 2)
        for fd in (nul,) + sauve:
            os.close(fd)


def lancer(mode, x_tr, y_tr, x_te, y_te, a, torch_dtype):
    """Entrainement + prediction pour un mode : 'dense', 'pca', '2means' ou 'box'."""
    if mode == "dense":
        mvtype = higp.MatvecAOT
    else:
        # min_size=0 : H2 quel que soit le nombre de points
        higp.set_h2_options(tree=mode, adm_alpha=a.alpha, min_size=0, h2_tol=a.h2_tol, verbose=0)
        mvtype = higp.MatvecAuto
    torch.manual_seed(a.graine); np.random.seed(a.graine)
    noyau = NOYAUX[a.noyau]
    with silence(not a.bavard):
        probleme = higp.gprproblem.setup(data=x_tr, label=y_tr, kernel_type=noyau, mvtype=mvtype,
                                         afn_rank=a.rang_train, afn_lfil=0, niter=a.cg_train, nvec=10, seed=a.graine)
        modele = higp.GPRModel(probleme, dtype=torch_dtype)
        optimiseur = torch.optim.Adam(modele.parameters(), lr=a.lr)
        t0 = time.perf_counter()
        if a.iterations > 0:
            higp.gpr_torch_minimize(modele, optimiseur, maxits=a.iterations, print_info=False)
        t_train = time.perf_counter() - t0
        params = modele.get_params()
        t0 = time.perf_counter()
        pred = higp.gpr_prediction(data_train=x_tr, label_train=y_tr, data_prediction=x_te, kernel_type=noyau,
                                   gp_params=params, mvtype=mvtype, niter=a.cg_pred, tol=a.tol_pred,
                                   afn_rank=a.rang_pred, afn_lfil=0)
        t_pred = time.perf_counter() - t0
        moyenne = np.array(pred.prediction_mean, dtype=np.float64)
        del probleme, modele, optimiseur
        gc.collect()   # les objets HiGP ecrivent un message quand ils sont liberes
    rmse = float(np.sqrt(np.mean((moyenne - y_te) ** 2)))
    return dict(mode=mode, rmse=rmse, t_train=t_train, t_pred=t_pred, params=np.array(params, dtype=np.float64), moyenne=moyenne)


def main():
    p = argparse.ArgumentParser(add_help=False, description="RMSE et temps de calcul d'un GP : dense, H2-PCA, H2-2means")
    p.add_argument("--aide", "-h", action="help", help="afficher cette aide")
    p.add_argument("--donnees", default="bike", help="bike (defaut), road3d, ou chemin d'un fichier CSV (derniere colonne = cible)")
    p.add_argument("--sans-fuite", action="store_true", help="bike : retirer casual et registered, dont la somme est la cible")
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--modes", default="dense,pca,2means", help="liste parmi dense, pca, 2means, box")
    p.add_argument("--noyau", default="gaussien", choices=sorted(NOYAUX))
    p.add_argument("--precision", default="simple", choices=["simple", "double"])
    p.add_argument("--iterations", type=int, default=500, help="iterations d'Adam (0 : pas d'entrainement)")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=1.5, help="coefficient d'admissibilite des arbres PCA / 2-means")
    p.add_argument("--h2-tol", type=float, default=0.0, help="tolerance H2, 0 = valeur de HiGP (1e-5 simple, 1e-10 double)")
    p.add_argument("--cg-train", type=int, default=20)
    p.add_argument("--cg-pred", type=int, default=50)
    p.add_argument("--tol-pred", type=float, default=1e-2)
    p.add_argument("--rang-train", type=int, default=10)
    p.add_argument("--rang-pred", type=int, default=100)
    p.add_argument("--graine", type=int, default=42)
    p.add_argument("--bavard", action="store_true", help="laisser HiGP afficher ses messages")
    a = p.parse_args()

    if a.donnees == "bike":
        X, y, nom = charger_bike(a.sans_fuite)
    elif a.donnees == "road3d":
        X, y, nom = charger_road3d()
    else:
        X, y, nom = charger_csv(a.donnees)
    np_dtype, torch_dtype = (np.float32, torch.float32) if a.precision == "simple" else (np.float64, torch.float64)
    torch.set_default_dtype(torch_dtype)
    x_tr, y_tr, x_te, y_te, sy = preparer(X, y, a.n_train, a.n_test, a.graine, np_dtype)
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]

    print(f"\nDonnees      : {nom}, {X.shape[0]} points, dimension {X.shape[1]}")
    print(f"Decoupage    : {a.n_train} entrainement, {a.n_test} test (graine {a.graine}), donnees centrees-reduites")
    print(f"Modele       : noyau {a.noyau}, precision {a.precision}, Adam lr {a.lr}, {a.iterations} iterations")
    print(f"Solveur      : CG {a.cg_train} it. (entrainement) / {a.cg_pred} it. (prediction), AFN rang {a.rang_train} / {a.rang_pred}")
    print(f"H2           : alpha = {a.alpha}, tolerance = {a.h2_tol if a.h2_tol > 0 else 'defaut HiGP'}\n", flush=True)

    resultats = []
    for mode in modes:
        print(f"  calcul en cours : {mode} ...", end="", flush=True)
        r = lancer(mode, x_tr, y_tr, x_te, y_te, a, torch_dtype)
        print(f" termine en {r['t_train'] + r['t_pred']:.1f} s", flush=True)
        resultats.append(r)

    ref = next((r for r in resultats if r["mode"] == "dense"), None)
    noms = {"dense": "dense (reference)", "pca": "H2, arbre PCA", "2means": "H2, arbre 2-means", "box": "H2, arbre en boites"}
    ligne = "+" + "-" * 21 + "+" + "-" * 12 + "+" + "-" * 16 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 16 + "+"
    print("\n" + ligne)
    print(f"| {'Matrice K11':<19} | {'RMSE':>10} | {'RMSE (unites':>14} | {'Entrainement':>12} | {'Prediction':>12} | {'Ecart au dense':>14} |")
    print(f"| {'':<19} | {'(norm.)':>10} | {'de la cible)':>14} | {'(s)':>12} | {'(s)':>12} | {'(relatif)':>14} |")
    print(ligne)
    for r in resultats:
        if ref is None or r is ref:
            ecart = "-"
        else:
            ecart = "%.1e" % (np.linalg.norm(r["moyenne"] - ref["moyenne"]) / np.linalg.norm(ref["moyenne"]))
        print(f"| {noms.get(r['mode'], r['mode']):<19} | {r['rmse']:>10.4f} | {r['rmse'] * sy:>14.4g} | {r['t_train']:>12.1f} | {r['t_pred']:>12.1f} | {ecart:>14} |")
    print(ligne)
    print("RMSE : racine de l'erreur quadratique moyenne sur les points de test. 'norm.' : cible centree-reduite (ecart-type 1).")
    print("Ecart au dense : difference relative entre la moyenne predite et celle du mode dense.")
    print("\nHyperparametres appris [l, f, s] (avant transformation softplus) :")
    for r in resultats:
        print(f"  {noms.get(r['mode'], r['mode']):<19} : " + "  ".join("%8.4f" % v for v in r["params"]))
    print()


if __name__ == "__main__":
    main()
