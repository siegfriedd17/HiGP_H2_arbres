"""
Prediction par processus gaussien avec HiGP_H2_arbres : la matrice de covariance des
points d'entrainement est une H2 matrice construite sur un arbre PCA ou 2-means, en
dimension quelconque. Le resultat est compare a celui du calcul dense de HiGP.

Lancer (depuis le dossier "Hmat dossier code") :
    hmat_env/bin/python HiGP_H2_arbres/exemple_h2_arbres.py

Le module doit avoir ete compile :
    cd HiGP_H2_arbres/py-interface
    "../../hmat_env/bin/python" setup.py build_ext --inplace --force
"""
import os
import sys
import time

# Nombre de threads : a fixer AVANT de charger numpy / torch / higp. Sur cette machine 
# (20 coeurs, OpenBLAS a threads), laisser la valeur par defaut de 20 threads rend le 
# calcul jusqu'a 100 fois plus lent (trop de threads en concurrence). 10 est un bon reglage ;
# pour en changer :  OMP_NUM_THREADS=8 hmat_env/bin/python ...
os.environ.setdefault("OMP_NUM_THREADS", "10")

import numpy as np

# Utiliser cette bibliotheque, et non le HiGP installe dans l'environnement
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "py-interface"))
import higp  # noqa: E402

# ============================================================
# PARAMETRES
# ============================================================
DIM = 20                        # dimension des points
N_TRAIN = 8000                  # nombre de points d'entrainement
N_TEST = 100                    # nombre de points de prediction
ARBRES = ("pca", "2means")      # arbres a essayer : "pca", "2means", "box" (box : DIM <= 3)
ALPHA = 1.5                     # admissibilite : dist(c_i, c_j) >= ALPHA * (r_i + r_j)
NOYAU = higp.Matern32Kernel     # higp.GaussianKernel, higp.Matern32Kernel, higp.Matern52Kernel
PARAMS = (0.5, 0.5, -2.0)       # hyperparametres [l, f, s] AVANT transformation softplus
TOL_PCG = 1e-6                  # tolerance du gradient conjugue
VERBEUX = 1                     # 1 : afficher l'arbre, les sample points et les rangs
GRAINE = 0


def generer(n, dim, rng, plongement):
    """Points de dimension intrinseque 3 plonges dans R^dim, tableau (dim, n)."""
    return plongement @ rng.uniform(0.0, 1.0, size=(3, n)) + 1e-3 * rng.normal(size=(dim, n))


def main():
    rng = np.random.default_rng(GRAINE)
    plongement = rng.uniform(-0.5, 0.5, size=(DIM, 3))
    f = lambda X: np.sin(3.0 * X[0]) + np.cos(2.0 * X[1 % DIM]) * X[DIM - 1]
    x_train = np.ascontiguousarray(generer(N_TRAIN, DIM, rng, plongement))
    x_test = np.ascontiguousarray(generer(N_TEST, DIM, rng, plongement))
    y_train = np.ascontiguousarray(f(x_train) + 0.05 * rng.normal(size=N_TRAIN))
    y_test = f(x_test)

    arguments = dict(data_train=x_train, label_train=y_train, data_prediction=x_test,
                     kernel_type=NOYAU, gp_params=np.array(PARAMS), niter=500, tol=TOL_PCG)

    # Reference : matrice de covariance dense
    t0 = time.time()
    ref = higp.gpr_prediction(mvtype=higp.MatvecAOT, **arguments)
    print(f"\n=== dense : {time.time() - t0:.1f} s, RMSE contre la vraie fonction = "
          f"{np.sqrt(np.mean((ref.prediction_mean - y_test) ** 2)):.4f}\n")

    for arbre in ARBRES:
        # min_size=0 : utiliser la H2 matrice quel que soit le nombre de points
        # (par defaut elle n'est utilisee qu'a partir de 20 000 points)
        higp.set_h2_options(tree=arbre, adm_alpha=ALPHA, min_size=0, verbose=VERBEUX)
        t0 = time.time()
        pred = higp.gpr_prediction(mvtype=higp.MatvecAuto, **arguments)
        ecart_moy = np.linalg.norm(pred.prediction_mean - ref.prediction_mean) / np.linalg.norm(ref.prediction_mean)
        ecart_std = np.linalg.norm(pred.prediction_stddev - ref.prediction_stddev) / np.linalg.norm(ref.prediction_stddev)
        print(f"\n=== H2, arbre {arbre} : {time.time() - t0:.1f} s, ecart relatif au dense : "
              f"moyenne {ecart_moy:.1e}, ecart-type {ecart_std:.1e}\n")

    print("Options H2 courantes :", higp.get_h2_options())


if __name__ == "__main__":
    main()
