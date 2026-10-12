# HiGP_H2_arbres : tutoriel rapide

HiGP_H2_arbres est une version modifiée de [HiGP](https://github.com/huanghua1994/HiGP),
une bibliothèque de processus gaussiens (régression, classification, entraînement des
hyperparamètres). La différence porte sur un seul point : la matrice de covariance peut
être compressée en matrice H² **en dimension quelconque**, grâce à deux arbres de
partition binaires (PCA et 2-means). Dans HiGP d'origine, la matrice H² est limitée à la
dimension 3.

L'interface Python est celle de HiGP, avec une fonction en plus : `higp.set_h2_options`.

## 1. Prérequis

- Linux, `gcc` / `g++` avec OpenMP
- OpenBLAS (paquet `libopenblas-dev` sous Debian / Ubuntu)
- Python ≥ 3.10 avec `numpy`, `scipy`, `torch`, `setuptools` (et `pandas` pour le script de benchmark)

## 2. Compiler

```bash
cd HiGP_H2_arbres/py-interface
python setup.py build_ext --inplace --force
```

Si OpenBLAS n'est pas dans `/usr/lib/x86_64-linux-gnu` :

```bash
HIGP_BLAS_DIR=/chemin/vers/lib python setup.py build_ext --inplace --force
```

La bibliothèque n'est pas installée dans l'environnement : elle s'appelle `higp`, comme
l'originale, et l'installer avec `pip` remplacerait un HiGP déjà présent. On la charge
depuis son dossier (étape suivante).

## 3. Charger la bibliothèque

À mettre en tête de script, **avant** d'importer `numpy` ou `torch` :

```python
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "10")          # voir la section 7
sys.path.insert(0, "/chemin/vers/HiGP_H2_arbres/py-interface")
import numpy as np
import higp
```

## 4. Premier exemple : prédiction

Les données sont des tableaux `(dimension, nombre de points)` contigus.

```python
rng = np.random.default_rng(0)
d, n, m = 10, 5000, 100
x_train = np.ascontiguousarray(rng.uniform(size=(d, n)))
y_train = np.ascontiguousarray(np.sin(3 * x_train[0]) + 0.05 * rng.normal(size=n))
x_test  = np.ascontiguousarray(rng.uniform(size=(d, m)))

# Choix de l'arbre et des réglages H2 (valables pour tous les appels suivants)
higp.set_h2_options(tree="pca", adm_alpha=1.5, min_size=0, verbose=1)

pred = higp.gpr_prediction(
    data_train=x_train, label_train=y_train, data_prediction=x_test,
    kernel_type=higp.Matern32Kernel,
    gp_params=np.array([0.5, 0.5, -2.0]),      # [l, f, s] avant transformation softplus
    mvtype=higp.MatvecAuto)

print(pred.prediction_mean)        # moyenne prédite, taille m
print(pred.prediction_stddev)      # écart-type prédit, taille m
```

`mvtype` choisit la façon de multiplier par la matrice de covariance :

| Valeur | Matrice de covariance |
|---|---|
| `higp.MatvecAuto` | H² si le nombre de points atteint `min_size`, sinon dense |
| `higp.MatvecAOT` | dense, stockée en mémoire |
| `higp.MatvecOTF` | dense, recalculée à chaque produit |

## 5. Entraîner les hyperparamètres

```python
import torch
probleme  = higp.gprproblem.setup(data=x_train, label=y_train,
                                  kernel_type=higp.Matern32Kernel, mvtype=higp.MatvecAuto)
modele    = higp.GPRModel(probleme, dtype=torch.float64)
optimiseur = torch.optim.Adam(modele.parameters(), lr=0.01)
higp.gpr_torch_minimize(modele, optimiseur, maxits=100, print_info=True)
params = modele.get_params()       # à passer à gpr_prediction(gp_params=params)
```

Le type de `dtype` doit correspondre à celui des données (`float32` ou `float64`).

## 6. Les options H2

```python
higp.set_h2_options(tree="2means", adm_alpha=1.2)   # seules les options citées changent
higp.get_h2_options()                               # dictionnaire des réglages courants
```

| Option | Défaut | Effet |
|---|---|---|
| `tree` | `"pca"` | `"pca"` : coupe à la médiane sur l'axe principal, arbre équilibré. `"2means"` : coupe en deux clusters. `"box"` : arbre d'origine, dimension ≤ 3. |
| `adm_alpha` | 1.5 | Deux nœuds sont compressés ensemble si `dist(centres) ≥ adm_alpha × (rayon_i + rayon_j)`. Plus petit : plus de blocs compressés, rangs plus élevés. En dessous de 1, contrôler l'erreur. |
| `min_size` | 20000 | Nombre de points à partir duquel `MatvecAuto` utilise la matrice H². 0 pour toujours. |
| `h2_tol` | 0 | Tolérance de compression. 0 : valeurs de HiGP (1e-10 en double, 1e-5 en simple précision). |
| `max_anchor` | 2048 | Plafond du nombre de points représentant le champ lointain d'un nœud. Plus grand : plus précis, plus lent. |
| `leaf_nmax` | 0 | Points par feuille. 0 : valeur de HiGP (400). |
| `kmeans_min_frac` | 0.05 | 2-means : taille minimale d'un cluster ; en dessous, la coupe PCA est utilisée. |
| `kmeans_max_iter` | 30 | 2-means : itérations de Lloyd par nœud. |
| `knn_max_neighbor` | 27 | Nœuds explorés par la recherche de voisins du préconditionneur. |
| `verbose` | 0 | 1 : afficher l'arbre, le nombre de blocs compressés et les rangs. |

Avec `verbose=1`, deux chiffres disent si la compression fonctionne : le pourcentage de
la matrice couvert par les paires admissibles, et le rang moyen des bases.

## 7. À savoir

- **Nombre de threads.** Avec un OpenBLAS à threads, laisser tous les cœurs à OpenMP peut
  ralentir le calcul d'un facteur 50 ou plus (mesuré sur une machine à 20 cœurs).
  Fixer `OMP_NUM_THREADS` à environ la moitié des cœurs, avant tout import.
- **La matrice H² ne gagne qu'à grande taille.** Sur un problème qui tient en mémoire
  (quelques milliers de points), le calcul dense est plus rapide. Mesure sur le jeu
  3D Road (dimension 2) : à 40 000 points, une itération d'entraînement prend 5,6 s en
  H² contre 25,5 s en dense ; à 3 000 points en dimension 15, c'est l'inverse (0,5 s
  contre 0,05 s).
- **En grande dimension, tout dépend des données.** Sur des points uniformes, presque
  aucun bloc n'est compressible avec `adm_alpha = 1.5` : le résultat est juste, mais sans
  gain. La compression existe quand les données ont une structure (faible dimension
  intrinsèque, clusters).
- **Noyaux.** Gaussien, Matérn 3/2, Matérn 5/2, en toute dimension.

## 8. Scripts fournis

```bash
python exemple_h2_arbres.py                 # prédiction en 20D, comparaison au dense
python benchmark_dataset.py --aide          # RMSE et temps sur un jeu de données réel
python benchmark_dataset.py --donnees mon_fichier.csv --n-train 20000
```

Pour `benchmark_dataset.py`, un fichier CSV numérique suffit : une ligne par point, la
dernière colonne est la valeur à prédire.

## 9. Origine et licence

Ce code dérive de HiGP (Hua Huang, Tianshi Xu, Yuanzhe Xi, Edmond Chow), distribué sous
licence Apache 2.0 ; le fichier `LICENSE` d'origine est conservé. La liste des fichiers
modifiés est dans `MODIFICATIONS.md`. Pour citer HiGP, voir `README.md`.
