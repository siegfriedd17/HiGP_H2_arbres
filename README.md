# HiGP_H2_arbres

**Processus gaussiens avec matrices H² en dimension quelconque.**

HiGP_H2_arbres est une version modifiée de [HiGP](https://github.com/huanghua1994/HiGP),
une bibliothèque Python / C++ de processus gaussiens pour les grands jeux de données
(régression, classification, entraînement des hyperparamètres).

Dans HiGP, la matrice de covariance peut être compressée en matrice H², mais seulement
jusqu'à la dimension 3, parce que la partition des points repose sur un octree. Cette
version remplace l'octree par deux arbres binaires, **PCA** et **2-means**, qui
fonctionnent en toute dimension. Le reste de HiGP est inchangé, et son interface Python
aussi, à une fonction près : `higp.set_h2_options`.

| | HiGP | HiGP_H2_arbres |
|---|---|---|
| Arbre de partition | boîtes (2<sup>d</sup> enfants par nœud) | boîtes, PCA ou 2-means (2 enfants) |
| Matrice H² | dimension ≤ 3 | dimension quelconque |
| Blocs compressibles | boîtes éloignées | boules éloignées : `dist(centres) ≥ α (r_i + r_j)` |
| Construction des bases | points proxy | « sample points » pris dans le champ lointain réel |
| Noyaux de Matérn en dimension > 3 | peuvent renvoyer NaN | corrigé |

## Installation

Prérequis : Linux, `gcc` / `g++` avec OpenMP, OpenBLAS (`libopenblas-dev`), Python ≥ 3.10
avec `numpy`, `scipy`, `torch` et `setuptools`.

```bash
git clone https://github.com/siegfriedd17/HiGP_H2_arbres.git
cd HiGP_H2_arbres/py-interface
python setup.py build_ext --inplace --force
```

Si OpenBLAS n'est pas dans `/usr/lib/x86_64-linux-gnu`, indiquer son dossier :
`HIGP_BLAS_DIR=/chemin/vers/lib python setup.py build_ext --inplace --force`.

Le paquet s'appelle `higp`, comme l'original : il n'est pas installé avec `pip`, pour ne
pas remplacer un HiGP déjà présent. On le charge depuis son dossier.

## Exemple

```python
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "10")              # avant numpy / torch, voir « À savoir »
sys.path.insert(0, "/chemin/vers/HiGP_H2_arbres/py-interface")
import numpy as np
import higp

# Données : tableaux (dimension, nombre de points) contigus
rng = np.random.default_rng(0)
d, n, m = 10, 5000, 100
x_train = np.ascontiguousarray(rng.uniform(size=(d, n)))
y_train = np.ascontiguousarray(np.sin(3 * x_train[0]) + 0.05 * rng.normal(size=n))
x_test  = np.ascontiguousarray(rng.uniform(size=(d, m)))

# Arbre et réglages H2, valables pour tous les appels suivants
higp.set_h2_options(tree="pca", adm_alpha=1.5, min_size=0, verbose=1)

pred = higp.gpr_prediction(
    data_train=x_train, label_train=y_train, data_prediction=x_test,
    kernel_type=higp.Matern32Kernel,
    gp_params=np.array([0.5, 0.5, -2.0]),                   # [l, f, s] avant transformation softplus
    mvtype=higp.MatvecAuto)

print(pred.prediction_mean, pred.prediction_stddev)
```

L'entraînement des hyperparamètres, la classification et les fonctions tout-en-un
s'utilisent comme dans HiGP. Le [tutoriel](TUTORIEL.md) donne un exemple d'entraînement.

## Options H2

```python
higp.set_h2_options(tree="2means", adm_alpha=1.2)   # seules les options citées changent
higp.get_h2_options()                               # réglages courants
```

| Option | Défaut | Effet |
|---|---|---|
| `tree` | `"pca"` | `"pca"` : coupe à la médiane sur l'axe principal, arbre équilibré. `"2means"` : coupe en deux clusters, suit la forme des données. `"box"` : arbre de HiGP, dimension ≤ 3. |
| `adm_alpha` | 1.5 | Seuil d'éloignement à partir duquel deux nœuds sont compressés ensemble. Plus petit : plus de blocs compressés, rangs plus élevés. En dessous de 1, contrôler l'erreur. |
| `min_size` | 20000 | Nombre de points à partir duquel `MatvecAuto` utilise la matrice H². 0 pour toujours. |
| `h2_tol` | 0 | Tolérance de compression. 0 : valeurs de HiGP (1e-10 en double, 1e-5 en simple précision). |
| `max_anchor` | 2048 | Plafond du nombre de points représentant le champ lointain d'un nœud. |
| `leaf_nmax` | 0 | Points par feuille. 0 : valeur de HiGP (400). |
| `kmeans_min_frac` | 0.05 | 2-means : taille minimale d'un cluster ; en dessous, la coupe PCA est utilisée. |
| `kmeans_max_iter` | 30 | 2-means : itérations de Lloyd par nœud. |
| `knn_max_neighbor` | 27 | Nœuds explorés par la recherche de voisins du préconditionneur. |
| `verbose` | 0 | 1 : afficher l'arbre, la part de la matrice compressée et les rangs. |

Le paramètre `mvtype` des fonctions de HiGP choisit la matrice de covariance :
`higp.MatvecAuto` (H² à partir de `min_size` points, sinon dense), `higp.MatvecAOT`
(dense stockée), `higp.MatvecOTF` (dense recalculée à chaque produit).

## Quand la matrice H² est-elle utile ?

Elle gagne à grande taille et perd à petite taille. Mesures sur une machine à 20 cœurs,
10 threads, simple précision, noyau gaussien ; temps d'**une itération d'entraînement**.

| Jeu de données | Points | Dimension | Dense | H², PCA | H², 2-means |
|---|---|---|---|---|---|
| UCI Bike Sharing | 3 000 | 15 | 0,05 s | 0,50 s | 0,51 s |
| UCI 3D Road | 20 000 | 2 | 3,1 s | 2,1 s | 1,8 s |
| UCI 3D Road | 40 000 | 2 | 25,5 s | 5,6 s | 4,3 s |

À 20 000 points sur 3D Road, la mémoire maximale passe de 3,9 Go en dense à 1,0 Go en H².
Dans tous ces cas, la RMSE de la prédiction avec H² reste à moins de 7 % de celle du
calcul dense, un écart du même ordre que celui entre deux entraînements denses.

En grande dimension, le gain dépend des données : la compression existe quand elles ont
une structure (faible dimension intrinsèque, clusters). Sur des points uniformes, presque
aucun bloc n'est compressible avec `adm_alpha = 1.5`, et le calcul reste dense.

## Vérifications

- **Produit par la matrice H²** contre le produit dense, 20 000 points, de la dimension 3
  à la dimension 100, noyaux gaussien, Matérn 3/2 et 5/2 : erreur relative ≤ 3e-9 en
  double précision.
- **Prédiction** contre le GP exact de HiGP, 5 000 points, dimensions 3, 10 et 20 :
  écart relatif ≤ 8e-9 sur la moyenne prédite.
- **Tests de fumée de HiGP** (`py-tests/test_smoke.py`) : ils passent en dense et avec
  la matrice H² imposée, pour les trois arbres.

Non vérifié : les tests C++ du dossier `cpp-tests` n'ont pas été relancés, et la
bibliothèque n'a été compilée que sur une machine (Linux, gcc 15, Python 3.14).

## À savoir

- **Nombre de threads.** Avec un OpenBLAS à threads, laisser tous les cœurs à OpenMP peut
  ralentir le calcul d'un facteur 50 ou plus. Fixer `OMP_NUM_THREADS` à environ la moitié
  des cœurs, avant d'importer `numpy`, `torch` ou `higp`.
- **Entraînement.** Les matrices H² sont reconstruites à chaque itération de
  l'optimiseur, puisque les hyperparamètres changent.
- **Efficacité de la compression.** Elle n'a pas été étudiée de façon systématique
  (rangs, mémoire et temps selon la taille, la dimension, `adm_alpha` et l'arbre).

## Scripts fournis

```bash
python exemple_h2_arbres.py                                   # prédiction en dimension 20, comparaison au dense
python benchmark_dataset.py                                   # RMSE et temps sur UCI Bike Sharing
python benchmark_dataset.py --donnees road3d --n-train 40000 --iterations 5
python benchmark_dataset.py --donnees mon_fichier.csv         # dernière colonne = valeur à prédire
python benchmark_dataset.py --aide
```

## Ce qui a changé par rapport à HiGP

- **Arbres PCA et 2-means**, centre et rayon par nœud, options globales : `cpp-src/h2mat/octree.cpp`.
- **Admissibilité par boules** : `cpp-src/h2mat/h2mat_utils.cpp`.
- **Sample points** (nouveau fichier) : `cpp-src/h2mat/h2mat_sample_points.cpp`.
- **Construction et produit H²** adaptés à un nombre d'enfants variable : `h2mat_build.cpp`, `h2mat_matmul.cpp`, `ss_h2mat.cpp`.
- **Recherche de voisins du préconditionneur** : `cpp-src/solvers/fsai_precond.cpp`.
- **Limite de dimension levée** : `cpp-src/gp/precond_gp.cpp`, `py-interface/higp_cext.c`.
- **Correction des NaN** des noyaux de Matérn en dimension > 3 : `cpp-src/kernels/matern32_kernel.cpp`, `matern52_kernel.cpp`.

La liste complète, fichier par fichier, est dans [MODIFICATIONS.md](MODIFICATIONS.md).

## Documentation

- [TUTORIEL.md](TUTORIEL.md) : prise en main de cette version.
- [docs/](docs/) : documentation d'origine de HiGP (usage de base et avancé, référence de l'API), toujours valable.

## Origine, citation et licence

Ce code dérive de HiGP, développé par Hua Huang, Tianshi Xu, Yuanzhe Xi et Edmond Chow.
Il est distribué sous la même licence, Apache 2.0 (fichier [LICENSE](LICENSE)).

Si vous utilisez ce code, merci de citer HiGP :

```text
@misc{HiGP2025,
    author = {Hua Huang and Tianshi Xu and Yuanzhe Xi and Edmond Chow},
    title = {{HiGP}: A high-performance {Python} package for {Gaussian} Process},
    year = {2025},
    eprint = {arXiv:2503.02259},
}
```

Les arbres PCA et 2-means, l'admissibilité par boules et l'adaptation des sample points
ont été ajoutés par Mohamed Chabani.
