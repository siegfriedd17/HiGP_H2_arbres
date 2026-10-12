# Modifications par rapport à HiGP

HiGP_H2_arbres dérive de [HiGP](https://github.com/huanghua1994/HiGP) (licence Apache 2.0).
Ce fichier liste les changements, comme le demande la licence. Modifications faites en
octobre 2026.

## Objet

Permettre l'usage de la matrice H² en dimension quelconque : arbres de partition binaires
(PCA, 2-means), admissibilité par boules, construction des bases par « sample points ».

## Fichiers ajoutés

| Fichier | Contenu |
|---|---|
| `cpp-src/h2mat/h2mat_sample_points.cpp`, `.h` | Sélection des sample points de chaque nœud |
| `exemple_h2_arbres.py` | Exemple de prédiction |
| `benchmark_dataset.py` | RMSE et temps sur un jeu de données réel |
| `TUTORIEL.md`, `MODIFICATIONS.md` | Documentation |

## Fichiers modifiés

| Fichier | Changement |
|---|---|
| `cpp-src/h2mat/octree.cpp`, `.h` | Arbres PCA et 2-means, options globales du module H2, centre et rayon par nœud |
| `cpp-src/h2mat/h2mat_utils.cpp`, `.h` | Test d'admissibilité par boules |
| `cpp-src/h2mat/h2mat_build.cpp`, `.h` | Construction avec sample points, nombre d'enfants variable, statistiques |
| `cpp-src/h2mat/h2mat_matmul.cpp` | Nombre d'enfants variable |
| `cpp-src/h2mat/ss_h2mat.cpp`, `.h`, `h2mat.h` | Sample points pour K et dK/dl sur les arbres PCA et 2-means |
| `cpp-src/solvers/fsai_precond.cpp` | Recherche de voisins par nœuds aux centres les plus proches |
| `cpp-src/gp/precond_gp.cpp`, `cpp-src/common.h` | Limite de dimension levée, tolérance H2 réglable |
| `cpp-src/kernels/matern32_kernel.cpp`, `matern52_kernel.cpp` | Correction : NaN en dimension > 3 (distance au carré légèrement négative par arrondi) |
| `py-interface/higp_cext.c`, `.h` | Fonctions `set_h2_options` et `get_h2_options`, choix automatique dense / H2 |
| `py-interface/setup.py` | Nouveau fichier source, emplacement d'OpenBLAS réglable |

Tous les autres fichiers sont ceux de HiGP.
