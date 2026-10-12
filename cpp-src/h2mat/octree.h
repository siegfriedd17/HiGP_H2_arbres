#ifndef __OCTREE_H__
#define __OCTREE_H__

// Partitioning tree types
#define H2_TREE_BOX     0   // 2^pt_dim-ary tree of axis-aligned boxes (binary tree, quadtree, or octree), pt_dim <= 3
#define H2_TREE_PCA     1   // Binary tree, a node is split at the median along its principal axis, any pt_dim
#define H2_TREE_2MEANS  2   // Binary tree, a node is split in two clusters by 2-means, any pt_dim
#define H2_TREE_IS_BALL(t) ((t) != H2_TREE_BOX)   // PCA & 2-means trees: nodes are described by a center and a radius

// Global options of the H2 matrix module. They are read when a tree / an H2 matrix 
// is built, so that the GP code does not have to carry them around.
struct h2_tree_options
{
    int    tree_type;           // H2_TREE_BOX, H2_TREE_PCA (default), or H2_TREE_2MEANS
    double adm_alpha;           // PCA & 2-means trees: nodes (i, j) are admissible if 
                                // dist(center_i, center_j) >= adm_alpha * (radius_i + radius_j), default 1.5
    int    leaf_nmax;           // Maximum number of points in a leaf node; <= 0 (default): use the value given by the caller
    int    sp_max_anchor;       // Maximum number of points of a sample point anchor grid, default 2048
    int    kmeans_max_iter;     // 2-means tree: maximum number of Lloyd iterations for splitting a node, default 30
    double kmeans_min_frac;     // 2-means tree: if a cluster gets less than this fraction of the points of a 
                                // node, the node is split at the median along its principal axis, default 0.05
    int    knn_max_neighbor;    // PCA & 2-means trees: number of nearest nodes searched in the FSAI fast KNN, default 27
    int    h2_min_size;         // Smallest number of points for which the "auto" matvec type uses an H2 matrix, default 20000
    int    verbose;             // If 1, print information on each tree and H2 matrix that is built, default 0
    double h2_reltol;           // H2 compression relative tolerance used by the GP code; <= 0 (default): built-in 
                                // values (1e-10 in double precision, 1e-5 in single precision)
};
typedef struct h2_tree_options h2_tree_options_s;

// Binary tree, quadtree, or octree
struct octree
{
    int  tree_type;         // H2_TREE_BOX, H2_TREE_PCA, or H2_TREE_2MEANS
    int  max_children;      // Maximum number of children of a node: 2^pt_dim (box tree) or 2 (PCA & 2-means trees)
    double adm_alpha;       // PCA & 2-means trees: admissible coefficient
    void *center;           // PCA & 2-means trees only: size n_node * pt_dim, row-major, centroid of the points of each node
    void *radius;           // PCA & 2-means trees only: size n_node, max distance between a node's centroid and its points
    int  npt, pt_dim;       // Number of points and point dimension (box tree: <= 3)
    int  n_level;           // Number of levels, root node (box) at level 0
    int  n_node;            // Number of nodes, root node has index n_node-1
    int  val_type;          // Data type of px and enbox, 0 for double, 1 for float
    int  *parent;           // Size n_node, parent node index of each node
    int  *children;         // Size n_node * max_children, row-major, each row contains children indices of each node
    int  *n_children;       // Size n_node, number of children of each node
    int  *node_lvl;         // Size n_node, level of each node, root node at level 0
    int  *node_npt;         // Size n_node, number of points in each node
    int  *lvl_nnode;        // Size n_level, number of nodes in each level
    int  *lvl_nodes;        // Size n_node, indices of nodes in each level
    int  *ln_displs;        // Size n_level+1, i-th level's node indices are lvl_nodes[ln_displs[i] : ln_displs[i+1]-1]
    int  *pt_cluster;       // Size n_node * 2, row-major, start and end (included) indices of permuted points in each node
    int  *fwd_perm;         // Size npt, the i-th original point is the fwd_perm[i]-th permuted points in the octree
    int  *bwd_perm;         // Size npt, the i-th permuted points in the octree is the bwd_perm[i]-th original point
    void *px;               // Size npt * pt_dim, col-major, coordinates of permuted points
    void *enbox;            // Size n_node * (2 * pt_dim), row-major, each row contains the enclosing box of a node, 
                            // the first pt_dim values are the lower corner, the last pt_dim values are the sizes of the enclosing box.
                            // Box tree: the regular box of the node; PCA & 2-means trees: the minimal box of the node's points.
};
typedef struct octree  octree_s;
typedef struct octree *octree_p;

#ifdef __cplusplus
extern "C" {
#endif

// Get / set the global options of the H2 matrix module
void h2_tree_get_options(h2_tree_options_s *opt);
void h2_tree_set_options(const h2_tree_options_s *opt);

// Check if the current tree type can partition points of dimension pt_dim
// (the box tree is limited to pt_dim <= 3, the other trees work in any dimension)
int h2_tree_dim_supported(const int pt_dim);

// Build a partitioning tree for a given point set. The tree type is the one in the 
// global options (h2_tree_set_options()): PCA binary tree by default.
// Input parameters:
//   npt, pt_dim : Number of points and point dimension (box tree: <= 3)
//   val_type    : Data type of coord, 0 for double, 1 for float
//   coord       : Size npt * pt_dim, col-major, each row is a point coordinate
//   leaf_nmax   : Maximum number of points in a leaf node
//   leaf_emax   : Maximum size of the enclosing box of a leaf node
// Output parameter:
//   *octree : Pointer to a constructed octree struct
void octree_build(
    const int npt, const int pt_dim, const int val_type, const void *coord, 
    const int leaf_nmax, const void *leaf_emax, octree_p *octree
);

// Same as octree_build(), with an explicit tree type and admissible coefficient
// Input parameters:
//   tree_type : H2_TREE_BOX, H2_TREE_PCA, or H2_TREE_2MEANS
//   adm_alpha : PCA & 2-means trees admissible coefficient, <= 0 for the value in the global options
void octree_build_type(
    const int npt, const int pt_dim, const int val_type, const void *coord, 
    const int leaf_nmax, const void *leaf_emax, const int tree_type, 
    const double adm_alpha, octree_p *octree
);

// Print the main characteristics of a partitioning tree
void octree_print_info(octree_p octree);

// Free an octree struct
void octree_free(octree_p *octree);

#ifdef __cplusplus
}
#endif

#endif  // "#ifndef __OCTREE_H__"
