#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>

#include "octree.h"
#include "../common.h"
#include "../utils.h"

#include <vector>

#define OCTREE_MAX_DIM      3
#define OCTREE_MAX_CHILDREN (1 << OCTREE_MAX_DIM)

// =========================== Global options =========================== //

static h2_tree_options_s h2_tree_options_ = {
    H2_TREE_PCA,    // tree_type
    1.5,            // adm_alpha
    0,              // leaf_nmax
    2048,           // sp_max_anchor
    30,             // kmeans_max_iter
    0.05,           // kmeans_min_frac
    27,             // knn_max_neighbor
    20000,          // h2_min_size
    0,              // verbose
    0.0             // h2_reltol
};

void h2_tree_get_options(h2_tree_options_s *opt) { *opt = h2_tree_options_; }

void h2_tree_set_options(const h2_tree_options_s *opt)
{
    h2_tree_options_s o = *opt;
    if (o.tree_type != H2_TREE_BOX && o.tree_type != H2_TREE_PCA && o.tree_type != H2_TREE_2MEANS)
    {
        ERROR_PRINTF("Unknown partitioning tree type %d, keep the current one\n", o.tree_type);
        o.tree_type = h2_tree_options_.tree_type;
    }
    if (!(o.adm_alpha > 0.0))       o.adm_alpha        = h2_tree_options_.adm_alpha;
    if (o.sp_max_anchor < 16)       o.sp_max_anchor    = 16;
    if (o.kmeans_max_iter < 1)      o.kmeans_max_iter  = 1;
    if (o.kmeans_min_frac < 0.0)    o.kmeans_min_frac  = 0.0;
    if (o.kmeans_min_frac > 0.5)    o.kmeans_min_frac  = 0.5;
    if (o.knn_max_neighbor < 1)     o.knn_max_neighbor = 1;
    if (o.h2_min_size < 0)          o.h2_min_size      = 0;
    h2_tree_options_ = o;
}

int h2_tree_dim_supported(const int pt_dim)
{
    if (pt_dim < 1) return 0;
    if (h2_tree_options_.tree_type == H2_TREE_BOX && pt_dim > OCTREE_MAX_DIM) return 0;
    return 1;
}

// ============================== Box tree ============================== //

// Extend the enclosing box a little bit to avoid numerical issues
// The original enclosing box and the extended enclosing box have the same center
template<typename VT>
static void extend_enbox(const int pt_dim, VT *enbox)
{
    for (int i = 0; i < pt_dim; i++)
    {
        VT ext_size = enbox[pt_dim + i] * 1e-5;
        enbox[i] -= ext_size;
        enbox[pt_dim + i] += 2.0 * ext_size;
    }
}

// Calculate the minimal enclosing box of a point set
// Input parameters:
//   npt    : Number of points in the point set
//   pt_dim : Dimension of the points
//   x      : Size ldx * pt_dim, col-major matrix, each row in x is a point coordinate
//   ldx    : Leading dimension of x, >= npt
// Output parameter:
//   enbox : Size 2 * pt_dim, the first pt_dim values are the lower corner, the 
//           second pt_dim values are the sizes of the enclosing box
template<typename VT>
static void calc_min_enbox(const int npt, const int pt_dim, const VT *x, const int ldx, VT *enbox)
{
    for (int i = 0; i < pt_dim; i++)
    {
        const VT *x_i = x + i * ldx;
        VT max_c = *std::max_element(x_i, x_i + npt);
        VT min_c = *std::min_element(x_i, x_i + npt);
        enbox[i] = min_c;
        enbox[pt_dim + i] = max_c - min_c;
    }
    extend_enbox(pt_dim, enbox);
}

// Recursive partitioning of the point
// Input parameters:
//   npt        : Number of points in the point set
//   pt_dim     : Dimension of the points
//   leaf_nmax  : Maximum number of points in a leaf node
//   leaf_emax  : Maximum size of the enclosing box of a leaf node
//   x_enbox    : If the enclosing box of the point set is already specified. 
//                If NULL, the enclosing box will be calculated from the point set
//   ldx        : Leading dimension of x and x_tmp, should always == global npt
//   x0idx      : Global index of the 1st point in x
//   my_level   : Level of the current node (root = 0)
//   workv      : Size global npt * pt_dim, VT work buffer
//   worki      : Size global npt * 3, int work buffer
// Input and output parameters:
//   x          : Size ldx * pt_dim, col-major matrix, each row in x is a point coordinate
//   pidx       : Size npt, the original point index of each point in x
//   node_lvl   : Size global npt, level of each node, root node at level 0
//   parent     : Size global npt, parent node index of each node
//   children   : Size global npt * (2^pt_dim), row-major, each row contains children indices of each node
//   n_children : Size global npt, number of children of each node
//   enbox      : Size global npt * (2 * pt_dim), each row contains the enclosing box of a node
//   pt_cluster : Size global npt * 2, start and end (included) indices of permuted points in each node
//   node_npt   : Size global npt, number of points in each node
//   node_idx   : Current number of nodes assigned in the whole tree
template<typename VT>
static int octree_recursive_partition(
    const int npt, const int pt_dim, const int leaf_nmax, const VT leaf_emax,
    const VT *x_enbox, VT *x, const int ldx, const int x0idx, int *pidx, 
    const int my_level, int *node_lvl, int *parent, int *children, int *n_children, 
    VT *enbox, int *pt_cluster, int *node_npt, int *node_idx, VT *workv, int *worki
)
{
    int max_children = 1 << pt_dim;

    // 1. Calculate the enclosing box of the point set
    VT curr_enbox[2 * OCTREE_MAX_DIM], curr_center[OCTREE_MAX_DIM];
    if (x_enbox == NULL) calc_min_enbox<VT>(npt, pt_dim, x, ldx, &curr_enbox[0]);
    else memcpy(&curr_enbox[0], x_enbox, sizeof(VT) * 2 * pt_dim);
    VT *curr_enbox_size = &curr_enbox[pt_dim];
    for (int i = 0; i < pt_dim; i++)
        curr_center[i] = curr_enbox[i] + curr_enbox_size[i] * 0.5;

    // 2. Check if the number of points or the size of the enclosing box is small enough
    VT max_enbox_size = *std::max_element(curr_enbox_size, curr_enbox_size + pt_dim);
    if ((npt <= leaf_nmax) || (max_enbox_size <= leaf_emax))
    {
        int my_node_idx = *node_idx;
        *node_idx += 1;
        node_lvl[my_node_idx] = my_level;
        n_children[my_node_idx] = 0;
        pt_cluster[2 * my_node_idx] = x0idx;
        pt_cluster[2 * my_node_idx + 1] = x0idx + npt - 1;
        node_npt[my_node_idx] = npt;
        memcpy(enbox + my_node_idx * 2 * pt_dim, curr_enbox, sizeof(VT) * 2 * pt_dim);
        return my_node_idx;
    }

    // 3. Assign each point to a child node
    int *clu_idx = worki;
    int clu_cnt[OCTREE_MAX_CHILDREN];
    memset(clu_cnt, 0, sizeof(int) * max_children);
    memset(clu_idx, 0, sizeof(int) * npt);
    int pow2k = 1;
    for (int k = 0; k < pt_dim; k++)
    {
        VT *x_k = x + k * ldx;
        for (int i = 0; i < npt; i++)
            if (x_k[i] >= curr_center[k]) clu_idx[i] += pow2k;
        pow2k *= 2;
    }
    for (int i = 0; i < npt; i++) clu_cnt[clu_idx[i]]++;

    // 4. Compute the enbox of each child node
    VT clu_enbox[OCTREE_MAX_CHILDREN * 2 * OCTREE_MAX_DIM];
    pow2k = 1;
    for (int k = 0; k < pt_dim; k++)
    {
        for (int i = 0; i < max_children; i += pow2k * 2)
        {
            for (int j = i; j < i + 2 * pow2k; j++)
            {
                VT *clu_enbox_j = &clu_enbox[j * 2 * pt_dim];
                clu_enbox_j[k] = (j < i + pow2k) ? curr_enbox[k] : curr_center[k];
                clu_enbox_j[pt_dim + k] = curr_enbox_size[k] * 0.5;
            }
        }
        pow2k *= 2;
    }
    for (int j = 0; j < max_children; j++)
    {
        VT *clu_enbox_j = &clu_enbox[j * 2 * pt_dim];
        extend_enbox(pt_dim, clu_enbox_j);
    }

    // 5. Shuffle the points
    VT *x_tmp = workv;
    int *pidx_tmp = worki + npt;
    int *perm_idx = worki + npt * 2;
    int clu_displs[OCTREE_MAX_CHILDREN];
    clu_displs[0] = 0;
    for (int i = 1; i < max_children; i++) clu_displs[i] = clu_displs[i - 1] + clu_cnt[i - 1];
    for (int i = 0; i < npt; i++)
    {
        int idx = clu_displs[clu_idx[i]];
        clu_displs[clu_idx[i]]++;
        perm_idx[idx] = i;
        pidx_tmp[idx] = pidx[i];
    }
    for (int j = 0; j < pt_dim; j++)
    {
        VT *src = x + j * ldx;
        VT *dst = x_tmp + j * ldx;
        for (int i = 0; i < npt; i++) dst[i] = src[perm_idx[i]];
    }
    clu_displs[0] = 0;
    for (int i = 1; i < max_children; i++) clu_displs[i] = clu_displs[i - 1] + clu_cnt[i - 1];
    for (int j = 0; j < pt_dim; j++)
    {
        VT *src = x_tmp + j * ldx;
        VT *dst = x + j * ldx;
        memcpy(dst, src, sizeof(VT) * npt);
    }
    memcpy(pidx, pidx_tmp, sizeof(int) * npt);

    // 5. Recursively partition the points in each cluster
    int children_cnt = 0;
    int children_nidx[OCTREE_MAX_CHILDREN];
    for (int i = 0; i < max_children; i++)
    {
        if (clu_cnt[i] == 0) continue;
        int disp = clu_displs[i];
        VT *clu_enbox_i = &clu_enbox[i * 2 * pt_dim];
        children_nidx[children_cnt] = octree_recursive_partition<VT>(
            clu_cnt[i], pt_dim, leaf_nmax, leaf_emax, 
            clu_enbox_i, x + disp, ldx, x0idx + disp, pidx + disp, 
            my_level + 1, node_lvl, parent, children, n_children, 
            enbox, pt_cluster, node_npt, node_idx, workv, worki
        );
        children_cnt++;
    }

    // 6. Add current node to the tree
    int my_node_idx = *node_idx;
    *node_idx += 1;
    node_lvl[my_node_idx] = my_level;
    n_children[my_node_idx] = children_cnt;
    pt_cluster[2 * my_node_idx] = x0idx;
    pt_cluster[2 * my_node_idx + 1] = x0idx + npt - 1;
    node_npt[my_node_idx] = npt;
    for (int i = 0; i < children_cnt; i++)
    {
        children[my_node_idx * max_children + i] = children_nidx[i];
        parent[children_nidx[i]] = my_node_idx;
    }
    memcpy(enbox + my_node_idx * 2 * pt_dim, curr_enbox, sizeof(VT) * 2 * pt_dim);

    return my_node_idx;
}

template<typename VT>
static void octree_build(
    const int npt, const int pt_dim, const int val_type, const VT *coord, 
    const int leaf_nmax, const VT leaf_emax, octree_p *octree_
)
{
    ASSERT_PRINTF(
        pt_dim <= OCTREE_MAX_DIM && pt_dim >= 1,
        "Expected 1 <= pt_dim <= %d, got pt_dim = %d\n", OCTREE_MAX_DIM, pt_dim
    );

    octree_p octree = (octree_p) malloc(sizeof(octree_s));
    *octree_ = octree;
    octree->npt = npt;
    octree->pt_dim = pt_dim;
    int max_children = 1 << pt_dim;
    octree->tree_type    = H2_TREE_BOX;
    octree->max_children = max_children;
    octree->adm_alpha    = 0.0;
    octree->center       = NULL;
    octree->radius       = NULL;

    // 1. Find a cubic enclosing box of the point set, so the enclosing box of 
    //    all nodes are also cubic (required for proxy point generation)
    VT root_enbox[2 * OCTREE_MAX_DIM];
    calc_min_enbox<VT>(npt, pt_dim, coord, npt, &root_enbox[0]);
    VT max_enbox_size = *std::max_element(&root_enbox[0] + pt_dim, &root_enbox[0] + 2 * pt_dim);
    max_enbox_size *= (1.0 + 2e-5);
    for (int i = 0; i < pt_dim; i++)
    {
        VT center_i = root_enbox[i] + root_enbox[pt_dim + i] * 0.5;
        root_enbox[i] = center_i - max_enbox_size * 0.5;
        root_enbox[pt_dim + i] = max_enbox_size;
    }

    // 2. Recursively partition the points using a given enclosing box
    //    to ensure that for all nodes at the same level, their enclosing boxes
    //    have the same size (required for proxy point generation)
    int *worki0 = (int *) malloc(sizeof(int) * npt * (max_children + 9));
    VT  *workv0 = (VT  *) malloc(sizeof(VT)  * npt * pt_dim * 4);
    assert(worki0 != NULL && workv0 != NULL);
    int *pidx       = worki0;
    int *node_lvl   = pidx       + npt;
    int *parent     = node_lvl   + npt;
    int *children   = parent     + npt;
    int *n_children = children   + npt * max_children;
    int *pt_cluster = n_children + npt;
    int *node_npt   = pt_cluster + npt;
    int *worki      = node_npt   + npt;  // npt * 3
    VT  *x1         = workv0;
    VT  *enbox      = x1    + npt * pt_dim;
    VT  *workv      = enbox + npt * pt_dim * 2;  // npt * pt_dim
    int node_idx = 0;
    for (int i = 0; i < npt; i++) pidx[i] = i;
    memcpy(x1, coord, sizeof(VT) * pt_dim * npt);
    int root_idx = octree_recursive_partition<VT>(
        npt, pt_dim, leaf_nmax, leaf_emax,
        &root_enbox[0], x1, npt, 0, pidx, 
        0, node_lvl, parent, children, n_children, 
        enbox, pt_cluster, node_npt, &node_idx, workv, worki
    );
    parent[root_idx] = -1;  // Root node has no parent

    // 2. Copy partitioning tree data
    int n_node = node_idx;
    int max_level = *std::max_element(node_lvl, node_lvl + n_node);
    int n_level = max_level + 1;
    octree->n_node   = n_node;
    octree->n_level  = n_level;
    octree->val_type = val_type;
    octree->parent     = (int *) malloc(sizeof(int) * n_node);
    octree->children   = (int *) malloc(sizeof(int) * n_node * max_children);
    octree->n_children = (int *) malloc(sizeof(int) * n_node);
    octree->node_lvl   = (int *) malloc(sizeof(int) * n_node);
    octree->node_npt   = (int *) malloc(sizeof(int) * n_node);
    octree->lvl_nnode  = (int *) malloc(sizeof(int) * n_level);
    octree->lvl_nodes  = (int *) malloc(sizeof(int) * n_node);
    octree->ln_displs  = (int *) malloc(sizeof(int) * (n_level + 1));
    octree->pt_cluster = (int *) malloc(sizeof(int) * n_node * 2);
    octree->fwd_perm   = (int *) malloc(sizeof(int) * npt);
    octree->bwd_perm   = (int *) malloc(sizeof(int) * npt);
    octree->px         = (VT  *) malloc(sizeof(VT)  * npt * pt_dim);
    octree->enbox      = (VT  *) malloc(sizeof(VT)  * n_node * pt_dim * 2);
    memcpy(octree->parent,     parent,     sizeof(int) * n_node);
    memcpy(octree->children,   children,   sizeof(int) * n_node * max_children);
    memcpy(octree->n_children, n_children, sizeof(int) * n_node);
    memcpy(octree->node_lvl,   node_lvl,   sizeof(int) * n_node);
    memcpy(octree->node_npt,   node_npt,   sizeof(int) * n_node);
    memcpy(octree->pt_cluster, pt_cluster, sizeof(int) * n_node * 2);
    memcpy(octree->px,         x1,         sizeof(VT)  * npt * pt_dim);
    memcpy(octree->enbox,      enbox,      sizeof(VT)  * n_node * pt_dim * 2);
    memcpy(octree->bwd_perm,   pidx,       sizeof(int) * npt);
    for (int i = 0; i < npt; i++) octree->fwd_perm[pidx[i]] = i;
    int *lvl_nnode = octree->lvl_nnode;
    int *lvl_nodes = octree->lvl_nodes;
    int *ln_displs = octree->ln_displs;
    memset(lvl_nnode, 0, sizeof(int) * n_level);
    for (int i = 0; i < n_node; i++) lvl_nnode[node_lvl[i]]++;
    ln_displs[0] = 0;
    for (int i = 1; i <= n_level; i++) 
        ln_displs[i] = ln_displs[i - 1] + lvl_nnode[i - 1];
    for (int i = 0; i < n_node; i++)
    {
        int idx = ln_displs[node_lvl[i]];
        ln_displs[node_lvl[i]]++;
        lvl_nodes[idx] = i;
    }
    ln_displs[0] = 0;
    for (int i = 1; i <= n_level; i++) 
        ln_displs[i] = ln_displs[i - 1] + lvl_nnode[i - 1];

    free(worki0);
    free(workv0);
}

// ======================== PCA and 2-means trees ======================== //

// Find the k-th smallest value (k starts from 0) of an array, the array is permuted
template<typename VT>
static VT ball_tree_quickselect(VT *a, const int n, const int k)
{
    int l = 0, r = n - 1;
    while (l < r)
    {
        VT pivot = a[l + (r - l) / 2];
        int i = l, j = r;
        while (i <= j)
        {
            while (a[i] < pivot) i++;
            while (a[j] > pivot) j--;
            if (i <= j)
            {
                std::swap(a[i], a[j]);
                i++;
                j--;
            }
        }
        if (k <= j) r = j;
        else if (k >= i) l = i;
        else return a[k];
    }
    return a[k];
}

// Calculate the centroid, the minimal enclosing box, and the radius (max distance 
// to the centroid) of a point cluster
// Input parameters:
//   npt, pt_dim : Number of points and point dimension
//   x, ldx      : Size ldx * pt_dim, col-major, each row is a point coordinate
// Output parameters:
//   center   : Size pt_dim, centroid of the points
//   enbox    : Size 2 * pt_dim, lower corner and sizes of the enclosing box
//   dist2    : Size npt, work array
//   <return> : Radius of the cluster
template<typename VT>
static VT ball_tree_node_geometry(const int npt, const int pt_dim, const VT *x, const int ldx, VT *center, VT *enbox, VT *dist2)
{
    memset(dist2, 0, sizeof(VT) * npt);
    for (int k = 0; k < pt_dim; k++)
    {
        const VT *x_k = x + (size_t) k * ldx;
        double sum = 0.0;   // Accumulate in double, also for float coordinates
        VT min_c = x_k[0], max_c = x_k[0];
        for (int i = 0; i < npt; i++)
        {
            sum += (double) x_k[i];
            if (x_k[i] < min_c) min_c = x_k[i];
            if (x_k[i] > max_c) max_c = x_k[i];
        }
        VT center_k = (VT) (sum / (double) npt);
        center[k] = center_k;
        enbox[k]  = min_c;
        enbox[pt_dim + k] = max_c - min_c;
        #pragma omp simd
        for (int i = 0; i < npt; i++)
        {
            VT diff = x_k[i] - center_k;
            dist2[i] += diff * diff;
        }
    }
    VT max_dist2 = *std::max_element(dist2, dist2 + npt);
    // The radius must not be smaller than the true one after rounding
    return std::sqrt(max_dist2) * (VT) (1.0 + 1e-6);
}

// Calculate the principal axis of a point cluster (eigenvector of the largest eigenvalue 
// of the covariance matrix, by power iteration) and project the points on it
// Input parameters:
//   npt, pt_dim, x, ldx : Same as ball_tree_node_geometry()
//   center              : Size pt_dim, centroid of the points
// Output parameter:
//   proj : Size npt, projection of the centered points on the principal axis
template<typename VT>
static void ball_tree_principal_axis(const int npt, const int pt_dim, const VT *x, const int ldx, const VT *center, VT *proj)
{
    std::vector<double> cov((size_t) pt_dim * pt_dim), axis(pt_dim, 0.0), axis1(pt_dim);

    // 1. Covariance matrix (not scaled by npt, it does not change the eigenvectors)
    for (int j = 0; j < pt_dim; j++)
    {
        const VT *x_j = x + (size_t) j * ldx;
        const VT center_j = center[j];
        for (int k = j; k < pt_dim; k++)
        {
            const VT *x_k = x + (size_t) k * ldx;
            const VT center_k = center[k];
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int i = 0; i < npt; i++)
                sum += (double) (x_j[i] - center_j) * (double) (x_k[i] - center_k);
            cov[(size_t) j * pt_dim + k] = sum;
            cov[(size_t) k * pt_dim + j] = sum;
        }
    }

    // 2. Power iteration, starting from the coordinate axis of largest variance. Any 
    //    direction of (near) maximal variance is good enough to split the points.
    int max_var_dim = 0;
    for (int j = 1; j < pt_dim; j++)
        if (cov[(size_t) j * pt_dim + j] > cov[(size_t) max_var_dim * pt_dim + max_var_dim]) max_var_dim = j;
    axis[max_var_dim] = 1.0;
    if (cov[(size_t) max_var_dim * pt_dim + max_var_dim] > 0.0)
    {
        for (int iter = 0; iter < 200; iter++)
        {
            double norm2 = 0.0;
            for (int j = 0; j < pt_dim; j++)
            {
                const double *cov_j = &cov[(size_t) j * pt_dim];
                double sum = 0.0;
                for (int k = 0; k < pt_dim; k++) sum += cov_j[k] * axis[k];
                axis1[j] = sum;
                norm2 += sum * sum;
            }
            double norm = std::sqrt(norm2);
            if (!(norm > 0.0)) break;
            double diff2 = 0.0;
            for (int j = 0; j < pt_dim; j++)
            {
                double axis1_j = axis1[j] / norm;
                diff2 += (axis1_j - axis[j]) * (axis1_j - axis[j]);
                axis[j] = axis1_j;
            }
            if (diff2 < 1e-20) break;
        }
    }

    // 3. Project the centered points on the principal axis
    memset(proj, 0, sizeof(VT) * npt);
    for (int k = 0; k < pt_dim; k++)
    {
        const VT *x_k = x + (size_t) k * ldx;
        const VT center_k = center[k], axis_k = (VT) axis[k];
        #pragma omp simd
        for (int i = 0; i < npt; i++) proj[i] += axis_k * (x_k[i] - center_k);
    }
}

// Split a point cluster in two halves at the median of the projections of its points 
// on its principal axis. Points whose projection equals the median are split between 
// the two children such that the first child has exactly npt / 2 points: both children 
// are always non-empty.
// Output parameters:
//   proj_sel : Size npt, work array
//   clu      : Size npt, 0 / 1, child each point goes to
//   <return> : Number of points in the first child
template<typename VT>
static int ball_tree_median_split(const int npt, const VT *proj, VT *proj_sel, int *clu)
{
    int n_left = npt / 2;
    memcpy(proj_sel, proj, sizeof(VT) * npt);
    VT median = ball_tree_quickselect<VT>(proj_sel, npt, n_left - 1);
    int n_less = 0;
    for (int i = 0; i < npt; i++) if (proj[i] < median) n_less++;
    int n_tie_left = n_left - n_less;
    for (int i = 0; i < npt; i++)
    {
        if (proj[i] < median) clu[i] = 0;
        else if (proj[i] == median && n_tie_left > 0)
        {
            clu[i] = 0;
            n_tie_left--;
        }
        else clu[i] = 1;
    }
    return n_left;
}

// Refine a split of a point cluster in two with the Lloyd iterations of 2-means: compute 
// the centroid of each child, move each point to the child with the nearest centroid, 
// repeat until no point moves. A point is nearer to centroid c1 than to c0 iff 
// w . (x - m) > 0 with w = c1 - c0 and m = (c0 + c1) / 2.
// Input parameters:
//   npt, pt_dim, x, ldx : Same as ball_tree_node_geometry()
//   min_child           : Minimum number of points of a child, >= 1
//   max_iter            : Maximum number of Lloyd iterations
//   clu                 : Size npt, 0 / 1, initial split
// Output parameters:
//   clu      : Size npt, 0 / 1, child each point goes to (not usable if <return> == -1)
//   score    : Size npt, work array
//   <return> : Number of points in the first child, or -1 if a child has less than 
//              min_child points (the points do not form two clusters)
template<typename VT>
static int ball_tree_2means_refine(
    const int npt, const int pt_dim, const VT *x, const int ldx, 
    const int min_child, const int max_iter, int *clu, VT *score
)
{
    std::vector<VT> w(pt_dim), mid(pt_dim);
    int n1 = 0;
    for (int iter = 0; iter < max_iter; iter++)
    {
        n1 = 0;
        for (int i = 0; i < npt; i++) n1 += clu[i];
        int n0 = npt - n1;
        if (n0 < min_child || n1 < min_child) break;
        for (int k = 0; k < pt_dim; k++)
        {
            const VT *x_k = x + (size_t) k * ldx;
            double sum = 0.0, sum1 = 0.0;
            for (int i = 0; i < npt; i++)
            {
                sum  += (double) x_k[i];
                sum1 += (double) x_k[i] * (double) clu[i];
            }
            double c0_k = (sum - sum1) / (double) n0;
            double c1_k = sum1 / (double) n1;
            w[k]   = (VT) (c1_k - c0_k);
            mid[k] = (VT) (0.5 * (c0_k + c1_k));
        }
        memset(score, 0, sizeof(VT) * npt);
        for (int k = 0; k < pt_dim; k++)
        {
            const VT *x_k = x + (size_t) k * ldx;
            const VT w_k = w[k], mid_k = mid[k];
            #pragma omp simd
            for (int i = 0; i < npt; i++) score[i] += w_k * (x_k[i] - mid_k);
        }
        int n_moved = 0;
        for (int i = 0; i < npt; i++)
        {
            int clu_i = (score[i] > 0) ? 1 : 0;
            n_moved += (clu_i != clu[i]);
            clu[i] = clu_i;
        }
        if (n_moved == 0) break;
    }
    n1 = 0;
    for (int i = 0; i < npt; i++) n1 += clu[i];
    if (npt - n1 < min_child || n1 < min_child) return -1;
    return npt - n1;
}

// Working data of the construction of a PCA / 2-means tree
template<typename VT>
struct ball_tree_work
{
    int npt0, pt_dim, leaf_nmax, tree_type, kmeans_max_iter;
    double kmeans_min_frac;
    VT  leaf_emax;
    VT  *x;                 // Size npt0 * pt_dim, col-major, points, permuted in place
    int *pidx;              // Size npt0, original index of each point in x
    VT  *x_tmp, *proj, *proj_sel;   // Size npt0 each, work arrays
    int *pidx_tmp, *clu;            // Size npt0 each, work arrays
    std::vector<int> node_lvl, parent, children, n_children, pt_cluster;
    std::vector<VT>  enbox, center, radius;
};

// Recursive partitioning of the points [x0idx, x0idx + npt) of a PCA / 2-means tree.
// PCA tree     : the node is split in two halves at the median of the projections of its 
//                points on its principal axis.
// 2-means tree : the node is split in two clusters by 2-means started from the PCA split. 
//                If one of the clusters is too small (outliers, duplicated points), the 
//                PCA split is kept for this node, so that the depth of the tree stays bounded.
// Nodes are numbered in post order: children before their parent, root node last.
// Return the index of the node.
template<typename VT>
static int ball_tree_recursive_partition(ball_tree_work<VT> &w, const int x0idx, const int npt, const int my_level)
{
    const int pt_dim = w.pt_dim, ldx = w.npt0;
    VT *x = w.x + x0idx;
    std::vector<VT> center(pt_dim), enbox(2 * pt_dim);
    VT radius = ball_tree_node_geometry<VT>(npt, pt_dim, x, ldx, center.data(), enbox.data(), w.proj);

    int child_idx[2] = {-1, -1}, n_child = 0;
    int is_leaf = (npt <= w.leaf_nmax) || (w.leaf_emax > 0 && 2 * radius <= w.leaf_emax) || (npt < 2);
    if (!is_leaf)
    {
        // 1. Split at the median of the projections on the principal axis
        ball_tree_principal_axis<VT>(npt, pt_dim, x, ldx, center.data(), w.proj);
        int n_left = ball_tree_median_split<VT>(npt, w.proj, w.proj_sel, w.clu);

        // 2. 2-means tree: refine the PCA split into two clusters, go back to the 
        //    PCA split if the points of this node do not form two clusters
        if (w.tree_type == H2_TREE_2MEANS)
        {
            int min_child = (int) (w.kmeans_min_frac * (double) npt);
            if (min_child < 1) min_child = 1;
            int n_left_2means = ball_tree_2means_refine<VT>(npt, pt_dim, x, ldx, min_child, w.kmeans_max_iter, w.clu, w.proj_sel);
            if (n_left_2means > 0) n_left = n_left_2means;
            else n_left = ball_tree_median_split<VT>(npt, w.proj, w.proj_sel, w.clu);
        }

        // 3. Stable reordering of the points: first child, then second child
        int pos[2] = {0, n_left};
        int *perm = w.pidx_tmp;
        for (int i = 0; i < npt; i++) perm[pos[w.clu[i]]++] = i;
        for (int k = 0; k < pt_dim; k++)
        {
            VT *x_k = x + (size_t) k * ldx;
            for (int i = 0; i < npt; i++) w.x_tmp[i] = x_k[perm[i]];
            memcpy(x_k, w.x_tmp, sizeof(VT) * npt);
        }
        int *pidx = w.pidx + x0idx;
        for (int i = 0; i < npt; i++) w.clu[i] = pidx[perm[i]];
        memcpy(pidx, w.clu, sizeof(int) * npt);

        // 4. Recursively partition the two children
        child_idx[0] = ball_tree_recursive_partition<VT>(w, x0idx, n_left, my_level + 1);
        child_idx[1] = ball_tree_recursive_partition<VT>(w, x0idx + n_left, npt - n_left, my_level + 1);
        n_child = 2;
    }

    // 5. Add current node to the tree
    int my_node_idx = (int) w.node_lvl.size();
    w.node_lvl.push_back(my_level);
    w.parent.push_back(-1);
    w.n_children.push_back(n_child);
    w.children.push_back(child_idx[0]);
    w.children.push_back(child_idx[1]);
    w.pt_cluster.push_back(x0idx);
    w.pt_cluster.push_back(x0idx + npt - 1);
    w.enbox.insert(w.enbox.end(), enbox.begin(), enbox.end());
    w.center.insert(w.center.end(), center.begin(), center.end());
    w.radius.push_back(radius);
    for (int i = 0; i < n_child; i++) w.parent[child_idx[i]] = my_node_idx;
    return my_node_idx;
}

template<typename VT>
static void ball_tree_build(
    const int npt, const int pt_dim, const int val_type, const VT *coord, const int leaf_nmax, 
    const VT leaf_emax, const int tree_type, const double adm_alpha, octree_p *octree_
)
{
    ASSERT_PRINTF(pt_dim >= 1 && npt >= 1, "Expected pt_dim >= 1 and npt >= 1, got pt_dim = %d, npt = %d\n", pt_dim, npt);

    // 1. Recursively partition the points
    ball_tree_work<VT> w;
    w.npt0 = npt;
    w.pt_dim = pt_dim;
    w.leaf_nmax = (leaf_nmax >= 1) ? leaf_nmax : 1;
    w.leaf_emax = leaf_emax;
    w.tree_type = tree_type;
    w.kmeans_max_iter = h2_tree_options_.kmeans_max_iter;
    w.kmeans_min_frac = h2_tree_options_.kmeans_min_frac;
    std::vector<VT>  workv((size_t) npt * (pt_dim + 3));
    std::vector<int> worki((size_t) npt * 3);
    w.x        = workv.data();
    w.x_tmp    = w.x     + (size_t) npt * pt_dim;
    w.proj     = w.x_tmp + npt;
    w.proj_sel = w.proj  + npt;
    w.pidx     = worki.data();
    w.pidx_tmp = w.pidx     + npt;
    w.clu      = w.pidx_tmp + npt;
    memcpy(w.x, coord, sizeof(VT) * (size_t) npt * pt_dim);
    for (int i = 0; i < npt; i++) w.pidx[i] = i;
    ball_tree_recursive_partition<VT>(w, 0, npt, 0);

    // 2. Copy partitioning tree data
    octree_p octree = (octree_p) malloc(sizeof(octree_s));
    *octree_ = octree;
    int n_node  = (int) w.node_lvl.size();
    int n_level = *std::max_element(w.node_lvl.begin(), w.node_lvl.end()) + 1;
    octree->tree_type    = tree_type;
    octree->max_children = 2;
    octree->adm_alpha    = adm_alpha;
    octree->npt      = npt;
    octree->pt_dim   = pt_dim;
    octree->n_node   = n_node;
    octree->n_level  = n_level;
    octree->val_type = val_type;
    octree->parent     = (int *) malloc(sizeof(int) * n_node);
    octree->children   = (int *) malloc(sizeof(int) * n_node * 2);
    octree->n_children = (int *) malloc(sizeof(int) * n_node);
    octree->node_lvl   = (int *) malloc(sizeof(int) * n_node);
    octree->node_npt   = (int *) malloc(sizeof(int) * n_node);
    octree->lvl_nnode  = (int *) malloc(sizeof(int) * n_level);
    octree->lvl_nodes  = (int *) malloc(sizeof(int) * n_node);
    octree->ln_displs  = (int *) malloc(sizeof(int) * (n_level + 1));
    octree->pt_cluster = (int *) malloc(sizeof(int) * n_node * 2);
    octree->fwd_perm   = (int *) malloc(sizeof(int) * npt);
    octree->bwd_perm   = (int *) malloc(sizeof(int) * npt);
    octree->px         = (VT  *) malloc(sizeof(VT)  * (size_t) npt * pt_dim);
    octree->enbox      = (VT  *) malloc(sizeof(VT)  * (size_t) n_node * pt_dim * 2);
    octree->center     = (VT  *) malloc(sizeof(VT)  * (size_t) n_node * pt_dim);
    octree->radius     = (VT  *) malloc(sizeof(VT)  * n_node);
    memcpy(octree->parent,     w.parent.data(),     sizeof(int) * n_node);
    memcpy(octree->children,   w.children.data(),   sizeof(int) * n_node * 2);
    memcpy(octree->n_children, w.n_children.data(), sizeof(int) * n_node);
    memcpy(octree->node_lvl,   w.node_lvl.data(),   sizeof(int) * n_node);
    memcpy(octree->pt_cluster, w.pt_cluster.data(), sizeof(int) * n_node * 2);
    memcpy(octree->px,         w.x,                 sizeof(VT)  * (size_t) npt * pt_dim);
    memcpy(octree->enbox,      w.enbox.data(),      sizeof(VT)  * (size_t) n_node * pt_dim * 2);
    memcpy(octree->center,     w.center.data(),     sizeof(VT)  * (size_t) n_node * pt_dim);
    memcpy(octree->radius,     w.radius.data(),     sizeof(VT)  * n_node);
    memcpy(octree->bwd_perm,   w.pidx,              sizeof(int) * npt);
    for (int i = 0; i < npt; i++) octree->fwd_perm[w.pidx[i]] = i;
    for (int i = 0; i < n_node; i++) octree->node_npt[i] = w.pt_cluster[2 * i + 1] - w.pt_cluster[2 * i] + 1;
    int *lvl_nnode = octree->lvl_nnode;
    int *lvl_nodes = octree->lvl_nodes;
    int *ln_displs = octree->ln_displs;
    memset(lvl_nnode, 0, sizeof(int) * n_level);
    for (int i = 0; i < n_node; i++) lvl_nnode[w.node_lvl[i]]++;
    ln_displs[0] = 0;
    for (int i = 1; i <= n_level; i++) ln_displs[i] = ln_displs[i - 1] + lvl_nnode[i - 1];
    for (int i = 0; i < n_node; i++) lvl_nodes[ln_displs[w.node_lvl[i]]++] = i;
    ln_displs[0] = 0;
    for (int i = 1; i <= n_level; i++) ln_displs[i] = ln_displs[i - 1] + lvl_nnode[i - 1];
}

// ============================ Public interface ============================ //

void octree_build_type(
    const int npt, const int pt_dim, const int val_type, const void *coord, 
    const int leaf_nmax, const void *leaf_emax, const int tree_type, 
    const double adm_alpha, octree_p *octree
)
{
    int tree_type_ = tree_type;
    if (tree_type_ == H2_TREE_BOX && pt_dim > OCTREE_MAX_DIM)
    {
        WARNING_PRINTF("The box tree is limited to pt_dim <= %d, use the PCA tree for pt_dim = %d\n", OCTREE_MAX_DIM, pt_dim);
        tree_type_ = H2_TREE_PCA;
    }
    double alpha = (adm_alpha > 0.0) ? adm_alpha : h2_tree_options_.adm_alpha;
    int leaf_nmax_ = (h2_tree_options_.leaf_nmax > 0) ? h2_tree_options_.leaf_nmax : leaf_nmax;
    if (val_type == VAL_TYPE_DOUBLE)
    {
        const double leaf_emax_ = *((const double *) leaf_emax);
        if (tree_type_ == H2_TREE_BOX) octree_build<double>(npt, pt_dim, val_type, (const double *) coord, leaf_nmax_, leaf_emax_, octree);
        else ball_tree_build<double>(npt, pt_dim, val_type, (const double *) coord, leaf_nmax_, leaf_emax_, tree_type_, alpha, octree);
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        const float leaf_emax_ = *((const float *) leaf_emax);
        if (tree_type_ == H2_TREE_BOX) octree_build<float>(npt, pt_dim, val_type, (const float *) coord, leaf_nmax_, leaf_emax_, octree);
        else ball_tree_build<float>(npt, pt_dim, val_type, (const float *) coord, leaf_nmax_, leaf_emax_, tree_type_, alpha, octree);
    }
    if (h2_tree_options_.verbose) octree_print_info(*octree);
}

void octree_build(
    const int npt, const int pt_dim, const int val_type, const void *coord, 
    const int leaf_nmax, const void *leaf_emax, octree_p *octree
)
{
    octree_build_type(npt, pt_dim, val_type, coord, leaf_nmax, leaf_emax, h2_tree_options_.tree_type, 0.0, octree);
}

// Print the main characteristics of a partitioning tree
void octree_print_info(octree_p octree)
{
    if (octree == NULL) return;
    int n_leaf = 0, min_leaf = octree->npt, max_leaf = 0, min_leaf_lvl = octree->n_level;
    for (int i = 0; i < octree->n_node; i++)
    {
        if (octree->n_children[i] > 0) continue;
        n_leaf++;
        if (octree->node_npt[i] < min_leaf) min_leaf = octree->node_npt[i];
        if (octree->node_npt[i] > max_leaf) max_leaf = octree->node_npt[i];
        if (octree->node_lvl[i] < min_leaf_lvl) min_leaf_lvl = octree->node_lvl[i];
    }
    const char *names[3] = {"box", "PCA", "2-means"};
    printf("[H2 tree] %s tree, %d points in %d-D, %s precision\n", names[octree->tree_type], octree->npt, octree->pt_dim, 
           (octree->val_type == VAL_TYPE_DOUBLE) ? "double" : "single");
    printf("[H2 tree]   %d nodes, %d leaves, %d levels (shallowest leaf at level %d), leaf size %d..%d\n", 
           octree->n_node, n_leaf, octree->n_level, min_leaf_lvl, min_leaf, max_leaf);
    if (H2_TREE_IS_BALL(octree->tree_type)) printf("[H2 tree]   admissible coefficient alpha = %.3f\n", octree->adm_alpha);
    fflush(stdout);
}

// Free an octree struct
void octree_free(octree_p *octree)
{
    octree_p octree_ = *octree;
    if (octree_ == NULL) return;
    free(octree_->parent);
    free(octree_->children);
    free(octree_->n_children);
    free(octree_->node_lvl);
    free(octree_->node_npt);
    free(octree_->lvl_nnode);
    free(octree_->lvl_nodes);
    free(octree_->ln_displs);
    free(octree_->pt_cluster);
    free(octree_->fwd_perm);
    free(octree_->bwd_perm);
    free(octree_->px);
    free(octree_->enbox);
    free(octree_->center);
    free(octree_->radius);
    free(octree_);
    *octree = NULL;
}
