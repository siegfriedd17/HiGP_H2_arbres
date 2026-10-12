#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>
#include <vector>
#include "omp.h"

#include "../common.h"
#include "../cpu_linalg.hpp"
#include "h2mat_sample_points.h"
#include "h2mat_build.h"
#include "id_ppqr.h"

// Sample points are an alternative to proxy points for building the basis of a node:
// a small set of REAL points taken in the far field of the node (the nodes that are
// admissible with it or with one of its ancestors). They do not assume anything on the
// shape of the nodes, so they work with any partitioning tree and any point dimension.
// The selection algorithm is the one of H2Pack (H2Pack_build_with_sample_point.c),
// itself based on the MATLAB code of the authors of doi/10.1109/IPDPS47924.2020.00082.

// A set of points: n points of dimension dim, stored col-major, x[i + k * n] is the
// k-th coordinate of the i-th point
template<typename VT>
struct sp_ptset
{
    int n = 0;
    std::vector<VT> x;
};

// Keep some points of a point set
template<typename VT>
static void sp_select_points(sp_ptset<VT> &ps, const int dim, const std::vector<int> &idx)
{
    int n_new = (int) idx.size();
    std::vector<VT> x_new((size_t) n_new * dim);
    for (int k = 0; k < dim; k++)
        for (int i = 0; i < n_new; i++) x_new[i + (size_t) k * n_new] = ps.x[idx[i] + (size_t) k * ps.n];
    ps.x.swap(x_new);
    ps.n = n_new;
}

// Append the points of a point set to another point set
template<typename VT>
static void sp_append_points(sp_ptset<VT> &dst, const int dim, const sp_ptset<VT> &src)
{
    if (src.n == 0) return;
    int n_new = dst.n + src.n;
    std::vector<VT> x_new((size_t) n_new * dim);
    for (int k = 0; k < dim; k++)
    {
        if (dst.n > 0) memcpy(&x_new[(size_t) k * n_new], &dst.x[(size_t) k * dst.n], sizeof(VT) * dst.n);
        memcpy(&x_new[(size_t) k * n_new + dst.n], &src.x[(size_t) k * src.n], sizeof(VT) * src.n);
    }
    dst.x.swap(x_new);
    dst.n = n_new;
}

// Minimal enclosing box of a point set
template<typename VT>
static void sp_enbox(const int dim, const VT *x, const int n, const int ldx, VT *cmin, VT *cmax, VT *size)
{
    for (int k = 0; k < dim; k++)
    {
        const VT *x_k = x + (size_t) k * ldx;
        VT min_c = x_k[0], max_c = x_k[0];
        for (int i = 1; i < n; i++)
        {
            if (x_k[i] < min_c) min_c = x_k[i];
            if (x_k[i] > max_c) max_c = x_k[i];
        }
        cmin[k] = min_c;
        cmax[k] = max_c;
        size[k] = max_c - min_c;
    }
}

// Decompose an integer into the sum of nelem integers which are approximately
// proportional to prop[0 : nelem-1]. decomp[] defines a tensor grid with
// (decomp[0]+1) * ... * (decomp[nelem-1]+1) points. This number is 2^decomp_sum when
// nelem >= decomp_sum, i.e. in high dimension: decomp_sum is reduced until the grid has
// no more than max_npt points.
// Return the number of points of the tensor grid.
template<typename VT>
static int sp_proportional_int_decompose(const int nelem, const int decomp_sum, const VT *prop, int *decomp, const int max_npt)
{
    double sum_prop = 0.0;
    for (int i = 0; i < nelem; i++) sum_prop += (double) prop[i];
    if (!(sum_prop > 0.0))
    {
        // All points coincide: a single anchor point
        for (int i = 0; i < nelem; i++) decomp[i] = 0;
        return 1;
    }
    std::vector<double> decomp_prop(nelem);
    long long prod1 = 1;
    for (int sum = decomp_sum; sum >= 0; sum--)
    {
        int decomp_sum0 = 0;
        for (int i = 0; i < nelem; i++)
        {
            decomp_prop[i] = (double) sum * (double) prop[i] / sum_prop;
            decomp[i] = (int) std::floor(decomp_prop[i]);
            decomp_sum0 += decomp[i];
        }
        for (int k = decomp_sum0; k < sum; k++)
        {
            // Add 1 to the position that got hit most by floor
            int max_idx = 0;
            double max_diff = decomp_prop[0] - (double) decomp[0];
            for (int i = 1; i < nelem; i++)
            {
                double diff = decomp_prop[i] - (double) decomp[i];
                if (diff > max_diff)
                {
                    max_diff = diff;
                    max_idx = i;
                }
            }
            decomp[max_idx]++;
        }
        prod1 = 1;
        for (int i = 0; i < nelem; i++)
        {
            prod1 *= (long long) (decomp[i] + 1);
            if (prod1 > (long long) max_npt) break;
        }
        if (prod1 <= (long long) max_npt) break;
    }
    return (int) prod1;
}

// Generate a tensor grid of anchor points in a box
// Input parameters:
//   dim        : Point dimension
//   cmin, cmax : Size dim, corners of the box
//   size       : Size dim, cmax - cmin
//   grid_size  : Size dim, the grid has grid_size[k] + 1 points in dimension k
//   grid_algo  : 2 for Chebyshev points, 6 for uniform points
// Output parameter:
//   anchor : Anchor points, col-major, anchor.n points
template<typename VT>
static void sp_anchor_grid(
    const int dim, const VT *cmin, const VT *cmax, const VT *size,
    const int *grid_size, const int grid_algo, sp_ptset<VT> &anchor
)
{
    int max_grid_size = 0, anchor_npt = 1;
    for (int k = 0; k < dim; k++)
    {
        if (grid_size[k] > max_grid_size) max_grid_size = grid_size[k];
        anchor_npt *= (grid_size[k] + 1);
    }
    max_grid_size++;

    // 1. Anchor coordinates in each dimension
    std::vector<VT> anchor_dim((size_t) dim * max_grid_size);
    for (int k = 0; k < dim; k++)
    {
        VT *anchor_dim_k = &anchor_dim[(size_t) k * max_grid_size];
        if (grid_size[k] == 0)
        {
            anchor_dim_k[0] = (cmin[k] + cmax[k]) * (VT) 0.5;
        } else if (grid_algo == 2) {
            VT s0 = (VT) 0.5 * (cmax[k] + cmin[k]);
            VT s1 = (VT) 0.5 * (cmax[k] - cmin[k]);
            double s2 = M_PI / (2.0 * (double) grid_size[k] + 2.0);
            for (int j = 0; j <= grid_size[k]; j++)
                anchor_dim_k[j] = s0 + s1 * (VT) std::cos((2.0 * (double) j + 1.0) * s2);
        } else {
            VT size_k = size[k] / ((VT) grid_size[k] + (VT) 0.5);
            for (int j = 0; j <= grid_size[k]; j++)
                anchor_dim_k[j] = cmin[k] + (VT) 0.25 * size_k + size_k * (VT) j;
        }
    }

    // 2. Tensor product
    anchor.n = anchor_npt;
    anchor.x.resize((size_t) anchor_npt * dim);
    int stride = 1;
    for (int k = 0; k < dim; k++)
    {
        const VT *anchor_dim_k = &anchor_dim[(size_t) k * max_grid_size];
        VT *anchor_k = &anchor.x[(size_t) k * anchor_npt];
        for (int i = 0; i < anchor_npt; i++) anchor_k[i] = anchor_dim_k[(i / stride) % (grid_size[k] + 1)];
        stride *= (grid_size[k] + 1);
    }
}

// Index of the point of a point set which is the nearest to a given point
// Input parameters:
//   dim       : Point dimension
//   y, ldy, i : The given point is the i-th point of y (col-major, leading dimension ldy)
//   x, n, ldx : Point set, n points, col-major, leading dimension ldx
//   dist2     : Size n, work array
template<typename VT>
static int sp_nearest_point(const int dim, const VT *y, const int ldy, const int i, const VT *x, const int n, const int ldx, VT *dist2)
{
    memset(dist2, 0, sizeof(VT) * n);
    for (int k = 0; k < dim; k++)
    {
        const VT y_k = y[i + (size_t) k * ldy];
        const VT *x_k = x + (size_t) k * ldx;
        #pragma omp simd
        for (int j = 0; j < n; j++)
        {
            VT diff = y_k - x_k[j];
            dist2[j] += diff * diff;
        }
    }
    int min_idx = 0;
    for (int j = 1; j < n; j++) if (dist2[j] < dist2[min_idx]) min_idx = j;
    return min_idx;
}

// For each point of an anchor grid laid over a point cluster, choose the nearest point
// of the cluster as a sample point
// Input parameters:
//   dim       : Point dimension
//   ps        : Point cluster
//   grid_size : Size dim, anchor grid size
//   grid_algo : Anchor grid generation algorithm
// Output parameter:
//   sample_idx : Sorted indices of the chosen points
// In high dimension (dim larger than the grid parameter) the anchor grid has at most 2 
// points per dimension: its points are corners of the enclosing box, and the points 
// nearest to them are only extreme points of the cluster. If fill_npt > 0, the selection 
// is then completed with randomly chosen points of the cluster (fixed seed), up to 
// fill_npt points in total, so that the interior of the cluster is represented too.
template<typename VT>
static void sp_select_cluster_sample(
    const int dim, const sp_ptset<VT> &ps, const int *grid_size, const int grid_algo, 
    std::vector<int> &sample_idx, const int fill_npt = 0
)
{
    int n = ps.n;
    std::vector<VT> cmin(dim), cmax(dim), size(dim), dist2(n);
    sp_enbox<VT>(dim, ps.x.data(), n, n, cmin.data(), cmax.data(), size.data());
    sp_ptset<VT> anchor;
    sp_anchor_grid<VT>(dim, cmin.data(), cmax.data(), size.data(), grid_size, grid_algo, anchor);
    std::vector<char> flag(n, 0);
    for (int i = 0; i < anchor.n; i++)
        flag[sp_nearest_point<VT>(dim, anchor.x.data(), anchor.n, i, ps.x.data(), n, n, dist2.data())] = 1;
    int max_grid_size = 0, n_sel = 0;
    for (int k = 0; k < dim; k++) if (grid_size[k] > max_grid_size) max_grid_size = grid_size[k];
    for (int i = 0; i < n; i++) n_sel += flag[i];
    if (fill_npt > 0 && max_grid_size <= 1 && n_sel < fill_npt && n_sel < n)
    {
        int target = (fill_npt < n) ? fill_npt : n;
        std::mt19937 gen(19241112u + (unsigned) n);
        std::vector<int> perm(n);
        for (int i = 0; i < n; i++) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), gen);
        for (int i = 0; i < n && n_sel < target; i++)
        {
            if (flag[perm[i]]) continue;
            flag[perm[i]] = 1;
            n_sel++;
        }
    }
    sample_idx.clear();
    for (int i = 0; i < n; i++) if (flag[i]) sample_idx.push_back(i);
}

// Find the anchor grid parameter r for a kernel and a target accuracy: two clusters of
// random points separated like two admissible nodes are generated, and r is increased
// until the sample points chosen in one cluster give an accurate interpolative
// decomposition of the kernel block between the two clusters.
// Input parameters:
//   dim        : Point dimension
//   k          : Number of points in each cluster
//   L          : Size of the clusters
//   tau        : Separation threshold, the second cluster is shifted by L / tau
//   krnl       : Kernel matrix evaluation function
//   param      : Kernel matrix parameters
//   reltol     : H2 compression relative error tolerance
//   max_anchor : Maximum number of points of an anchor grid
template<typename VT>
static int sp_select_r(
    const int dim, const int k, const VT L, const VT tau, krnl_func krnl,
    const void *param, const VT reltol, const int max_anchor
)
{
    const int val_type = std::is_same<VT, double>::value ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;
    const VT eps = std::numeric_limits<VT>::epsilon();
    VT stop_tol = reltol * (VT) 1e-3;
    if (stop_tol < 50 * eps) stop_tol = 50 * eps;
    VT stop_err = reltol * (VT) 1e-4;
    if (stop_err < 500 * eps) stop_err = 500 * eps;

    // The initial guess formula is the one of H2Pack
    int r1 = (int) std::floor(-1.25 * std::log10((double) stop_tol) - 0.25);
    if (r1 < 1) r1 = 1;
    int r2 = (int) std::ceil(std::sqrt(std::floor(2.0 * std::log10((double) stop_tol) / std::log10(0.7))));
    int r = (r1 < r2) ? r1 : r2;

    // Two sets of points in boxes of size L. A local random number generator with a
    // fixed seed is used: the result is reproducible and rand() is not disturbed.
    std::mt19937_64 gen(19241112);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    sp_ptset<VT> c0, c1;
    c0.n = c1.n = k;
    c0.x.resize((size_t) k * dim);
    c1.x.resize((size_t) k * dim);
    VT shift = L / tau;
    for (size_t i = 0; i < (size_t) k * dim; i++)
    {
        c0.x[i] = L * (VT) unif(gen);
        c1.x[i] = L * (VT) unif(gen) + shift;
    }

    // A0 = K(c1, c0), the block to approximate
    std::vector<VT> A0((size_t) k * k), A0J, UA((size_t) k * k);
    krnl(k, k, c1.x.data(), k, k, c0.x.data(), param, k, A0.data(), val_type);
    double A0_fnorm = 0.0;
    for (size_t i = 0; i < (size_t) k * k; i++) A0_fnorm += (double) A0[i] * (double) A0[i];
    A0_fnorm = std::sqrt(A0_fnorm);
    // In high dimension the two clusters can be so far apart that the block is
    // numerically zero: nothing to calibrate, keep the initial guess
    if (!(A0_fnorm > 0.0) || !std::isfinite(A0_fnorm)) return r;

    std::vector<VT> size(dim), cmin(dim), cmax(dim);
    std::vector<int> r_list(dim), sample_idx;
    sp_enbox<VT>(dim, c1.x.data(), k, k, cmin.data(), cmax.data(), size.data());
    int last_rank = 1;
    const VT v_one = 1.0, v_zero = 0.0;
    for (int iter = 0; iter < 64; iter++)
    {
        // Sample points of c1, and A1 = K(c1(sample_idx), c0)
        sp_proportional_int_decompose<VT>(dim, r, size.data(), r_list.data(), max_anchor);
        sp_select_cluster_sample<VT>(dim, c1, r_list.data(), 6, sample_idx);
        sp_ptset<VT> c1s = c1;
        sp_select_points<VT>(c1s, dim, sample_idx);
        int n_sample = c1s.n;
        std::vector<VT> A1((size_t) n_sample * k);
        krnl(n_sample, n_sample, c1s.x.data(), k, k, c0.x.data(), param, n_sample, A1.data(), val_type);

        // ID of A1: A1 ~= A1(:, J) * V, J are skeleton points of c0
        int rank = 0, *col_idx = NULL;
        void *V = NULL;
        id_ppqr(n_sample, k, val_type, A1.data(), n_sample, 0, (const void *) &stop_tol, 1, &rank, &col_idx, &V, NULL, NULL);

        // err = || A0(:, J) * V - A0 ||_F / || A0 ||_F
        double relerr = 1.0;
        if (rank > 0)
        {
            A0J.resize((size_t) k * rank);
            for (int j = 0; j < rank; j++) memcpy(&A0J[(size_t) j * k], &A0[(size_t) col_idx[j] * k], sizeof(VT) * k);
            xgemm_(notrans, notrans, &k, &k, &rank, &v_one, A0J.data(), &k, (VT *) V, &rank, &v_zero, UA.data(), &k);
            double err_fnorm = 0.0;
            for (size_t i = 0; i < (size_t) k * k; i++)
            {
                double diff = (double) A0[i] - (double) UA[i];
                err_fnorm += diff * diff;
            }
            relerr = std::sqrt(err_fnorm) / A0_fnorm;
        }
        free(col_idx);
        free(V);

        int stop = 0;
        if (rank - last_rank < 2) stop = 1;     // More sample points do not help any more
        if (relerr < (double) stop_err) stop = 1;
        last_rank = rank;
        if (stop) break;
        r++;
    }
    return r;
}

template<typename VT>
static void h2m_tree_sample_points(
    octree_p octree, const int n_krnl, krnl_func *krnls, const void *param,
    const VT reltol, h2m_2dbuf_p **sample_points
)
{
    const int npt        = octree->npt;
    const int dim        = octree->pt_dim;
    const int n_level    = octree->n_level;
    const int n_node     = octree->n_node;
    const int max_child  = octree->max_children;
    const int *n_children = octree->n_children;
    const int *children  = octree->children;
    const int *lvl_nnode = octree->lvl_nnode;
    const int *lvl_nodes = octree->lvl_nodes;
    const int *ln_displs = octree->ln_displs;
    const int *pt_cluster = octree->pt_cluster;
    const VT  *px        = (const VT *) octree->px;
    const VT  *enbox     = (const VT *) octree->enbox;

    h2_tree_options_s opt;
    h2_tree_get_options(&opt);
    const int max_anchor = opt.sp_max_anchor;

    // 1. Far (admissible) lists of each node
    int min_far_lvl = 0;
    std::vector<int> node_n_near(n_node), node_n_far(n_node);
    int *node_near = (int *) malloc(sizeof(int) * (size_t) n_node * n_node);
    int *node_far  = (int *) malloc(sizeof(int) * (size_t) n_node * n_node);
    ASSERT_PRINTF(node_near != NULL && node_far != NULL, "Failed to allocate near / far lists for %d nodes\n", n_node);
    h2m_calc_near_far_lists(octree, node_n_near.data(), node_near, node_n_far.data(), node_far, &min_far_lvl);
    free(node_near);

    // 2. Anchor grid parameter. The calibration clusters have the size used by H2Pack:
    //    1 / 6 of the root box (largest extent, as for a cubic root box).
    int max_leaf_npt = 1;
    for (int i = 0; i < n_node; i++)
        if (n_children[i] == 0 && octree->node_npt[i] > max_leaf_npt) max_leaf_npt = octree->node_npt[i];
    const int root = lvl_nodes[ln_displs[0]];
    const VT *root_size = enbox + (size_t) root * 2 * dim + dim;
    VT L = *std::max_element(root_size, root_size + dim) / (VT) 6.0;
    int ri = 1;
    if (L > 0)
    {
        for (int i = 0; i < n_krnl; i++)
        {
            int ri_i = sp_select_r<VT>(dim, 2 * max_leaf_npt + 300, L, (VT) 0.7, krnls[i], param, reltol, max_anchor);
            if (ri_i > ri) ri = ri_i;
        }
    }
    const int r_up = ri;
    const int r_down = (ri > 3) ? ((ri + 3 > 10) ? (ri + 3) : 10) : (ri + 3);

    std::vector<sp_ptset<VT>> clu_refine(n_node), sample_pt(n_node);

    // 3. Bottom-up sweep: each node keeps a thinned subset of its points ("refined
    //    points"). All levels are swept: a leaf above min_far_lvl can still be in the
    //    far field of a deeper node.
    for (int l = n_level - 1; l >= 0; l--)
    {
        const int *lvl_l_nodes = lvl_nodes + ln_displs[l];
        #pragma omp parallel
        {
            std::vector<VT> cmin(dim), cmax(dim), size(dim);
            std::vector<int> ri_list(dim), sample_idx;
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < lvl_nnode[l]; j++)
            {
                int node = lvl_l_nodes[j];
                sp_ptset<VT> &cr = clu_refine[node];
                if (n_children[node] == 0)
                {
                    int pt_s = pt_cluster[2 * node];
                    cr.n = pt_cluster[2 * node + 1] - pt_s + 1;
                    cr.x.resize((size_t) cr.n * dim);
                    for (int k = 0; k < dim; k++)
                        memcpy(&cr.x[(size_t) k * cr.n], px + pt_s + (size_t) k * npt, sizeof(VT) * cr.n);
                } else {
                    const int *node_children = children + (size_t) node * max_child;
                    for (int k = 0; k < n_children[node]; k++) sp_append_points<VT>(cr, dim, clu_refine[node_children[k]]);
                }
                if (cr.n <= 5) continue;
                sp_enbox<VT>(dim, cr.x.data(), cr.n, cr.n, cmin.data(), cmax.data(), size.data());
                int ri_npt = sp_proportional_int_decompose<VT>(dim, r_up, size.data(), ri_list.data(), max_anchor);
                if (ri_npt < cr.n)
                {
                    sp_select_cluster_sample<VT>(dim, cr, ri_list.data(), 6, sample_idx, max_anchor);
                    sp_select_points<VT>(cr, dim, sample_idx);
                }
            }
        }
    }

    // 4. Top-down sweep: the sample points of a node are chosen among the refined points
    //    of its far nodes, added to the sample points inherited from its parent, thinned
    //    again and passed to its children.
    for (int l = min_far_lvl; l < n_level; l++)
    {
        const int *lvl_l_nodes = lvl_nodes + ln_displs[l];
        #pragma omp parallel
        {
            std::vector<VT> cmin(dim), cmax(dim), size(dim), dist2;
            std::vector<int> ri_list(dim), sample_idx;
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < lvl_nnode[l]; j++)
            {
                int node = lvl_l_nodes[j];
                int n_far = node_n_far[node];
                const int *far = node_far + (size_t) node * n_node;
                sp_ptset<VT> &sp = sample_pt[node];

                if (n_far > 0)
                {
                    // (1) Enclosing box of all far-field refined points
                    int nFp = 0;
                    for (int k = 0; k < dim; k++)
                    {
                        cmin[k] =  std::numeric_limits<VT>::max();
                        cmax[k] = -std::numeric_limits<VT>::max();
                    }
                    for (int f = 0; f < n_far; f++)
                    {
                        const sp_ptset<VT> &cr = clu_refine[far[f]];
                        nFp += cr.n;
                        for (int k = 0; k < dim; k++)
                        {
                            const VT *x_k = &cr.x[(size_t) k * cr.n];
                            for (int i = 0; i < cr.n; i++)
                            {
                                if (x_k[i] < cmin[k]) cmin[k] = x_k[i];
                                if (x_k[i] > cmax[k]) cmax[k] = x_k[i];
                            }
                        }
                    }
                    for (int k = 0; k < dim; k++) size[k] = cmax[k] - cmin[k];

                    sp_ptset<VT> yF;
                    int ri_npt0 = sp_proportional_int_decompose<VT>(dim, r_down, size.data(), ri_list.data(), max_anchor);
                    if (ri_npt0 > nFp)
                    {
                        // (2a) Few far-field refined points: take them all
                        for (int f = 0; f < n_far; f++) sp_append_points<VT>(yF, dim, clu_refine[far[f]]);
                    } else {
                        // (2b) Lay a Chebyshev anchor grid over the far field, give each anchor
                        //      to the far node with the nearest center, and take in each far
                        //      node the refined points nearest to its anchors
                        sp_ptset<VT> centers, anchor;
                        centers.n = n_far;
                        centers.x.resize((size_t) n_far * dim);
                        for (int f = 0; f < n_far; f++)
                        {
                            const sp_ptset<VT> &cr = clu_refine[far[f]];
                            for (int k = 0; k < dim; k++)
                            {
                                const VT *x_k = &cr.x[(size_t) k * cr.n];
                                double sum = 0.0;
                                for (int i = 0; i < cr.n; i++) sum += (double) x_k[i];
                                centers.x[f + (size_t) k * n_far] = (VT) (sum / (double) cr.n);
                            }
                        }
                        sp_anchor_grid<VT>(dim, cmin.data(), cmax.data(), size.data(), ri_list.data(), 2, anchor);
                        // Degenerate grid (high dimension, see sp_select_cluster_sample()): each far 
                        // node also gives randomly chosen refined points, in proportion to its size
                        int max_grid_size = 0;
                        for (int k = 0; k < dim; k++) if (ri_list[k] > max_grid_size) max_grid_size = ri_list[k];
                        const int fill = (max_grid_size <= 1);
                        std::vector<std::vector<int>> group(n_far);
                        dist2.resize(n_far);
                        for (int a = 0; a < anchor.n; a++)
                            group[sp_nearest_point<VT>(dim, anchor.x.data(), anchor.n, a, centers.x.data(), n_far, n_far, dist2.data())].push_back(a);
                        for (int f = 0; f < n_far; f++)
                        {
                            const sp_ptset<VT> &cr = clu_refine[far[f]];
                            if (cr.n == 0) continue;
                            dist2.resize(cr.n);
                            sample_idx.clear();
                            if (group[f].empty())
                            {
                                // No anchor for this far node: take its point nearest to its center
                                sample_idx.push_back(sp_nearest_point<VT>(dim, centers.x.data(), n_far, f, cr.x.data(), cr.n, cr.n, dist2.data()));
                            } else {
                                std::vector<char> flag(cr.n, 0);
                                for (int a : group[f])
                                    flag[sp_nearest_point<VT>(dim, anchor.x.data(), anchor.n, a, cr.x.data(), cr.n, cr.n, dist2.data())] = 1;
                                for (int i = 0; i < cr.n; i++) if (flag[i]) sample_idx.push_back(i);
                            }
                            if (fill)
                            {
                                int quota = (int) std::ceil((double) max_anchor * (double) cr.n / (double) nFp);
                                if (quota > cr.n) quota = cr.n;
                                std::vector<char> flag(cr.n, 0);
                                for (int i : sample_idx) flag[i] = 1;
                                int n_sel = (int) sample_idx.size();
                                std::mt19937 gen(19241112u + (unsigned) far[f]);
                                std::vector<int> perm(cr.n);
                                for (int i = 0; i < cr.n; i++) perm[i] = i;
                                std::shuffle(perm.begin(), perm.end(), gen);
                                for (int i = 0; i < cr.n && n_sel < quota; i++)
                                {
                                    if (flag[perm[i]]) continue;
                                    flag[perm[i]] = 1;
                                    n_sel++;
                                }
                                sample_idx.clear();
                                for (int i = 0; i < cr.n; i++) if (flag[i]) sample_idx.push_back(i);
                            }
                            sp_ptset<VT> sel = cr;
                            sp_select_points<VT>(sel, dim, sample_idx);
                            sp_append_points<VT>(yF, dim, sel);
                        }
                    }

                    // (3) Add them to the sample points passed by the parent, and thin the result
                    sp_append_points<VT>(sp, dim, yF);
                    if (sp.n > 0)
                    {
                        sp_enbox<VT>(dim, sp.x.data(), sp.n, sp.n, cmin.data(), cmax.data(), size.data());
                        int ri_npt = sp_proportional_int_decompose<VT>(dim, r_down, size.data(), ri_list.data(), max_anchor);
                        if (ri_npt < sp.n)
                        {
                            sp_select_cluster_sample<VT>(dim, sp, ri_list.data(), 2, sample_idx, max_anchor);
                            sp_select_points<VT>(sp, dim, sample_idx);
                        }
                    }
                }

                // (4) Pass the sample points to the children. This is also done when the
                //     node has no far node at its own level: the far field of its ancestors
                //     is also the far field of its children.
                const int *node_children = children + (size_t) node * max_child;
                for (int k = 0; k < n_children[node]; k++) sample_pt[node_children[k]] = sp;
            }
        }
    }
    free(node_far);

    // 5. Return the sample points
    h2m_2dbuf_p *sp_out = (h2m_2dbuf_p *) malloc(sizeof(h2m_2dbuf_p) * n_node);
    long long sum_sp = 0;
    int max_sp = 0, n_sp_node = 0;
    for (int i = 0; i < n_node; i++)
    {
        const sp_ptset<VT> &sp = sample_pt[i];
        h2m_2dbuf_init(sp_out + i, sizeof(VT), sp.n, dim);
        if (sp.n > 0) memcpy(sp_out[i]->data, sp.x.data(), sizeof(VT) * (size_t) sp.n * dim);
        sp_out[i]->nrow = sp.n;
        sp_out[i]->ncol = dim;
        if (sp.n > 0)
        {
            sum_sp += sp.n;
            n_sp_node++;
            if (sp.n > max_sp) max_sp = sp.n;
        }
    }
    *sample_points = sp_out;
    if (opt.verbose)
    {
        printf("[H2 sample points] anchor grid parameter r = %d (up) / %d (down), at most %d anchors per grid\n", r_up, r_down, max_anchor);
        printf("[H2 sample points]   %d nodes with sample points, max / avg = %d / %.0f points\n",
               n_sp_node, max_sp, (n_sp_node > 0) ? (double) sum_sp / n_sp_node : 0.0);
        fflush(stdout);
    }
}

// Select the sample points of each node of a partitioning tree
void h2m_tree_sample_points(
    octree_p octree, const int n_krnl, krnl_func *krnls, const void *param,
    const void *reltol, h2m_2dbuf_p **sample_points
)
{
    if (octree->val_type == VAL_TYPE_DOUBLE)
        h2m_tree_sample_points<double>(octree, n_krnl, krnls, param, *((const double *) reltol), sample_points);
    if (octree->val_type == VAL_TYPE_FLOAT)
        h2m_tree_sample_points<float> (octree, n_krnl, krnls, param, *((const float *)  reltol), sample_points);
}
