#ifndef __H2MAT_BUILD_H__
#define __H2MAT_BUILD_H__

#include "../kernels/kernels.h"
#include "octree.h"
#include "h2mat_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Build an H2 matrix for K(X, X, l) with a given point partitioning 
// octree and a set of proxy points
// Input parameter:
//   octree : Point partitioning octree
//   pp     : Size octree->n_level, proxy points for each level
//   krnl   : Kernel matrix evaluation function
//   param  : Kernel matrix parameters [dim, l]
//   reltol : H2 compression relative error tolerance
// Output parameter:
//   *h2mat : Constrcuted h2mat struct
// Notes: 
//   1. DO NOT free octree and param before calling h2mat_free()
//   2. octree->val_type specifies the data type of param, reltol, and h2mat->V_mats
void h2mat_build_with_proxy_points(
    octree_p octree, h2m_2dbuf_p *pp, krnl_func krnl,
    void *param, const void *reltol, h2mat_p *h2mat
);

// Build an H2 matrix for K(X, X, l) with a given point partitioning tree and the 
// sample points of each node (see h2mat_sample_points.h). Works for any tree type.
// Input parameter:
//   octree : Point partitioning tree
//   sp     : Size octree->n_node, sample points of each node
// Other parameters and notes are the same as h2mat_build_with_proxy_points()
void h2mat_build_with_sample_points(
    octree_p octree, h2m_2dbuf_p *sp, krnl_func krnl,
    void *param, const void *reltol, h2mat_p *h2mat
);

// Compute the H2 matrix reduced near / far (inadmissible / admissible) lists of each node
// Input parameter:
//   octree : Point partitioning tree
// Output parameters:
//   node_n_near : Size octree->n_node, number of near nodes of each node
//   node_near   : Size octree->n_node * octree->n_node, each row stores the near nodes of a node
//   node_n_far  : Size octree->n_node, number of far nodes of each node
//   node_far    : Size octree->n_node * octree->n_node, each row stores the far nodes of a node
//   min_far_lvl : The minimum level of far pairs
void h2m_calc_near_far_lists(
    octree_p octree, int *node_n_near, int *node_near, 
    int *node_n_far, int *node_far, int *min_far_lvl
);

// Print the main characteristics of an H2 matrix
void h2mat_print_info(h2mat_p h2mat, const char *name);

#ifdef __cplusplus
}
#endif

#endif
