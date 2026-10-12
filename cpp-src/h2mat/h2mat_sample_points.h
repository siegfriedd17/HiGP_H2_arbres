#ifndef __H2MAT_SAMPLE_POINTS_H__
#define __H2MAT_SAMPLE_POINTS_H__

#include "../common.h"
#include "../kernels/kernels.h"
#include "h2mat_utils.h"
#include "octree.h"

#ifdef __cplusplus
extern "C" {
#endif

// Select the sample points of each node of a partitioning tree. The sample points of 
// a node are real points taken in its far field (the nodes that are admissible with 
// it or with one of its ancestors). Unlike proxy points, they do not assume that the 
// nodes of a level are boxes of the same size: they work with any tree type (box, 
// PCA, 2-means) and any point dimension.
// Input parameters:
//   octree : A constructed partitioning tree
//   n_krnl : Number of kernel functions that will be compressed with these sample points
//   krnls  : Size n_krnl, kernel matrix evaluation functions, only used to calibrate 
//            the number of sample points for the target accuracy
//   param  : Pointer to kernel matrix evaluation function parameter list
//   reltol : H2 compression relative error tolerance
// Output parameters:
//   *sample_points : Array of h2m_2dbuf_p, size octree->n_node, i-th element stores the 
//                    sample points of node i, size sp[i]->nrow * pt_dim, col-major. 
//                    sp[i]->nrow == 0 if node i has no far field.
void h2m_tree_sample_points(
    octree_p octree, const int n_krnl, krnl_func *krnls, const void *param, 
    const void *reltol, h2m_2dbuf_p **sample_points
);

#ifdef __cplusplus
};
#endif

#endif
