#ifndef REORDER_H
#define REORDER_H

#include <stdint.h>
#include "edgeList.h"
#include "graphCSR.h"

#include "cache.h"

struct EdgeList *relabelEdgeListFromFile(struct EdgeList *edgeList, const char *fnameb, uint32_t size);
void writeLabelsToFile(const char *fnameb, uint32_t *labels, uint32_t size);
struct EdgeList *relabelEdgeList(struct EdgeList *edgeList, uint32_t *labels);
struct EdgeList *reorderGraphProcess(struct EdgeList *edgeList, struct Arguments *arguments);
uint32_t *reorderGraphGenerateInOutDegrees(uint32_t *degrees, struct EdgeList *edgeList, uint32_t lmode);

// ********************************************************************************************
// ***************                  AccelGraph label-Masking                     **************
// ********************************************************************************************

#define ACCELGRAPH_CACHE PSL_L1_SIZE + WARM_L1_SIZE + HOT_L1_SIZE

#define VERTEX_VALUE_MASK_U32 0xC000
#define VERTEX_CACHE_MASK_U32 0x3FFF

#define VERTEX_VALUE_HOT_U32       0xC000
#define VERTEX_CACHE_WARM_U32      0x4000
#define VERTEX_VALUE_LUKEWARM_U32  0x1000
#define VERTEX_CACHE_COLD_U32      0x0000






// ********************************************************************************************
// ***************                  Degree relabel                               **************
// ********************************************************************************************
struct EdgeList *reorderGraphProcessDegree( uint32_t sort, struct EdgeList *edgeList, uint32_t lmode);
struct EdgeList *reorderGraphListDegree(struct EdgeList *edgeList, uint32_t *degrees, uint32_t lmode);


// ********************************************************************************************
// ***************                  DBG relabel                                  **************
// ********************************************************************************************
struct EdgeList *reorderGraphProcessDBG( uint32_t sort, struct EdgeList *edgeList, uint32_t lmode);
struct EdgeList *reorderGraphListDBG(struct EdgeList *edgeList, uint32_t *degrees, uint32_t *thresholds, uint32_t num_buckets, uint32_t lmode);


// ********************************************************************************************
// ***************                  HUBSort relabel                              **************
// ********************************************************************************************
struct EdgeList *reorderGraphProcessHUBSort( uint32_t sort, struct EdgeList *edgeList, uint32_t lmode);
struct EdgeList *reorderGraphListHUBSort(struct EdgeList *edgeList, uint32_t *degrees, uint32_t *thresholds, uint32_t num_buckets, uint32_t lmode);


// ********************************************************************************************
// ***************                  HUBCluster relabel                           **************
// ********************************************************************************************
struct EdgeList *reorderGraphProcessHUBCluster( uint32_t sort, struct EdgeList *edgeList, uint32_t lmode);
struct EdgeList *reorderGraphListHUBCluster(struct EdgeList *edgeList, uint32_t *degrees, uint32_t *thresholds, uint32_t num_buckets, uint32_t lmode);


void radixSortCountSortEdges(uint32_t **pageRanksFP, uint32_t **pageRanksFPTemp, uint32_t **labels, uint32_t **labelsTemp, uint32_t radix, uint32_t buckets, uint32_t *buckets_count, uint32_t num_vertices);
uint32_t *radixSortEdgesByDegree (uint32_t *degrees, uint32_t *labels, uint32_t num_vertices);
uint32_t *radixSortEdgesByPageRank (float *pageRanks, uint32_t *labels, uint32_t num_vertices);
#endif