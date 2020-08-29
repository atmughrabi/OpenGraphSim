#ifndef CACHE_H
#define CACHE_H

#include <stdint.h>

// Policy TYPES
#define LRU_POLICY     0
#define LFU_POLICY     1
#define GRASP_POLICY   2
#define SRRIP_POLICY   3
#define PIN_POLICY     4
#define PLRU_POLICY    5
#define GRASPXP_POLICY 6
#define MASK_POLICY    7


//CAPI PSL CACHE default CONFIGS

// #define PSL_L1_SIZE   262144
#define PSL_L1_SIZE    262144
#define PSL_L1_ASSOC  8
#define PSL_POLICY    PLRU_POLICY
#define PSL_BLOCKSIZE 128

#define WARM_L1_SIZE   131072
#define WARM_L1_ASSOC  8
#define WARM_POLICY    PLRU_POLICY
#define WARM_BLOCKSIZE 4

#define HOT_L1_SIZE   131072
#define HOT_L1_ASSOC  8
#define HOT_POLICY    PLRU_POLICY
#define HOT_BLOCKSIZE 4


// General cache configuration
//GRASP/Ref_cache default configs
// GRASP EXPRESS (GRASP-XP)
// CHOOSE global Policys
// #define POLICY LRU_POLICY
// #define POLICY SRRIP_POLICY
// #define POLICY LFU_POLICY
#define POLICY GRASP_POLICY
// #define POLICY PIN_POLICY
// #define POLICY PLRU_POLICY
// #define POLICY GRASPXP_POLICY
// #define POLICY MASK_POLICY

#ifndef POLICY1
#define POLICY1 PLRU_POLICY
#define POLICY2 SRRIP_POLICY
#define POLICY3 GRASP_POLICY
#define POLICY4 MASK_POLICY
#endif

// #define BLOCKSIZE   64
// #define L1_SIZE     1048576
// #define L1_ASSOC    16

// #define BLOCKSIZE   64
// #define L1_SIZE     1048576
// #define L1_ASSOC    16

// #define BLOCKSIZE   128
// #define L1_SIZE     262144
// #define L1_ASSOC    8

#define BLOCKSIZE   128
#define L1_SIZE     524288
#define L1_ASSOC    8

// #define BLOCKSIZE   128
// #define L1_SIZE     262144 + 32768 + 32768
// #define L1_ASSOC    8

// #define BLOCKSIZE   64
// #define L1_SIZE     32768
// #define L1_ASSOC    8

// Cache states Constants
#define INVALID 0
#define VALID 1
#define DIRTY 2

// LFU Policy Constants
#define FREQ_BITS 8
#define FREQ_MAX (uint8_t)((uint32_t)(1 << FREQ_BITS) - 1)

// GRASP Policy Constants RRIP (re-refernece insertion prediction)
#define NUM_BITS_RRIP 3

#define DEFAULT_INSERT_RRPV ((1 << NUM_BITS_RRIP) - 1)
#define COLD_INSERT_RRPV DEFAULT_INSERT_RRPV
#define WARM_INSERT_RRPV DEFAULT_INSERT_RRPV - 1
#define HOT_INSERT_RRPV  1
#define HOT_HIT_RRPV  0
#define RRPV_INIT DEFAULT_INSERT_RRPV

// GRASP-XP Policy Constants RRIP (re-refernece insertion prediction)
#define NUM_BITS_XPRRIP 8

#define DEFAULT_INSERT_XPRRPV ((1 << NUM_BITS_XPRRIP) - 1)
#define COLD_INSERT_XPRRPV  DEFAULT_INSERT_XPRRPV - 1
#define HOT_INSERT_XPRRPV 1
#define XPRRPV_INIT DEFAULT_INSERT_XPRRPV

// SRRIP Policy Constants RRIP (re-refernece insertion prediction)
#define NUM_BITS_SRRIP 8

#define SRRPV_INIT ((1 << NUM_BITS_SRRIP) - 1)
#define DEFAULT_INSERT_SRRPV (SRRPV_INIT - 1)
#define HIT_SRRPV  0


struct PropertyMetaData
{
    uint64_t base_address;
    uint32_t size;
    uint32_t data_type_size;
};

struct PropertyRegion
{
    uint64_t base_address;
    uint32_t size;
    uint32_t data_type_size;
    uint64_t lower_bound;
    uint64_t hot_bound;
    uint64_t warm_bound;
    uint64_t upper_bound;
    uint32_t fraction;
};

struct CacheLine
{
    uint32_t idx;
    uint64_t addr;
    uint64_t tag;
    uint8_t Flags; // 0:invalid, 1:valid, 2:dirty
    uint64_t seq;  // LRU   POLICY 0
    uint8_t freq;  // LFU   POLICY 1
    uint8_t RRPV;  // GRASP POLICY 2
    uint8_t SRRPV; // SRRPV POLICY 3
    uint8_t PIN;   // PIN   POLICY 4
    uint8_t PLRU;  // PLRU  POLICY 5
    uint8_t XPRRPV;// GRASP POLICY 6
};

struct Cache
{
    uint64_t size, lineSize, assoc, sets, log2Sets, log2Blk, tagMask, numLines, evictions;
    uint64_t reads, readMisses, readsPrefetch, readMissesPrefetch, writes, writeMisses, writeBacks;
    uint64_t access_counter;
    uint32_t policy;

    struct CacheLine **cacheLines;

    //GRASP for graph performance on the cache
    struct PropertyRegion *propertyRegions;
    uint32_t numPropertyRegions;

    uint64_t currentCycle;
    uint64_t currentCycle_cache;
    uint64_t currentCycle_preftcher;

    //counters for graph performance on the cache
    uint64_t *verticesMiss;
    uint64_t *verticesHit;
    uint64_t *vertices_base_reuse;
    uint64_t *vertices_total_reuse;
    uint64_t *vertices_accesses;
    uint32_t  numVertices;

    uint32_t  num_buckets;
    uint64_t *thresholds;
    uint64_t *thresholds_count;
    uint64_t *thresholds_totalDegrees;
    uint64_t *thresholds_avgDegrees;
    uint64_t **regions_avgDegrees;
};


struct AccelGraphCache
{
    struct Cache *cold_cache;// psl_cache
    struct Cache *warm_cache;// warm_cache
    struct Cache *hot_cache; // hot_cache
};

struct DoubleTaggedCache
{
    struct AccelGraphCache *accel_graph_mask;// psl_cache
    struct AccelGraphCache *accel_graph_grasp;// psl_cache
    struct Cache *ref_cache; // PLRU
    struct Cache *ref2_cache; // psl_cache
    struct Cache *ref3_cache; // psl_cache
    struct Cache *ref4_cache; // psl_cache
};

///cacheline helper functions
void initCache(struct Cache *cache, int s, int a, int b, int p);
void initCacheLine(struct CacheLine *cacheLine);

uint64_t getTag(struct CacheLine *cacheLine);
uint64_t getAddr(struct CacheLine *cacheLine);
uint8_t getFlags(struct CacheLine *cacheLine);
uint64_t getSeq(struct CacheLine *cacheLine);
uint8_t getFreq(struct CacheLine *cacheLine);
uint8_t getRRPV(struct CacheLine *cacheLine);
uint8_t getSRRPV(struct CacheLine *cacheLine);
uint8_t getPIN(struct CacheLine *cacheLine);
uint8_t getPLRU(struct CacheLine *cacheLine);
uint8_t getXPRRPV(struct CacheLine *cacheLine);

void setFlags(struct CacheLine *cacheLine, uint8_t flags);
void setTag(struct CacheLine *cacheLine, uint64_t a);
void setAddr(struct CacheLine *cacheLine, uint64_t addr);
void setSeq(struct CacheLine *cacheLine, uint64_t Seq);
void setFreq(struct CacheLine *cacheLine, uint8_t freq);
void setRRPV(struct CacheLine *cacheLine, uint8_t RRPV);
void setSRRPV(struct CacheLine *cacheLine, uint8_t SRRPV);
void setPIN(struct CacheLine *cacheLine, uint8_t PIN);
void setPLRU(struct CacheLine *cacheLine, uint8_t PLRU);
void setXPRRPV(struct CacheLine *cacheLine, uint8_t XPRRPV);

void invalidate(struct CacheLine *cacheLine);
uint32_t isValid(struct CacheLine *cacheLine);

uint64_t calcTag(struct Cache *cache, uint64_t addr);
uint64_t calcIndex(struct Cache *cache, uint64_t addr);
uint64_t calcAddr4Tag(struct Cache *cache, uint64_t tag);

uint64_t getRM(struct Cache *cache);
uint64_t getWM(struct Cache *cache);
uint64_t getReads(struct Cache *cache);
uint64_t getWrites(struct Cache *cache);
uint64_t getWB(struct Cache *cache);
uint64_t getEVC(struct Cache *cache);
uint64_t getRMPrefetch(struct Cache *cache);
uint64_t getReadsPrefetch(struct Cache *cache);

void Prefetch(struct Cache *cache, uint64_t addr, unsigned char op, uint32_t node, uint32_t mask);
uint32_t checkInCache(struct Cache *cache, uint64_t addr);

// ********************************************************************************************
// ***************         Cacheline lookups                                     **************
// ********************************************************************************************

void writeBack(struct Cache *cache, uint64_t addr);
void Access(struct Cache *cache, uint64_t addr, unsigned char op, uint32_t node, uint32_t mask);
struct CacheLine *findLine(struct Cache *cache, uint64_t addr);
struct CacheLine *fillLine(struct Cache *cache, uint64_t addr, uint32_t mask);
struct CacheLine *findLineToReplace(struct Cache *cache, uint64_t addr, uint32_t mask);

// ********************************************************************************************
// ***************         VICTIM EVICTION POLICIES                              **************
// ********************************************************************************************

struct CacheLine *getVictimPolicy(struct Cache *cache, uint64_t addr);
struct CacheLine *getVictimLRU(struct Cache *cache, uint64_t addr);
struct CacheLine *getVictimLFU(struct Cache *cache, uint64_t addr);
struct CacheLine *getVictimGRASP(struct Cache *cache, uint64_t addr);
struct CacheLine *getVictimSRRIP(struct Cache *cache, uint64_t addr);
struct CacheLine *getVictimPIN(struct Cache *cache, uint64_t addr);
uint8_t getVictimPINBypass(struct Cache *cache, uint64_t addr);
struct CacheLine *getVictimPLRU(struct Cache *cache, uint64_t addr);
struct CacheLine *getVictimGRASPXP(struct Cache *cache, uint64_t addr);
struct CacheLine *getVictimMASK(struct Cache *cache, uint64_t addr);

// ********************************************************************************************
// ***************         VICTIM PEEK POLICIES                                  **************
// ********************************************************************************************
struct CacheLine *peekVictimPolicy(struct Cache *cache, uint64_t addr);
struct CacheLine *peekVictimLRU(struct Cache *cache, uint64_t addr);
struct CacheLine *peekVictimLFU(struct Cache *cache, uint64_t addr);
struct CacheLine *peekVictimGRASP(struct Cache *cache, uint64_t addr);
struct CacheLine *peekVictimSRRIP(struct Cache *cache, uint64_t addr);
struct CacheLine *peekVictimPIN(struct Cache *cache, uint64_t addr);
struct CacheLine *peekVictimPLRU(struct Cache *cache, uint64_t addr);
struct CacheLine *peekVictimGRASPXP(struct Cache *cache, uint64_t addr);
struct CacheLine *peekVictimMASK(struct Cache *cache, uint64_t addr);

// ********************************************************************************************
// ***************         INSERTION POLICIES                                    **************
// ********************************************************************************************

void updateInsertionPolicy(struct Cache *cache, struct CacheLine *line, uint32_t mask);
void updateInsertLRU(struct Cache *cache, struct CacheLine *line);
void updateInsertLFU(struct Cache *cache, struct CacheLine *line);
void updateInsertGRASP(struct Cache *cache, struct CacheLine *line);
void updateInsertSRRIP(struct Cache *cache, struct CacheLine *line);
void updateInsertPIN(struct Cache *cache, struct CacheLine *line);
void updateInsertPLRU(struct Cache *cache, struct CacheLine *line);
void updateInsertGRASPXP(struct Cache *cache, struct CacheLine *line);
void updateInsertMASK(struct Cache *cache, struct CacheLine *line, uint32_t mask);

// ********************************************************************************************
// ***************         PROMOTION POLICIES                                    **************
// ********************************************************************************************

void updatePromotionPolicy(struct Cache *cache, struct CacheLine *line, uint32_t mask);
void updatePromoteLRU(struct Cache *cache, struct CacheLine *line);
void updatePromoteLFU(struct Cache *cache, struct CacheLine *line);
void updatePromoteGRASP(struct Cache *cache, struct CacheLine *line);
void updatePromoteSRRIP(struct Cache *cache, struct CacheLine *line);
void updatePromotePIN(struct Cache *cache, struct CacheLine *line);
void updatePromotePLRU(struct Cache *cache, struct CacheLine *line);
void updatePromoteGRASPXP(struct Cache *cache, struct CacheLine *line);
void updatePromoteMASK(struct Cache *cache, struct CacheLine *line, uint32_t mask);

// ********************************************************************************************
// ***************         AGING POLICIES                                        **************
// ********************************************************************************************

void updateAgingPolicy(struct Cache *cache);
void updateAgeLRU(struct Cache *cache);
void updateAgeLFU(struct Cache *cache);
void updateAgeGRASP(struct Cache *cache);
void updateAgeSRRIP(struct Cache *cache);
void updateAgePIN(struct Cache *cache);
void updateAgePLRU(struct Cache *cache);
void updateAgeGRASPXP(struct Cache *cache);

// ********************************************************************************************
// ***************               Cache Orignzation                                **************
// ********************************************************************************************

struct DoubleTaggedCache *newDoubleTaggedCache(uint32_t l1_size, uint32_t l1_assoc, uint32_t blocksize, uint32_t num_vertices,  uint32_t policy, uint32_t numPropertyReg);
void initDoubleTaggedCacheRegion(struct DoubleTaggedCache *cache, struct PropertyMetaData *propertyMetaData);
void freeDoubleTaggedCache(struct DoubleTaggedCache *cache);

// ********************************************************************************************
// ***************              AccelGraph Cache configuration                   **************
// ********************************************************************************************

struct AccelGraphCache *newAccelGraphCache(uint32_t l1_size, uint32_t l1_assoc, uint32_t blocksize, uint32_t num_vertices,  uint32_t policy, uint32_t numPropertyReg);
void initAccelGraphCacheRegion(struct AccelGraphCache *cache, struct PropertyMetaData *propertyMetaData);
void freeAccelGraphCache(struct AccelGraphCache *cache);

// ********************************************************************************************
// ***************               ACCElGraph Policy                               **************
// ********************************************************************************************
void AccessDoubleTaggedCacheUInt32(struct DoubleTaggedCache *cache, uint64_t addr, unsigned char op, uint32_t node, uint32_t mask);
void AccessAccelGraphGRASP(struct AccelGraphCache *accel_graph, uint64_t addr, unsigned char op, uint32_t node, uint32_t mask);
void AccessAccelGraphExpress(struct AccelGraphCache *accel_graph, uint64_t addr, unsigned char op, uint32_t node, uint32_t mask);
// ********************************************************************************************
// ***************               GRASP-XP Policy                                 **************
// ********************************************************************************************
void setCacheThresholdDegreeAvg(struct Cache *cache, uint32_t  *degrees);
void setAccelGraphCacheThresholdDegreeAvg(struct AccelGraphCache *cache, uint32_t  *degrees);
void setDoubleTaggedCacheThresholdDegreeAvg(struct DoubleTaggedCache *cache, uint32_t  *degrees);
void setCacheRegionDegreeAvg(struct Cache *cache);
uint64_t getCacheRegionAddrGRASPXP(struct Cache *cache, uint64_t addr);
uint64_t getCacheRegionGRASPXP(struct Cache *cache, struct CacheLine *line);
// ********************************************************************************************
// ***************               GRASP Policy                                    **************
// ********************************************************************************************

void initialzeCachePropertyRegions (struct Cache *cache, struct PropertyMetaData *propertyMetaData, uint64_t size);
uint32_t inHotRegion(struct Cache *cache, struct CacheLine *line);
uint32_t inWarmRegion(struct Cache *cache, struct CacheLine *line);
uint32_t inHotRegionAddrGRASP(struct Cache *cache, uint64_t addr);
uint32_t inWarmRegionAddrGRASP(struct Cache *cache, uint64_t addr);

// ********************************************************************************************
// ***************               Stats output                                    **************
// ********************************************************************************************
void online_cache_graph_stats(struct Cache *cache, uint32_t node);
// void printStatsDoubleTaggedCache(struct DoubleTaggedCache *cache);
void printStats(struct Cache *cache);
void printStatsCache(struct Cache *cache);
void printStatsCacheToFile(struct Cache *cache, char *fname_perf);
void printStatsGraphReuse(struct Cache *cache, uint32_t *degrees);
void printStatsGraphCache(struct Cache *cache, uint32_t *in_degree, uint32_t *out_degree);
void printStatsAccelGraphCache(struct AccelGraphCache *cache, uint32_t *in_degree, uint32_t *out_degree);
void printStatsDoubleTaggedCache(struct DoubleTaggedCache *cache, uint32_t *in_degree, uint32_t *out_degree);
void printStatsDoubleTaggedCacheToFile(struct DoubleTaggedCache *cache, char *fname_perf);
void printStatsAccelGraphCachetoFile(struct AccelGraphCache *cache, char *fname_perf);

struct Cache *newCache( uint32_t l1_size, uint32_t l1_assoc, uint32_t blocksize, uint32_t num_vertices, uint32_t policy, uint32_t numPropertyRegions);
void freeCache(struct Cache *cache);


#endif