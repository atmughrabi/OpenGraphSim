// -----------------------------------------------------------------------------
//
//      "00_AccelGraph"
//
// -----------------------------------------------------------------------------
// Copyright (c) 2014-2019 All rights reserved
// -----------------------------------------------------------------------------
// Author : Abdullah Mughrabi
// Email  : atmughra@ncsu.edu||atmughrabi@gmail.com
// File   : test_accel-graph.c
// Create : 2019-07-29 16:52:00
// Revise : 2019-09-28 15:36:29
// Editor : Abdullah Mughrabi
// -----------------------------------------------------------------------------
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <argp.h>
#include <stdbool.h>
#include <omp.h>
#include <assert.h>

#include "graphStats.h"
#include "edgeList.h"
#include "myMalloc.h"

#include "graphCSR.h"
#include "graphAdjLinkedList.h"
#include "graphAdjArrayList.h"
#include "graphGrid.h"

#include "mt19937.h"
#include "graphConfig.h"
#include "timer.h"
#include "graphRun.h"

#include "BFS.h"
#include "DFS.h"
#include "pageRank.h"
#include "incrementalAggregation.h"
#include "bellmanFord.h"
#include "SSSP.h"
#include "connectedComponents.h"
#include "triangleCount.h"

#ifdef CACHE_HARNESS
#include "cache.h"
#endif

#include "graphTest.h"
#define GRAPH_NUM 5
#define MODE_NUM 3

int
main (int argc, char **argv)
{
    char graph_dir[1024];
    char label_dir[1024];
    char unified_perf_file[1024];
    char express_perf_file[1024];
    char grasp_perf_file[1024];

    char *benchmarks_graphs[GRAPH_NUM] =
    {
        "LAW-amazon-2008",
        "LAW-cnr-2000",
        "LAW-dblp-2010",
        "LAW-enron",
        "SNAP-web-Google"
    };

    char *benchmarks_dir[GRAPH_NUM] =
    {
        "../01_test_graphs/LAW/LAW-amazon-2008",
        "../01_test_graphs/LAW/LAW-cnr-2000",
        "../01_test_graphs/LAW/LAW-dblp-2010",
        "../01_test_graphs/LAW/LAW-enron",
        "../01_test_graphs/SNAP/SNAP-web-Google"
    };

    // char *benchmarks_graphs[GRAPH_NUM] =
    // {
    //     "LAW-in-2004",
    //     "LAW-it-2004",
    //     "LAW-uk-2002",
    //     "LAW-uk-2005",
    //     "LAW-webbase-2001",
    //     "KONECT-wikipedia_link_en",
    //     "GAP-road",
    //     "GAP-twitter",
    //     "GONG-gplus",
    //     "SNAP-cit-Patents",
    //     "SNAP-com-Orkut",
    //     "SNAP-soc-LiveJournal1",
    //     "SNAP-soc-Pokec",
    //     "SNAP-web-Google"
    // };

    // char *benchmarks_dir[GRAPH_NUM] =
    // {
    //     "../../01_GraphDatasets/LAW/LAW-in-2004",
    //     "../../01_GraphDatasets/LAW/LAW-it-2004",
    //     "../../01_GraphDatasets/LAW/LAW-uk-2002",
    //     "../../01_GraphDatasets/LAW/LAW-uk-2005",
    //     "../../01_GraphDatasets/LAW/LAW-webbase-2001",
    //     "../../01_GraphDatasets/KONECT/KONECT-wikipedia_link_en",
    //     "../../01_GraphDatasets/GAP/GAP-road",
    //     "../../01_GraphDatasets/GAP/GAP-twitter",
    //     "../../01_GraphDatasets/GONG/GONG-gplus",
    //     "../../01_GraphDatasets/SNAP/SNAP-cit-Patents",
    //     "../../01_GraphDatasets/SNAP/SNAP-com-Orkut",
    //     "../../01_GraphDatasets/SNAP/SNAP-soc-LiveJournal1",
    //     "../../01_GraphDatasets/SNAP/SNAP-soc-Pokec",
    //     "../../01_GraphDatasets/SNAP/SNAP-web-Google"
    // };

    char *reorder_labels[MODE_NUM] =
    {
        "NO.labels",
        "graph_Rabbit.labels",
        "graph_Gorder.labels"
    };


    uint32_t lmode[MODE_NUM] =
    {
        0,
        11,
        11
    };

    uint32_t lmode_l2[MODE_NUM] =
    {
        0,
        4,
        0
    };

    uint32_t mmode[MODE_NUM] =
    {
        0,
        0,
        1
    };


    struct Arguments arguments;
    /* Default values. */

    arguments.wflag = 0;
    arguments.xflag = 0;
    arguments.sflag = 0;
    arguments.dflag = 1;

    arguments.iterations = 200;
    arguments.trials = 100;
    arguments.epsilon = 0.0001;
    arguments.source = 5319;
    arguments.algorithm = 1;
    arguments.datastructure = 0;
    arguments.pushpull = 0;
    arguments.sort = 1;

    arguments.symmetric = 0;
    arguments.weighted = 0;
    arguments.delta = 1;

    arguments.pre_numThreads  = omp_get_max_threads();
    arguments.algo_numThreads = omp_get_max_threads();
    arguments.ker_numThreads = 1;

    arguments.fnameb = "../01_test_graphs/LAW/LAW-enron/graph.bin";
    arguments.fnamel = "../01_test_graphs/LAW/LAW-enron/graph_Gorder.labels";
    arguments.fnameb_format = 1;
    arguments.convert_format = 1;
    arguments.iterations = 1;
    arguments.trials = 1; // random number of trials
    initializeMersenneState (&(arguments.mt19937var), 27491095);

    arguments.lmode = 0;
    arguments.lmode_l2 = 0;
    arguments.lmode_l3 = 0;
    arguments.mmode = 0;

#ifdef CACHE_HARNESS_META
    arguments.l1_size   = L1_SIZE;
    arguments.l1_assoc  = L1_ASSOC;
    arguments.blocksize = BLOCKSIZE;
    arguments.policey   = LRU_POLICY;
#endif

    void *graph = NULL;

    uint32_t i = 0;
    uint32_t j = 0;
    uint32_t k = 0;
    void *ref_data;

    for ( i = 0; i < GRAPH_NUM; ++i)
    {
        printf("Begin tests for %s\n", benchmarks_dir[i]);
        for (j = 0; j < MODE_NUM; ++j)
        {
            for (k = 0; k < MODE_NUM; ++k)
            {
                sprintf(unified_perf_file, "%s/%s_algo%u.unified.%s", "./cache-results", benchmarks_graphs[i], arguments.algorithm, "perf");
                sprintf(express_perf_file, "%s/%s_algo%u.express.%s", "./cache-results", benchmarks_graphs[i], arguments.algorithm, "perf");
                sprintf(grasp_perf_file, "%s/%s_algo%u.grasp.%s", "./cache-results", benchmarks_graphs[i], arguments.algorithm, "perf");

                sprintf (graph_dir, "%s/%s", benchmarks_dir[i], "graph.bin");
                arguments.fnameb = graph_dir;
                sprintf (label_dir, "%s/%s", benchmarks_dir[i], reorder_labels[j]);
                arguments.fnamel = label_dir;

                arguments.lmode = 10; // base is random order
                arguments.lmode_l2 =  lmode[j];
                arguments.lmode_l3 = lmode_l2[k];
                arguments.mmode = mmode[k];

                printf("\n%u %u %u %s %s\n", arguments.lmode, arguments.lmode_l2, arguments.mmode, reorder_labels[j], unified_perf_file);

                graph = generateGraphDataStructure(&arguments);

                arguments.source = generateRandomRootGeneral(&arguments, graph); // random root each trial
                ref_data = runGraphAlgorithmsTest(&arguments, graph); // ref stats should mach oother algo
                struct PageRankStats *ref_stats_tmp = (struct PageRankStats * )ref_data;

                printStatsDoubleTaggedCacheToFile(ref_stats_tmp->cache, unified_perf_file);
                printStatsAccelGraphCachetoFile(ref_stats_tmp->cache->accel_graph_mask, express_perf_file);
                printStatsAccelGraphCachetoFile(ref_stats_tmp->cache->accel_graph_grasp, grasp_perf_file);
                freeGraphStatsGeneral(ref_data, arguments.algorithm);

                freeGraphDataStructure(graph, arguments.datastructure);
            }
        }
    }

    exit (0);
}





