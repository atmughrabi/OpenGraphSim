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

#include "cache.h"

#include "graphTest.h"
#define GRAPH_NUM 4

#define CACHE_CONFIGS 1
#define CACHE_POLICY 6
#define MODE_NUM 3
#define ORDER_CONFIG 6
#define TOTAL_CONFIG (MODE_NUM+MODE_NUM+ORDER_CONFIG)



int
main (int argc, char **argv)
{
    char graph_dir[1024];
    char label_dir[1024];
    char unified_perf_file[1024];
    // char express_perf_file[1024];
    // char grasp_perf_file[1024];

    // uint32_t cache_size[CACHE_CONFIGS] = {32768, 65536, 131072, 262144, 524288,  1048576, 2097152, 4194304, 8388608, 16777216,  33554432,   67108864};
    // uint32_t Associativity[CACHE_CONFIGS] = {4,  4, 4, 8,  8,  16, 16, 16, 16, 32, 32, 32};
    // uint32_t Block_size[CACHE_CONFIGS] = {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};

    // uint32_t cache_size[CACHE_CONFIGS]    = {262144, 524288,  1048576, 2097152, 4194304};
    // uint32_t Associativity[CACHE_CONFIGS] = {8,  8,  16, 16, 16};
    // uint32_t Block_size[CACHE_CONFIGS]    = {128, 128, 128, 128, 128};

    uint32_t cache_size[CACHE_CONFIGS]    = {524288};
    uint32_t Associativity[CACHE_CONFIGS] = {8};
    uint32_t Block_size[CACHE_CONFIGS]    = {128};

    uint32_t policy[CACHE_POLICY]         = {PLRU_POLICY, SRRIP_POLICY, GRASP_POLICY, MASK_POLICY, GRASP_CAPI_POLICY, MASK_CAPI_POLICY};
    float PLRU_stats[GRAPH_NUM][ORDER_CONFIG]       = {0};
    float SSRIP_stats[GRAPH_NUM][ORDER_CONFIG]      = {0};
    float GRASP_stats[GRAPH_NUM][MODE_NUM]          = {0};
    float EXPRESS_stats[GRAPH_NUM][MODE_NUM]        = {0};
    float GRASP_stats_capi[GRAPH_NUM][MODE_NUM]     = {0};
    float EXPRESS_stats_capi[GRAPH_NUM][MODE_NUM]   = {0};
    uint32_t lmode_l2[TOTAL_CONFIG] = {0, 4, 11, 11, 11, 11, 0, 11, 11, 0, 11, 11};
    uint32_t lmode_l3[TOTAL_CONFIG] = {0, 0, 0, 4, 0, 4, 4, 4, 4, 0, 0, 0 };
    uint32_t mmode[TOTAL_CONFIG]    = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1 };

    char *config_labels[TOTAL_CONFIG] =
    {
        "Rand-Order",
        "DBG",
        "Rabbit",
        "Rabbit+DBG",
        "Gorder",
        "Gorder+DBG",
        "DBG",
        "Rabbit+DBG",
        "Gorder+DBG",
        "MASK",
        "Rabbit+MASK",
        "Gorder+MASK"
    };

    char *reorder_labels[TOTAL_CONFIG] =
    {
        "NO.labels",
        "NO.labels",
        "graph_Rabbit.rand.labels",
        "graph_Rabbit.rand.labels",
        "graph_Gorder.rand.labels",
        "graph_Gorder.rand.labels",
        "NO.labels",
        "graph_Rabbit.rand.labels",
        "graph_Gorder.rand.labels",
        "NO.labels",
        "graph_Rabbit.rand.labels",
        "graph_Gorder.rand.labels"
    };


    char *benchmarks_graphs[GRAPH_NUM] =
    {
        "LAW-amazon-2008",
        "LAW-cnr-2000",
        "LAW-dblp-2010",
        "LAW-enron"
    };

    char *benchmarks_dir[GRAPH_NUM] =
    {
        "../01_test_graphs/LAW/LAW-amazon-2008",
        "../01_test_graphs/LAW/LAW-cnr-2000",
        "../01_test_graphs/LAW/LAW-dblp-2010",
        "../01_test_graphs/LAW/LAW-enron"
    };

    // char *benchmarks_graphs[GRAPH_NUM] =
    // {
    //     "GAP-road",
    //     "GAP-twitter",
    //     "GONG-gplus",
    //     "KONECT-wikipedia_link_en",
    //     "LAW-in-2004",
    //     "LAW-it-2004",
    //     "LAW-uk-2002",
    //     "LAW-uk-2005",
    //     "LAW-webbase-2001",
    //     "SNAP-cit-Patents",
    //     "SNAP-com-Orkut",
    //     "SNAP-soc-LiveJournal1",
    //     "SNAP-soc-Pokec",
    //     "SNAP-web-Google"
    // };

    // char *benchmarks_dir[GRAPH_NUM] =
    // {
    //     "../../01_GraphDatasets/GAP/GAP-road",
    //     "../../01_GraphDatasets/GAP/GAP-twitter",
    //     "../../01_GraphDatasets/GONG/GONG-gplus",
    //     "../../01_GraphDatasets/KONECT/KONECT-wikipedia_link_en",
    //     "../../01_GraphDatasets/LAW/LAW-in-2004",
    //     "../../01_GraphDatasets/LAW/LAW-it-2004",
    //     "../../01_GraphDatasets/LAW/LAW-uk-2002",
    //     "../../01_GraphDatasets/LAW/LAW-uk-2005",
    //     "../../01_GraphDatasets/LAW/LAW-webbase-2001",
    //     "../../01_GraphDatasets/SNAP/SNAP-cit-Patents",
    //     "../../01_GraphDatasets/SNAP/SNAP-com-Orkut",
    //     "../../01_GraphDatasets/SNAP/SNAP-soc-LiveJournal1",
    //     "../../01_GraphDatasets/SNAP/SNAP-soc-Pokec",
    //     "../../01_GraphDatasets/SNAP/SNAP-web-Google"
    // };

    struct Arguments arguments;
    /* Default values. */

    arguments.wflag = 0;
    arguments.xflag = 0;
    arguments.sflag = 0;
    arguments.dflag = 1;

    arguments.iterations = 200;
    arguments.trials = 5;
    arguments.epsilon = 0.0001;
    arguments.source = 5319;

    arguments.datastructure = 0;
    arguments.pushpull = 0;
    arguments.sort = 1;

    arguments.symmetric = 0;
    arguments.weighted = 0;
    arguments.delta = 1;

    arguments.pre_numThreads  = omp_get_max_threads();
    arguments.algo_numThreads = omp_get_max_threads();
    arguments.ker_numThreads = 1;

    arguments.fnameb = "../01_test_graphs/LAW/LAW-enron/graph.rand.bin";
    arguments.fnamel = "../01_test_graphs/LAW/LAW-enron/graph_Gorder.labels";
    arguments.fnameb_format = 1;
    arguments.convert_format = 1;

    arguments.algorithm = 0;
    arguments.iterations = 1;
    arguments.trials = 1; // random number of trials

    initializeMersenneState (&(arguments.mt19937var), 27491095);

    arguments.lmode = 0;
    arguments.lmode_l2 = 0;
    arguments.lmode_l3 = 0;
    arguments.mmode = 0;

    arguments.l1_size   = L1_SIZE;
    arguments.l1_assoc  = L1_ASSOC;
    arguments.blocksize = BLOCKSIZE;
    arguments.policey   = LRU_POLICY;

    void *graph = NULL;

    uint32_t i = 0;
    uint32_t j = 0;
    uint32_t k = 0;
    uint32_t kk = 0;
    uint32_t jj = 0;
    void *ref_data;

    sprintf(unified_perf_file, "%s/results_algo%u_cache.unified.%s", "./cache-results", arguments.algorithm, "perf");

    for( k = 0; k < CACHE_CONFIGS ; k++)
    {
        arguments.l1_size   = cache_size[k];
        arguments.l1_assoc  = Associativity[k];
        arguments.blocksize = Block_size[k];

        for ( i = 0; i < GRAPH_NUM; ++i)
        {
            printf("graph %s\n", benchmarks_dir[i]);

            // sprintf(express_perf_file, "%s/%s_algo%u_cache%u.express.%s", "./cache-results", benchmarks_graphs[i], arguments.algorithm, arguments.l1_size, "perf");
            // sprintf(grasp_perf_file, "%s/%s_algo%u_cache%u.grasp.%s", "./cache-results", benchmarks_graphs[i], arguments.algorithm, arguments.l1_size, "perf");

            arguments.policey   = policy[0];
            for (j = 0; j < ORDER_CONFIG; ++j)
            {
                sprintf (graph_dir, "%s/%s", benchmarks_dir[i], "graph.rand.bin");
                sprintf (label_dir, "%s/%s", benchmarks_dir[i], reorder_labels[j]);
                arguments.lmode = 0; // base is random order
                arguments.lmode_l2 =  lmode_l2[j];
                arguments.lmode_l3 = lmode_l3[j];
                arguments.mmode = mmode[j];
                arguments.fnameb = graph_dir;
                arguments.fnamel = label_dir;
                printf("graph config %5u - %u %u %u - %u %s\n", j, arguments.lmode_l2, arguments.lmode_l3, arguments.mmode, arguments.l1_size / 1024, config_labels[j]);

                graph = generateGraphDataStructure(&arguments);

                initializeMersenneState (&(arguments.mt19937var), 27491095);
                for(jj = 0 ; jj < arguments.trials; jj++)
                {
                    arguments.source = generateRandomRootGeneral(&arguments, graph); // random root each trial
                    ref_data = runGraphAlgorithmsTest(&arguments, graph); // ref stats should mach oother algo
                    PLRU_stats[i][j] += getGraphAlgorithmsTestMissRateRef(ref_data, arguments.algorithm);
                    // printStatsDoubleTaggedCacheToFile(ref_stats_tmp->cache, unified_perf_file);
                    freeGraphStatsGeneral(ref_data, arguments.algorithm);
                }

                freeGraphDataStructure(graph, arguments.datastructure);
            }

            arguments.policey   = policy[1];
            for (j = 0; j < ORDER_CONFIG; ++j)
            {
                sprintf (graph_dir, "%s/%s", benchmarks_dir[i], "graph.rand.bin");
                sprintf (label_dir, "%s/%s", benchmarks_dir[i], reorder_labels[j]);
                arguments.lmode = 0; // base is random order
                arguments.lmode_l2 =  lmode_l2[j];
                arguments.lmode_l3 = lmode_l3[j];
                arguments.mmode = mmode[j];
                arguments.fnameb = graph_dir;
                arguments.fnamel = label_dir;
                printf("graph config %5u - %u %u %u - %u %s\n", j, arguments.lmode_l2, arguments.lmode_l3, arguments.mmode, arguments.l1_size / 1024, config_labels[j]);

                graph = generateGraphDataStructure(&arguments);

                initializeMersenneState (&(arguments.mt19937var), 27491095);
                for(jj = 0 ; jj < arguments.trials; jj++)
                {
                    arguments.source = generateRandomRootGeneral(&arguments, graph); // random root each trial
                    ref_data = runGraphAlgorithmsTest(&arguments, graph); // ref stats should mach oother algo
                    SSRIP_stats[i][j] += getGraphAlgorithmsTestMissRateRef(ref_data, arguments.algorithm);
                    // printStatsDoubleTaggedCacheToFile(ref_stats_tmp->cache, unified_perf_file);
                    freeGraphStatsGeneral(ref_data, arguments.algorithm);
                }

                freeGraphDataStructure(graph, arguments.datastructure);
            }

            arguments.policey   = policy[2];
            for (kk = 0, j = ORDER_CONFIG; j < ORDER_CONFIG + MODE_NUM; ++j, ++kk)
            {
                sprintf (graph_dir, "%s/%s", benchmarks_dir[i], "graph.rand.bin");
                sprintf (label_dir, "%s/%s", benchmarks_dir[i], reorder_labels[j]);
                arguments.lmode = 0; // base is random order
                arguments.lmode_l2 =  lmode_l2[j];
                arguments.lmode_l3 = lmode_l3[j];
                arguments.mmode = mmode[j];
                arguments.fnameb = graph_dir;
                arguments.fnamel = label_dir;
                printf("graph config %5u - %u %u %u - %u %s\n", j, arguments.lmode_l2, arguments.lmode_l3, arguments.mmode, arguments.l1_size / 1024, config_labels[j]);

                graph = generateGraphDataStructure(&arguments);

                initializeMersenneState (&(arguments.mt19937var), 27491095);
                for(jj = 0 ; jj < arguments.trials; jj++)
                {
                    arguments.source = generateRandomRootGeneral(&arguments, graph); // random root each trial
                    ref_data = runGraphAlgorithmsTest(&arguments, graph); // ref stats should mach oother algo
                    GRASP_stats[i][kk] += getGraphAlgorithmsTestMissRateRef(ref_data, arguments.algorithm);
                    // printStatsDoubleTaggedCacheToFile(ref_stats_tmp->cache, unified_perf_file);
                    freeGraphStatsGeneral(ref_data, arguments.algorithm);
                }

                freeGraphDataStructure(graph, arguments.datastructure);
            }

            arguments.policey   = policy[3];
            for (kk = 0, j = (ORDER_CONFIG + MODE_NUM); j < (ORDER_CONFIG + MODE_NUM + MODE_NUM); ++j, ++kk)
            {
                sprintf (graph_dir, "%s/%s", benchmarks_dir[i], "graph.rand.bin");
                sprintf (label_dir, "%s/%s", benchmarks_dir[i], reorder_labels[j]);
                arguments.lmode = 0; // base is random order
                arguments.lmode_l2 =  lmode_l2[j];
                arguments.lmode_l3 = lmode_l3[j];
                arguments.mmode = mmode[j];
                arguments.fnameb = graph_dir;
                arguments.fnamel = label_dir;
                printf("graph config %5u - %u %u %u - %u %s\n", j, arguments.lmode_l2, arguments.lmode_l3, arguments.mmode, arguments.l1_size / 1024, config_labels[j]);

                graph = generateGraphDataStructure(&arguments);

                initializeMersenneState (&(arguments.mt19937var), 27491095);
                for(jj = 0 ; jj < arguments.trials; jj++)
                {
                    arguments.source = generateRandomRootGeneral(&arguments, graph); // random root each trial
                    ref_data = runGraphAlgorithmsTest(&arguments, graph); // ref stats should mach oother algo
                    EXPRESS_stats[i][kk] += getGraphAlgorithmsTestMissRateRef(ref_data, arguments.algorithm);
                    // printStatsDoubleTaggedCacheToFile(ref_stats_tmp->cache, unified_perf_file);
                    freeGraphStatsGeneral(ref_data, arguments.algorithm);
                }

                freeGraphDataStructure(graph, arguments.datastructure);
            }

            arguments.policey   = policy[4];
            for (kk = 0, j = ORDER_CONFIG; j < ORDER_CONFIG + MODE_NUM; ++j, ++kk)
            {
                sprintf (graph_dir, "%s/%s", benchmarks_dir[i], "graph.rand.bin");
                sprintf (label_dir, "%s/%s", benchmarks_dir[i], reorder_labels[j]);
                arguments.lmode = 0; // base is random order
                arguments.lmode_l2 =  lmode_l2[j];
                arguments.lmode_l3 = lmode_l3[j];
                arguments.mmode = mmode[j];
                arguments.fnameb = graph_dir;
                arguments.fnamel = label_dir;
                printf("graph config %5u - %u %u %u - %u %s\n", j, arguments.lmode_l2, arguments.lmode_l3, arguments.mmode, arguments.l1_size / 1024, config_labels[j]);

                graph = generateGraphDataStructure(&arguments);

                initializeMersenneState (&(arguments.mt19937var), 27491095);
                for(jj = 0 ; jj < arguments.trials; jj++)
                {
                    arguments.source = generateRandomRootGeneral(&arguments, graph); // random root each trial
                    ref_data = runGraphAlgorithmsTest(&arguments, graph); // ref stats should mach oother algo
                    GRASP_stats_capi[i][kk] += getGraphAlgorithmsTestMissRateCAPI(ref_data, arguments.algorithm);
                    // printStatsDoubleTaggedCacheToFile(ref_stats_tmp->cache, unified_perf_file);
                    freeGraphStatsGeneral(ref_data, arguments.algorithm);
                }

                freeGraphDataStructure(graph, arguments.datastructure);
            }

            arguments.policey   = policy[5];
            for (kk = 0, j = (ORDER_CONFIG + MODE_NUM); j < (ORDER_CONFIG + MODE_NUM + MODE_NUM); ++j, ++kk)
            {
                sprintf (graph_dir, "%s/%s", benchmarks_dir[i], "graph.rand.bin");
                sprintf (label_dir, "%s/%s", benchmarks_dir[i], reorder_labels[j]);
                arguments.lmode = 0; // base is random order
                arguments.lmode_l2 =  lmode_l2[j];
                arguments.lmode_l3 = lmode_l3[j];
                arguments.mmode = mmode[j];
                arguments.fnameb = graph_dir;
                arguments.fnamel = label_dir;
                printf("graph config %5u - %u %u %u - %u %s\n", j, arguments.lmode_l2, arguments.lmode_l3, arguments.mmode, arguments.l1_size / 1024, config_labels[j]);

                graph = generateGraphDataStructure(&arguments);

                initializeMersenneState (&(arguments.mt19937var), 27491095);
                for(jj = 0 ; jj < arguments.trials; jj++)
                {
                    arguments.source = generateRandomRootGeneral(&arguments, graph); // random root each trial
                    ref_data = runGraphAlgorithmsTest(&arguments, graph); // ref stats should mach oother algo
                    EXPRESS_stats_capi[i][kk] += getGraphAlgorithmsTestMissRateCAPI(ref_data, arguments.algorithm);
                    // printStatsDoubleTaggedCacheToFile(ref_stats_tmp->cache, unified_perf_file);
                    freeGraphStatsGeneral(ref_data, arguments.algorithm);
                }

                freeGraphDataStructure(graph, arguments.datastructure);
            }



        }
        // print out stats to file each graph processed
        FILE *fptr1;
        fptr1 = fopen(unified_perf_file, "a+");
        fprintf(fptr1, " -----------------------------------------------------\n");
        fprintf(fptr1, " CacheSize : %u KB \n", cache_size[k] / 1024 );
        fprintf(fptr1, " -----------------------------------------------------\n");

        fprintf(fptr1, "%-25s, ",  " ");
        fprintf(fptr1, "%-25s, ",  " ");
        for (j = 0; j < ORDER_CONFIG; ++j)
        {
            fprintf(fptr1, "%-14s, ",  config_labels[j]);
        }
        fprintf(fptr1, " \n");

        for ( i = 0; i < GRAPH_NUM; ++i)
        {
            fprintf(fptr1, "%-25s, ",  benchmarks_graphs[i]);
            fprintf(fptr1, "%-25s, ",  "PLRU");
            for (j = 0; j < ORDER_CONFIG; ++j)
            {
                fprintf(fptr1, "%-14f, ",  PLRU_stats[i][j] / arguments.trials);
            }
            fprintf(fptr1, " \n");
            fprintf(fptr1, "%-25s, ",  " ");
            fprintf(fptr1, "%-25s, ",  "SSRIP");
            for (j = 0; j < ORDER_CONFIG; ++j)
            {
                fprintf(fptr1, "%-14f, ",  SSRIP_stats[i][j] / arguments.trials);
            }
            fprintf(fptr1, " \n");
        }
        fprintf(fptr1, " -----------------------------------------------------\n");

        fprintf(fptr1, "%-25s, ",  "GRASP");
        for (j = (ORDER_CONFIG); j < (ORDER_CONFIG + MODE_NUM); ++j)
        {
            fprintf(fptr1, "%-14s, ",  config_labels[j]);
        }
        fprintf(fptr1, " \n");

        for ( i = 0; i < GRAPH_NUM; ++i)
        {
            fprintf(fptr1, "%-25s, ",  benchmarks_graphs[i]);
            for (j = 0; j < MODE_NUM; ++j)
            {
                fprintf(fptr1, "%-14f, ",  GRASP_stats[i][j] / arguments.trials);
            }
            fprintf(fptr1, " \n");
        }
        fprintf(fptr1, " -----------------------------------------------------\n");

        fprintf(fptr1, "%-25s, ",  "EXPRESS");
        for (j = (ORDER_CONFIG + MODE_NUM); j < (ORDER_CONFIG + MODE_NUM + MODE_NUM); ++j)
        {
            fprintf(fptr1, "%-14s, ",  config_labels[j]);
        }
        fprintf(fptr1, " \n");

        for ( i = 0; i < GRAPH_NUM; ++i)
        {
            fprintf(fptr1, "%-25s, ",  benchmarks_graphs[i]);
            for (j = 0; j < MODE_NUM; ++j)
            {
                fprintf(fptr1, "%-14f, ",  EXPRESS_stats[i][j] / arguments.trials);
            }
            fprintf(fptr1, " \n");
        }
        fprintf(fptr1, " -----------------------------------------------------\n");

        fprintf(fptr1, "%-25s, ",  "GRASP_CAPI");
        for (j = (ORDER_CONFIG); j < (ORDER_CONFIG + MODE_NUM); ++j)
        {
            fprintf(fptr1, "%-14s, ",  config_labels[j]);
        }
        fprintf(fptr1, " \n");

        for ( i = 0; i < GRAPH_NUM; ++i)
        {
            fprintf(fptr1, "%-25s, ",  benchmarks_graphs[i]);
            for (j = 0; j < MODE_NUM; ++j)
            {
                fprintf(fptr1, "%-14f, ",  GRASP_stats_capi[i][j] / arguments.trials);
            }
            fprintf(fptr1, " \n");
        }
        fprintf(fptr1, " -----------------------------------------------------\n");

        fprintf(fptr1, "%-25s, ",  "EXPRESS_CAPI");
        for (j = (ORDER_CONFIG + MODE_NUM); j < (ORDER_CONFIG + MODE_NUM + MODE_NUM); ++j)
        {
            fprintf(fptr1, "%-14s, ",  config_labels[j]);
        }
        fprintf(fptr1, " \n");

        for ( i = 0; i < GRAPH_NUM; ++i)
        {
            fprintf(fptr1, "%-25s, ",  benchmarks_graphs[i]);
            for (j = 0; j < MODE_NUM; ++j)
            {
                fprintf(fptr1, "%-14f, ",  EXPRESS_stats_capi[i][j] / arguments.trials);
            }
            fprintf(fptr1, " \n");
        }
        fprintf(fptr1, " -----------------------------------------------------\n");
        fclose(fptr1);
    }
    exit (0);
}





