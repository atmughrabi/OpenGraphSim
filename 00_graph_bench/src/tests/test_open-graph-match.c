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

#include "graphTest.h"

int
main (int argc, char **argv)
{

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
    arguments.algorithm = 0;
    arguments.datastructure = 0;
    arguments.pushpull = 0;
    arguments.sort = 1;

    arguments.lmode = 0;
    arguments.lmode_l2 = 0;
    arguments.lmode_l3 = 0;
    arguments.mmode = 0;

    arguments.symmetric = 0;
    arguments.weighted = 0;
    arguments.delta = 1;
    arguments.pre_numThreads = 4;
    arguments.algo_numThreads = 4;
    arguments.ker_numThreads = 4;
    
    arguments.fnameb = "../01_test_graphs/LAW/LAW-enron/graph.bin";
    arguments.fnamel = "../01_test_graphs/LAW/LAW-enron/graph_Gorder.labels";
    arguments.fnameb_format = 1;
    arguments.convert_format = 1;
    initializeMersenneState (&(arguments.mt19937var), 27491095);

    void *graph = NULL;
    uint32_t missmatch = 0;
    uint32_t total_missmatch = 0;
    void *ref_data;
    void *cmp_data;

    for(arguments.algorithm = 0 ; arguments.algorithm < 9; arguments.algorithm++)
    {
        for(arguments.datastructure = 0 ; arguments.datastructure < 4; arguments.datastructure++)
        {

            if(arguments.algorithm == 8)  // Triangle counting depends on order
            {

                arguments.sort = 1;
                // arguments.lmode = 2;
            }
            if(arguments.algorithm == 9)  // Incremental aggregation order
            {

                arguments.sort = 1;
                // arguments.lmode = 2;
            }

            graph = generateGraphDataStructure(&arguments);

            arguments.iterations = 100;
            arguments.trials = (generateRandInt(&(arguments.mt19937var)) % 50) + 1; // random number of trials


            while(arguments.trials)
            {
                arguments.source = generateRandomRootGeneral(&arguments, graph); // random root each trial
                ref_data = runGraphAlgorithmsTest(&arguments, graph); // ref stats should mach oother algo

                for(arguments.pushpull = 0 ; arguments.pushpull < 9; arguments.pushpull++)
                {

                    cmp_data = runGraphAlgorithmsTest(&arguments, graph);

                    if(ref_data != NULL && cmp_data != NULL)
                    {
                        missmatch = cmpGraphAlgorithmsTestStats(ref_data, cmp_data, arguments.algorithm);
                    }

                    total_missmatch += missmatch;

                    if(missmatch != 0)
                    {
                        printf("FAIL : Trial [%u] Graph [%s] Missmatches [%u] \nFAIL : DataStructure [%u] Algorithm [%u] Direction [%u]\n\n", arguments.trials, arguments.fnameb, missmatch, arguments.datastructure, arguments.algorithm, arguments.pushpull);
                        exit (1);
                    }
                    else
                    {
                        printf("PASS : Trial [%u] Graph [%s] Missmatches [%u] \nPASS : DataStructure [%u] Algorithm [%u] Direction [%u]\n\n", arguments.trials, arguments.fnameb, missmatch, arguments.datastructure, arguments.algorithm, arguments.pushpull);
                    }

                    freeGraphStatsGeneral(cmp_data, arguments.algorithm);
                }

                freeGraphStatsGeneral(ref_data, arguments.algorithm);
                arguments.trials--;

            }
            freeGraphDataStructure(graph, arguments.datastructure);
        }
    }

    if(total_missmatch != 0)
    {
        printf("FAIL : Trial [%u] Graph [%s] Missmatches [%u]\n", arguments.trials, arguments.fnameb, total_missmatch);
    }
    else
    {
        printf("PASS : Trial [%u] Graph [%s] Missmatches [%u]\n", arguments.trials, arguments.fnameb, total_missmatch);
    }

    exit (0);
}





