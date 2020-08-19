#ifndef GRAPHCONFIG_H
#define GRAPHCONFIG_H

#include <stdint.h>
#include "mt19937.h"


#define WEIGHTED 0
#define DIRECTED 1

extern int numThreads;
extern mt19937state *mt19937var;

/* Used by main to communicate with parse_opt. */
struct Arguments
{
    int wflag;
    int xflag;
    int sflag;
    int Sflag;
    int dflag;
    uint32_t binSize;
    uint32_t verbosity;
    uint32_t iterations;
    uint32_t trials;
    double epsilon;
    int root;
    uint32_t algorithm;
    uint32_t datastructure;
    uint32_t pushpull;
    uint32_t sort;
    uint32_t lmode; // reorder mode
    uint32_t lmode_l2; // reorder mode second layer
    uint32_t mmode; // mask mode
    uint32_t symmetric;
    uint32_t weighted;
    uint32_t delta;
    uint32_t numThreads;
    char *fnameb;
    char *fnamel;
    uint32_t fnameb_format;
    uint32_t convert_format;
};

#endif