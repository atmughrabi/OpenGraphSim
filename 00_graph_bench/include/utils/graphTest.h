#ifndef GRAPHTEST_H
#define GRAPHTEST_H


#include <stdint.h>
#include "graphConfig.h"


uint32_t compareRealRanks(uint32_t *arr1, uint32_t *arr2, uint32_t arr1_size, uint32_t arr2_size);
uint32_t cmpGraphAlgorithmsTestStats(void *ref_stats, void *cmp_stats, uint32_t algorithm);
uint32_t compareDistanceArrays(uint32_t *arr1, uint32_t *arr2, uint32_t arr1_size, uint32_t arr2_size);
void *runGraphAlgorithmsTest(struct Arguments *arguments, void *graph);
uint32_t equalFloat(float a, float b, double epsilon);
uint32_t compareFloatArrays(float *arr1, float *arr2, uint32_t arr1_size, uint32_t arr2_size);
float getGraphAlgorithmsTestTime(void *ref_stats, uint32_t algorithm);

#endif