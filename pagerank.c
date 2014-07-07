#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include <stdalign.h>
#include "immintrin.h"

#include "pagerank.h"

#define square(x) ((x)*(x))

bool euclidean(double *aOldScores, double *aScores, const int npages, double EP) {
    double tempN4;
    double tempN8;
    double tempN12;
    double tempN16;
    double tempN20;
    double tempN24;
    double tempN28;
    double tempN32;
    register double total = 0;
    register double *scores = aScores, *oldScores = aOldScores;
    register int i;
    // Unravelled loop
    for(i = 0; i < npages - 7; i+=8) {
        tempN4 = scores[0] - oldScores[0];
        tempN8 = scores[1] - oldScores[1];
        tempN12 = scores[2] - oldScores[2];
        tempN16 = scores[3] - oldScores[3];
        tempN20 = scores[4] - oldScores[4];
        tempN24 = scores[5] - oldScores[5];
        tempN28 = scores[6] - oldScores[6];
        tempN32 = scores[7] - oldScores[7];
        total += square(tempN4) + square(tempN8) + square(tempN12) + square(tempN16) + square(tempN20) + square(tempN24) + square(tempN28) + square(tempN32);
        // Early termination, if total is greater than EP at any stage in loop, bail out. Huge Speedup
        if(total > EP) 
            return true;
        scores += 8;
        oldScores += 8;
    }
    // Pick at the remainder
    double final = 0, temp;
    for(; i < npages; i++) {
        temp = aScores[i] - aOldScores[i];
        final += square(temp);
    }
    final += total;
    return (final > EP);
}

void pagerank(list* plist, int ncores, int npages, int nedges, double dampener) {
    const double EP = square(EPSILON);
    const int pagesPlusOne = npages + 1;

    int *row_index = malloc(nedges * sizeof(int));
    int *col_index = malloc(nedges * sizeof(int));
    double *adjacencyMatrix = malloc(nedges * sizeof(double));
    double *values = malloc(pagesPlusOne * sizeof(double));

    register int i, index;
    // Caches possible float divisions of noutlinks
    for(i = 1; i < pagesPlusOne; i++) {
        values[i] = dampener/i;
    }

    list* inlinks = NULL;
    i = 0;
    // Iterate over each page
    for(node* current = plist->head; current != NULL; current = current->next) {
        index = current->page->index;
        inlinks = current->page->inlinks;
        if(inlinks == NULL) continue;
        // Iterate over all its inlinks
        for(node* ncurr = inlinks->head; ncurr != NULL; ncurr = ncurr->next, i++) {
            // Sets the value of its outlinks in the CSR matrix
            row_index[i] = index;
            col_index[i] = ncurr->page->index;
            adjacencyMatrix[i] = values[ncurr->page->noutlinks];
        }
    }

    /*****
     * Start PageRank Algorithm
     */
    // Declare array memory for algorithm
    double *starting = malloc(npages * sizeof(double));
    // Aligned memory exhibits a small speed  up in these cases
    double *scores = aligned_alloc(32, npages * sizeof(double));
    double *oldScores = aligned_alloc(32, npages * sizeof(double));
    const double one_over_pages = 1.0/npages;
    const double base = (1 - dampener) * one_over_pages;

    // Set the initial scores array to 1/n, and set the values of the starting array
    for(i = 0; i < npages; i++) {
        scores[i] = one_over_pages;
        starting[i] = base;
    }

    // Decclare loop pointers
    register int *p_col_index, *p_row_index;
    double *p_adjacencyMatrix;
    do {
        // Reuse existing allocated memory, instead of freeing it each time
        p_adjacencyMatrix = scores;
        scores = oldScores;
        oldScores = p_adjacencyMatrix;
        // Reset incremented pointers for loop unrolling
        p_col_index = col_index;
        p_row_index = row_index;
        p_adjacencyMatrix = adjacencyMatrix;
        // Copy initial scores to array (faster than assigning each time)
        memcpy(scores, starting, npages * sizeof(double));
        // Unrolled the loop to reduce the number of comparisons at each turn. 
        // 8 operations has the highest speed gain, likely from making the best use of cache
        for(i = 0; i < nedges - 7; i += 8) {
            scores[p_row_index[0]] += oldScores[p_col_index[0]] * p_adjacencyMatrix[0];
            scores[p_row_index[1]] += oldScores[p_col_index[1]] * p_adjacencyMatrix[1];
            scores[p_row_index[2]] += oldScores[p_col_index[2]] * p_adjacencyMatrix[2];
            scores[p_row_index[3]] += oldScores[p_col_index[3]] * p_adjacencyMatrix[3];
            scores[p_row_index[4]] += oldScores[p_col_index[4]] * p_adjacencyMatrix[4];
            scores[p_row_index[5]] += oldScores[p_col_index[5]] * p_adjacencyMatrix[5];
            scores[p_row_index[6]] += oldScores[p_col_index[6]] * p_adjacencyMatrix[6];
            scores[p_row_index[7]] += oldScores[p_col_index[7]] * p_adjacencyMatrix[7];
            p_col_index += 8;
            p_row_index += 8;
            p_adjacencyMatrix += 8;
        }
        // Do the rest
        for(; i < nedges; i++) {
            scores[row_index[i]] += oldScores[col_index[i]] * adjacencyMatrix[i];
        }
    } while(euclidean(oldScores, scores, npages, EP));

    // Iterate over linked nodes and print score | Optimising this has no significant effect
    for(node* current = plist->head; current != NULL; current = current->next) {
        printf("%s %.4lf\n", current->page->name, scores[current->page->index]);
    }

    // Free Memory
    free(starting);
    free(scores);
    free(oldScores);
    free(adjacencyMatrix);
    free(row_index);
    free(col_index);
    free(values);
}

/*
######################################
### DO NOT MODIFY BELOW THIS POINT ###
######################################
*/

int main(void) {

    /*
######################################################
### DO NOT MODIFY THE MAIN FUNCTION OR HEADER FILE ###
######################################################
*/

    list* plist = NULL;

    double dampener;
    int ncores, npages, nedges;

    /* read the input then populate settings and the list of pages */
    read_input(&plist, &ncores, &npages, &nedges, &dampener);

    /* run pagerank and output the results */
    pagerank(plist, ncores, npages, nedges, dampener);

    /* clean up the memory used by the list of pages */
    page_list_destroy(plist);

    return 0;
}
