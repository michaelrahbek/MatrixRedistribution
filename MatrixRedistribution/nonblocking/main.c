
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include "warm_up.h"
#include "redistribute.h"
#include "matrix_functions.h"

int main(int argc, char *argv []) {

  // Initialize MPI
  MPI_Init(&argc,&argv);

  // Query size and rank
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // No buffering of stdout
  setbuf(stdout, NULL);

  // Get user defined input
  int N, 
      p_r1, 
      p_c1, 
      nblock_r1, 
      nblock_c1, 
      p_r2,
      p_c2,
      nblock_r2,
      nblock_c2;

  // Checking if enough inputs are given
  if ( argc < 10 ) {
    if (rank==0) printf("\nError. Not enough inputs.\n\n");
    MPI_Finalize();
    return 0;
  }
  
  N = atoi(argv[1]);
  p_r1 = atoi(argv[2]);
  p_c1 = atoi(argv[3]);
  nblock_r1 = atoi(argv[4]);
  nblock_c1 = atoi(argv[5]);
  p_r2 = atoi(argv[6]);
  p_c2 = atoi(argv[7]);
  nblock_r2 = atoi(argv[8]);
  nblock_c2 = atoi(argv[9]);

  if ( rank == 0 ) {
    printf("Arguments:\n");
    printf("  N = %d\n", N);
    printf("  nblock_r1 = %d and nblock_c1 = %d\n", nblock_r1, nblock_c1);
    printf("  p_r1 = %d and p_c1 = %d\n", p_r1, p_c1);
    printf("  nblock_r2 = %d and nblock_c2 = %d\n", nblock_r2, nblock_c2);
    printf("  p_r2 = %d and p_c2 = %d\n", p_r2, p_c2);
    printf("  MPI processors: %d\n", size);
  }

  // Error checking the product of processors along columns and rows
  if ( p_r1*p_c1 != size || p_r2*p_c2 != size ){
    if (rank==0) printf("\nError. Total number of processors does not equal the product of processors along columns and rows.\n\n");
    MPI_Finalize();
   return 0;
  }

  // This makes the 'project' executable for perfomance analysis
  #ifndef TEST
    //Warming up
    warm_up();

    // Initializing matrix to use for input
    double **A_1 = NULL;

    // Performing the redistribution
    double **A_2 = redistribute(N, 
                              A_1,
                              nblock_r1,
                              nblock_c1,
                              nblock_r2,
                              nblock_c2,
                              rank,
                              p_r1,
                              p_c1, 
                              p_r2,
                              p_c2);

    // Clean-up memory (has to check for the case of returned NULL pointer)
    if ( A_1 != NULL ) dfree_2d(A_1);
    if ( A_2 != NULL ) dfree_2d(A_2);
  #endif

  // This makes the 'test' executable for testing correctness of the code
  #ifdef TEST
    // Initialize a matrix with *random* data
    double **A_dense = NULL;
    if ( rank == 0 ) {
      // allocate and initialize
      A_dense = dmalloc_2d(N, N);
      A_dense= matrix_init(A_dense, N, N);
      
      printf("Done initializing matrix on root node\n");
      fflush(stdout);
    }
  
    // Distribute the first (dense) matrix
    double **A_1 = redistribute(N, 
                                A_dense, 
                                N,
                                N,
                                nblock_r1,
                                nblock_c1,
                                rank,
                                1, 
                                1, 
                                p_r1,
                                p_c1);
  
    // Redistribute to the next level
    double **A_2 = redistribute(N, 
                                A_1,
                                nblock_r1,
                                nblock_c1,
                                nblock_r2,
                                nblock_c2,
                                rank,
                                p_r1,
                                p_c1, 
                                p_r2,
                                p_c2);
    // Clean-up memory
    if ( A_1 != NULL ) dfree_2d(A_1);

    // Redistribute to the local one again to check we have done it correctly
    double **A_final = redistribute(N, 
                                    A_2,
                                    nblock_r2,
                                    nblock_c2,
                                    N,
                                    N,
                                    rank, 
                                    p_r2,
                                    p_c2, 
                                    1,
                                    1);
    // Clean-up memory
    if ( A_2 != NULL ) dfree_2d(A_2);
  
    if ( rank == 0 ) {
      printf("\nTesting function correctness. Errors will be shown here if any.\n\n");
      for ( int i = 0 ; i < N ; i++ ) {
        for ( int j = 0 ; j < N ; j++ ) {
          if ( fabs(A_dense[i][j] - A_final[i][j]) > 0.1 ) {
  	        printf("Error on index [%3d][%3d]   %5.1f %5.1f\n", i, j, A_dense[i][j],   A_final[i][j]);
          }
        }
      }
    }
  
    // Clean-up memory!
    if ( A_final != NULL ) dfree_2d(A_final);
    if ( A_dense != NULL ) dfree_2d(A_dense);
  #endif

  MPI_Finalize();
  return 0;
}
