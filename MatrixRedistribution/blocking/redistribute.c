#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "matrix_functions.h"
#include "index_size_functions.h"

double ** redistribute(
          int N, 
          double **A_in, 
          int nblock_in_r,
          int nblock_in_c,
		      int nblock_out_r,
          int nblock_out_c,
		      int rank,
          int p_in_r, 
          int p_in_c, 
          int p_out_r,
          int p_out_c){

  #ifndef TEST
    // First we calculate the input (local) matrix size
    int N_local_in_r = local_size(N, nblock_in_r, rank, p_in_r, p_in_c, 'r');
    int N_local_in_c = local_size(N, nblock_in_c, rank, p_in_c, p_in_r, 'c');
    
    // Allocate and initialize
    A_in = dmalloc_2d(N_local_in_r, N_local_in_c);
    A_in = matrix_init(A_in, N_local_in_r, N_local_in_c);
  #endif

  // We calculate the new (local) matrix size
  int N_local_out_r = local_size(N, nblock_out_r, rank, p_out_r, p_out_c, 'r');
  int N_local_out_c = local_size(N, nblock_out_c, rank, p_out_c, p_out_r, 'c');

  if ( rank == 0 ) {
    printf("\nStart redistribute matrix from nb_r=%d and nb_c=%d to nb_r=%d and nb_c=%d\n", nblock_in_r, nblock_in_c, nblock_out_r, nblock_out_c);
    fflush(stdout);
  }

  // Allocate memory for the new distributed matrix
  double **A_out = dmalloc_2d(N_local_out_r, N_local_out_c);

  // align ranks and start timing
  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();

  // allocating memory for storing axis' ranks and local indexes
  int *rank_in_r = malloc(N*sizeof(int));
  int *rank_in_c = malloc(N*sizeof(int));
  int *rank_out_r = malloc(N*sizeof(int));
  int *rank_out_c = malloc(N*sizeof(int));
  int *idx_in_r = malloc(N*sizeof(int));
  int *idx_in_c = malloc(N*sizeof(int));
  int *idx_out_r = malloc(N*sizeof(int));
  int *idx_out_c = malloc(N*sizeof(int));
  int rank_in;
  int rank_out;

  // Storing axis' ranks and local indexes in arrays
  for ( int i = 0 ; i < N ; i++ ) {
    rank_in_r[i] = global2rank(i, nblock_in_r, p_in_r);
    rank_in_c[i] = global2rank(i, nblock_in_c, p_in_c);
    rank_out_r[i] = global2rank(i, nblock_out_r, p_out_r);
    rank_out_c[i] = global2rank(i, nblock_out_c, p_out_c);
    idx_in_r[i] = global2local(i, nblock_in_r, p_in_r);
    idx_in_c[i] = global2local(i, nblock_in_c, p_in_c);
    idx_out_r[i] = global2local(i, nblock_out_r, p_out_r);
    idx_out_c[i] = global2local(i, nblock_out_c, p_out_c);
  }

  // Now all processors are ready for send/recv data
  for ( int i = 0 ; i < N ; i++ ) {
    for ( int j = 0 ; j < N ; j++ ) {
      // Figure out which ranks has and which should have the given global index
      rank_in = rank_in_r[i] * p_in_c + rank_in_c[j];
      rank_out = rank_out_r[i] * p_out_c + rank_out_c[j];
      
      if ( rank_in == rank ) {
        if ( rank_out == rank ) {
          A_out[idx_out_r[i]][idx_out_c[j]] = A_in[idx_in_r[i]][idx_in_c[j]];
  	
        } else {
  	      MPI_Send(&A_in[idx_in_r[i]][idx_in_c[j]], 1, MPI_DOUBLE, rank_out, i*N + j, MPI_COMM_WORLD);
        }
        
      } else if ( rank_out == rank ) {
        MPI_Recv(&A_out[idx_out_r[i]][idx_out_c[j]], 1, MPI_DOUBLE, rank_in, i*N + j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  }

  // Clean-up memory!
  free(rank_in_r);
  free(rank_in_c);
  free(rank_out_r);
  free(rank_out_c);
  free(idx_in_r);
  free(idx_in_c);
  free(idx_out_r);
  free(idx_out_c);
  
  // Final timing
  double t1 = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);

  if ( rank == 0 ) {
    printf("Done redistributing matrix from nb_r=%d and nb_c=%d to nb_r=%d and nb_c=%d\n", nblock_in_r, nblock_in_c, nblock_out_r, nblock_out_c);
    fflush(stdout);
  }

  // Print timings, min/avg/max
  double t = t1 - t0;
  double t_avg;
  MPI_Reduce(&t, &t0, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&t, &t1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&t, &t_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  t_avg /= (double) (p_in_r * p_in_c > p_out_r * p_out_c) ? p_in_r * p_in_c : p_out_r * p_out_c;
  if ( rank == 0 ) {
    printf("Time min/avg/max  %12.8f / %12.8f / %12.8f\n", t0, t_avg, t1);
    fflush(stdout);
  }

  return A_out;
}