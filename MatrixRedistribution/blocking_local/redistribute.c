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

  // First we calculate the input (local) matrix size
  int N_local_in_r, N_local_in_c;
  if (rank < p_in_r*p_in_c) {
    N_local_in_r = local_size(N, nblock_in_r, rank, p_in_r, p_in_c, 'r');
    N_local_in_c = local_size(N, nblock_in_c, rank, p_in_c, p_in_r, 'c');
  }
  else {
     N_local_in_r = 0;
     N_local_in_c = 0;
  }

  // We calculate the new (local) matrix size
  int N_local_out_r, N_local_out_c;
  if (rank < p_out_r*p_out_c) {
    N_local_out_r = local_size(N, nblock_out_r, rank, p_out_r, p_out_c, 'r');
    N_local_out_c = local_size(N, nblock_out_c, rank, p_out_c, p_out_r, 'c');
  }
  else {
     N_local_out_r = 0;
     N_local_out_c = 0;
  }

  #ifndef TEST
    // Allocate and initialize
    A_in = dmalloc_2d(N_local_in_r, N_local_in_c);
    A_in = matrix_init(A_in, N_local_in_r, N_local_in_c);
  #endif

  if ( rank == 0 ) {
    printf("\nStart redistribute matrix from nb_r=%d and nb_c=%d to nb_r=%d and nb_c=%d\n", nblock_in_r, nblock_in_c, nblock_out_r, nblock_out_c);
    fflush(stdout);
  }

  // Allocate memory for the new distributed matrix
  double **A_out = dmalloc_2d(N_local_out_r, N_local_out_c);
  
  // align ranks and start timing
  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();

  /* SEND */

  int *rank_out_r = malloc(N*sizeof(int));
  int *rank_out_c = malloc(N*sizeof(int));
  int *arr_idx_out_r = malloc(N*sizeof(int));
  int *arr_idx_out_c = malloc(N*sizeof(int));

  for ( int i = 0 ; i < N ; i++ ) {
    rank_out_r[i] = global2rank(i, nblock_out_r, p_out_r);
    rank_out_c[i] = global2rank(i, nblock_out_c, p_out_c);
    arr_idx_out_r[i] = global2local(i, nblock_out_r, p_out_r);
    arr_idx_out_c[i] = global2local(i, nblock_out_c, p_out_c);
  }

  int *global_in_c = malloc(N_local_in_c*sizeof(int));
  for(int idx_in_c = 0 ; idx_in_c < N_local_in_c ; idx_in_c++ ) 
    global_in_c[idx_in_c] = local2global_col(idx_in_c, nblock_in_c, p_in_c, rank);  

  // Now all processors are ready for send data
  for ( int idx_in_r = 0 ; idx_in_r < N_local_in_r ; idx_in_r++ ) {
    // get global indices based on the whole matrix NxN --> same notation as the first version
    int i = local2global_row(idx_in_r, nblock_in_r, p_in_r, rank, p_in_c);
    for ( int idx_in_c = 0 ; idx_in_c < N_local_in_c ; idx_in_c++ ) {
      // get global indices based on the whole matrix NxN --> same notation as the first version
      int j = global_in_c[idx_in_c];
      // Figure out which rank should have the given global index
      int rank_out = rank_out_r[i] * p_out_c + rank_out_c[j];
      if ( rank_out == rank ) {
	      // this rank also has the receive
        int idx_out_r = arr_idx_out_r[i];
        int idx_out_c = arr_idx_out_c[j];
        A_out[idx_out_r][idx_out_c] = A_in[idx_in_r][idx_in_c];
      } else {
	      MPI_Send(&A_in[idx_in_r][idx_in_c], 1, MPI_DOUBLE, rank_out, i*N + j, MPI_COMM_WORLD);
      }
        
    }
  }

  free(global_in_c);
  free(rank_out_r);
  free(rank_out_c);
  free(arr_idx_out_r);
  free(arr_idx_out_c);

  /* RECV */

  int *rank_in_r = malloc(N*sizeof(int));
  int *rank_in_c = malloc(N*sizeof(int));

  for ( int i = 0 ; i < N ; i++ ) {
    rank_in_r[i] = global2rank(i, nblock_in_r, p_in_r);
    rank_in_c[i] = global2rank(i, nblock_in_c, p_in_c);
  }

  int *global_out_c = malloc(N_local_out_c*sizeof(int));
  for(int idx_out_c = 0 ; idx_out_c < N_local_out_c ; idx_out_c++) 
    global_out_c[idx_out_c] = local2global_col(idx_out_c, nblock_out_c, p_out_c, rank);
  
  // Now all processors are ready for recv data  
  for ( int idx_out_r = 0 ; idx_out_r < N_local_out_r ; idx_out_r++ ) {
    // get global indices based on the whole matrix NxN --> same notation as the first version
    int i = local2global_row(idx_out_r, nblock_out_r, p_out_r, rank, p_out_c);
    for ( int idx_out_c = 0 ; idx_out_c < N_local_out_c ; idx_out_c++ ) {
      // get global indices based on the whole matrix NxN --> same notation as the first version
      int j = global_out_c[idx_out_c];
      // Figure out which rank had the given global index
      int rank_in = rank_in_r[i] * p_in_c + rank_in_c[j];
      if ( rank_in != rank ) {
        MPI_Recv(&A_out[idx_out_r][idx_out_c], 1, MPI_DOUBLE, rank_in, i*N + j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);        
      }
        
    }
  }

  free(global_out_c);
  free(rank_in_r);
  free(rank_in_c);

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