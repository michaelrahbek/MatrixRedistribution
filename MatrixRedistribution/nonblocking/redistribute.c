#include <stdlib.h>
#include <stdio.h>
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

  // Dynamically allocate memory for gathered elements
  int mpi_size_recv = p_in_c*p_in_r;

  // initial guess of the amount of elements that are gonna be recv: 2* is for (values, positions) and 2* for safety
  int INITIAL_CAPACITY_recv = (int) (2*2*( (double)(N_local_out_r*N_local_out_c) / (double) (p_in_r*p_in_c)));
  INITIAL_CAPACITY_recv = (INITIAL_CAPACITY_recv > 0) ? INITIAL_CAPACITY_recv : 2;
  //printf("IN RECV%d \n", INITIAL_CAPACITY_recv);
  
  double **gather_recv = dmalloc_2d_dynamic(mpi_size_recv, INITIAL_CAPACITY_recv);
  int amount_recv[mpi_size_recv]; //size that is gonna be recv to each rank
  int capacity_recv[mpi_size_recv]; //bookeep how much memory has been allocated

  for (int i=0; i<mpi_size_recv; ++i) {
    amount_recv[i] = 0;
    capacity_recv[i] = INITIAL_CAPACITY_recv;
  }

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
        // push value
        push(&gather_recv[rank_in], (double) 0, &amount_recv[rank_in], &capacity_recv[rank_in]);
        // push position
        push(&gather_recv[rank_in], (double) 0, &amount_recv[rank_in], &capacity_recv[rank_in]);
      }  
    }
  }

  free(rank_in_r);
  free(rank_in_c);
  
  // prepare nonblocking recv
  MPI_Request recv_requests[mpi_size_recv-1];
  int recv_request_count = 0;

  // Send gathered arrays
  for(int i = 0; i < mpi_size_recv ; ++i) {
    if ((i != rank) && (amount_recv[i]>0)) {
      //MPI_Recv(gather_recv[i], amount_recv[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Irecv(gather_recv[i], amount_recv[i], MPI_DOUBLE, i, i, MPI_COMM_WORLD, &recv_requests[recv_request_count]);
      recv_request_count++;
    }
  }

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

  // Dynamically allocate memory for gathered elements
  int mpi_size_send = p_out_c*p_out_r;
  
  // initial guess of the amount of elements that are gonna be sent: 2* is for (values, positions) and 2* for safety
  int INITIAL_CAPACITY_send = (int) (2*2*( (double)(N_local_in_r*N_local_in_c) / (double) (p_out_r*p_out_c)));
  INITIAL_CAPACITY_send = (INITIAL_CAPACITY_send > 0) ? INITIAL_CAPACITY_send : 2;
  //printf("IN SEND%d \n", INITIAL_CAPACITY_send);
  
  double **gather_send = dmalloc_2d_dynamic(mpi_size_send, INITIAL_CAPACITY_send);
  int amount_send[mpi_size_send]; //size that is gonna be send to each rank
  int capacity_send[mpi_size_send]; //bookeep how much memory has been allocated

  for (int i=0; i<mpi_size_send; ++i) {
    amount_send[i] = 0;
    capacity_send[i] = INITIAL_CAPACITY_send;
  }

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
        // push value
        push(&gather_send[rank_out], A_in[idx_in_r][idx_in_c], &amount_send[rank_out], &capacity_send[rank_out]);
        // push position
        push(&gather_send[rank_out], (double) i*N + j, &amount_send[rank_out], &capacity_send[rank_out]);
      }
    }
  }

  free(global_in_c);
  free(rank_out_r);
  free(rank_out_c);
  free(arr_idx_out_r);
  free(arr_idx_out_c);

  // Send gathered arrays
  for(int i = 0; i < mpi_size_send ; ++i) {
    if ((i != rank) && (amount_send[i]>0)) 
      MPI_Rsend(gather_send[i], amount_send[i], MPI_DOUBLE, i, rank, MPI_COMM_WORLD);
  }

  // free send bookeeping
  dfree_2d_dynamic(gather_send, mpi_size_send);

  //MPI_Waitall(send_request_count, send_requests, MPI_STATUSES_IGNORE);
  
  /* RECV continuation */
  MPI_Waitall(recv_request_count, recv_requests, MPI_STATUSES_IGNORE);

  for(int i = 0; i < mpi_size_recv ; ++i) {
      if ((i != rank) && (amount_recv[i]>0)) {
          for(int j = 0; j < amount_recv[i]; j += 2) {
              int i_global = (int) gather_recv[i][j+1] / N;
              int j_global = (int) gather_recv[i][j+1] % N;
              int idx_out_r = global2local(i_global, nblock_out_r, p_out_r);
              int idx_out_c = global2local(j_global, nblock_out_c, p_out_c);
              A_out[idx_out_r][idx_out_c] = gather_recv[i][j];
          }
      }
  }

  // free recv bookeeping
  dfree_2d_dynamic(gather_recv, mpi_size_recv);

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