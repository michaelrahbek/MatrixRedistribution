// Convert a global index into a local index (irrespective of which rank has the index)
int global2local(int index_global, int nblock, int size) {
  int cur_block = index_global / (size * nblock);
  return nblock * cur_block + index_global % nblock;
}


// Calculate which rank has the global index residing in its local index
int global2rank(int index_global, int nblock, int size) {
  return (index_global / nblock) % size;
}
/*int global2rank(int index_global_i,
				int nblock_r,
				int p_r,
				int index_global_j,
				int nblock_c,
				int p_c) {
	int rank_in_r = (index_global_i / nblock_r) % p_r;
	int rank_in_c = (index_global_j / nblock_c) % p_c;
  return rank_in_r * p_c + rank_in_c;
}*/


// Calculate total number of elements on the local rank
int local_size(int N, int nblock, int rank, int p1, int p2, char d) {
  // Total number of full blocks
  int full_blocks = N / nblock;

  // Minimum number of elements on each rank
  int min_elem = (full_blocks / p1) * nblock;
  
  // Overlapping blocks
  int overlap = full_blocks % p1;
  int dim_rank;
  if ( d == 'r')  dim_rank = rank / p2;
  else            dim_rank = rank % p1;

  if ( dim_rank < overlap ) {
    return min_elem + nblock;
  } else if ( dim_rank == overlap ) {
    return min_elem + N % nblock;
  } else {
    return min_elem;
  }
}


/********************** 
**** NEW FUNCTIONS ****
***********************/

// Convert a local index into a global index: global row
int local2global_row(int index_local, int nblock, int p1, int rank, int p2){
  return (index_local / nblock) * (nblock * p1) + (rank / p2) * nblock + (index_local % nblock);
}

// Convert a local index into a global index: global column
int local2global_col(int index_local, int nblock, int p_c, int rank){
  return (index_local / (nblock) ) * (nblock * p_c) + (rank % p_c) * nblock + (index_local %  nblock);
}
