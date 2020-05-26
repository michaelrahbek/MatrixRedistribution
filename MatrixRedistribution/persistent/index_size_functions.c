// Convert a global index into a local index (irrespective of which rank has the index)
int global2local(int index_global, int nblock, int size) {
  int cur_block = index_global / (size * nblock);
  return nblock * cur_block + index_global % nblock;
}



// Calculate which rank has the global index residing in its local index
int global2rank(int index_global, int nblock, int size) {
  return (index_global / nblock) % size;
}



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