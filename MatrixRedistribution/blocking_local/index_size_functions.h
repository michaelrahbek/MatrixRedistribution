int global2local(int index_global, int nblock, int size);

int global2rank(int index_global, int nblock, int size);

int local_size(int N, int nblock, int rank, int p1, int p2, char d);

int local2global_row(int index_local, int nblock, int p1, int rank, int p2);

int local2global_col(int index_local, int nblock, int p_c, int rank);