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
          int p_out_c);