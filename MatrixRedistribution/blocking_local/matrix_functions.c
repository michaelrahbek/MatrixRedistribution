#include <stdlib.h>

// Allocating memory for matrix
double ** dmalloc_2d(int m, int n) {
    if (m <= 0 || n <= 0) return NULL;
    double **A = malloc(m * sizeof(double *));
    if (A == NULL) return NULL;
    A[0] = malloc(m*n*sizeof(double));
    if (A[0] == NULL) {
        free(A);
        return NULL;
    }
    for (int i = 1; i < m; i++)
        A[i] = A[0] + i * n;
    return A;
}



// Initializing values for matrix (Consecutive numbers - linear matrix index)
double ** matrix_init(double **A, int m, int n){
    for ( int i = 0 ; i < m ; i++ )
        for ( int j = 0 ; j < n ; j++ )
            A[i][j] = i*n + j;
    return A;
}



// Freeing allocated memory for matrix
void dfree_2d(double **A){
  free(A[0]);
  free(A);
}