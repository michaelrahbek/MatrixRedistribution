#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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

int ** malloc_2d(int m, int n) {
    if (m <= 0 || n <= 0) return NULL;
    int **A = malloc(m * sizeof(int *));
    if (A == NULL) return NULL;
    A[0] = malloc(m*n*sizeof(int));
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
void free_2d(int **A) {
    free(A[0]);
    free(A);
}

void dfree_2d(double **A){
  free(A[0]);
  free(A);
}

/******************************
 ******* DYNAMIC VERSION *******
 ******************************/

// Allocating memory for matrix
double ** dmalloc_2d_dynamic(int m, int n) {
    if (m <= 0 || n <= 0) return NULL;
    double **A = malloc(m * sizeof(double *));
    for (int i = 0; i < m; i++)
        A[i] = malloc(n * sizeof(double)); //A[0] + i * n;
    return A;
}

// Freeing dynamic allocated memory for matrix
void dfree_2d_dynamic(double **A, int m){
    for (int i = 0; i < m; i++)
        free(A[i]);
    free(A);
}

// Dynamically allocate 1d array: Double size
void push(double **arr, double value, int *size, int *capacity){
    if(*size >= *capacity){
        //printf(" HERE\n" );
        double *tmp = malloc( sizeof(double) * *capacity * 2);
        memcpy(tmp, *arr, *capacity * sizeof(double));
        free(*arr);
        *arr = tmp;
        *capacity = *size * 2;
    }
    (*arr)[*size] = value;
    *size = *size + 1;
}
