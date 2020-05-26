double ** dmalloc_2d(int m, int n);

double ** matrix_init(double **A, int m, int n);

void dfree_2d(double **A);

double ** dmalloc_2d_dynamic(int m, int n);

void dfree_2d_dynamic(double **A, int m);

void push(double **arr, double value, int *size, int *capacity);