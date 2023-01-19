#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct
{
    double * m;
    unsigned columns;
    unsigned rows;
}  matrix_t;


matrix_t * alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
    res->m = (double *) calloc(columns * rows, sizeof(double));
    res->columns = columns;
    res->rows = rows;
    return res;
}

void initMatrix(matrix_t *m){
    for (int i=0; i<100; i++){
        for (int j=0; j<100; j++){
            m->m[i*100+j] = 10;
        }
    }
    m->columns = 100;
    m->rows = 100;
}

__global__
void matrix_dot_kernel(matrix_t *m1, matrix_t *m2, matrix_t *res){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < m1->rows && col < m2->columns) {
        int idx = col + row * m2->columns;
        double var = 0.0;
        for (int ii = 0; ii < m1->columns; ii++){
            var += m1->m[ii + row * m1->columns] * m2->m[col + ii * m2->columns];
            }
        res->m[idx] = var;
    }
}

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    
    matrix_t *d1;
    matrix_t *d2;
    matrix_t *dres;

    cudaMalloc((void **)&d1, m1->rows * m1->columns * sizeof(double));
    cudaMalloc((void **)&d2, m2->rows * m2->columns * sizeof(double));
    cudaMalloc((void **)&dres, res->rows * res->columns * sizeof(double));

    cudaMemcpy(d1, m1, m1->rows * m1->columns * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d2, m2, m2->rows * m2->columns * sizeof(double), cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((unsigned)m2->columns) / blockDim.x), ceil(((unsigned)m1->rows) / blockDim.y));

    matrix_dot_kernel <<< gridDim, blockDim >>> (d1, d2, dres);

    cudaMemcpy(res, dres, res->rows * res->columns * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d1);
    cudaFree(d2);
    cudaFree(dres);
}

int main(){
    matrix_t *m1 = alloc_matrix(100,100);
    matrix_t *m2 = alloc_matrix(100,100);
    matrix_t *m3 = alloc_matrix(100,100);
    initMatrix(m1);
    initMatrix(m2);
    matrix_dot(m1,m2,m3);
    printf("m[5] = %d", (int)m3->m[5]);
    return 0;
}