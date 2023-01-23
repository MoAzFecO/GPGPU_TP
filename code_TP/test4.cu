#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <math.h>

typedef struct
{
    double * m;
    unsigned columns;
    unsigned rows;
}  matrix_t;


matrix_t * alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );

    double *m;
    cudaMalloc((void **)&m, rows * columns * sizeof(double));

    res->m = m;
    res->columns = columns;
    res->rows = rows;
    return res;
}

const int TILE_WIDTH = 16;

__global__
void matrix_dot_shared_memory (double * A, double * B, double * C, int numARows, int numAColumns, int numBColumns )
{
    __shared__ double ds_M[ TILE_WIDTH ][ TILE_WIDTH ];
    __shared__ double ds_N[ TILE_WIDTH ][ TILE_WIDTH ];

    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    int Row = blockIdx.y * blockDim.y+ ty;
    int Col = blockIdx.x * blockDim.x+ tx;

    double Pvalue = 0;

    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
            ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
        else
            ds_M[ty][tx] = 0;
        if (Col < numBColumns && m*TILE_WIDTH+ty < numAColumns)
           ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
        else
           ds_N[ty][tx] = 0;

        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k)
           Pvalue += ds_M[ty][k] * ds_N[k][tx];
        __syncthreads();
    }
    if (Row < numARows && Col < numBColumns)
        C[Row*numBColumns+Col] = Pvalue;
}

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));

    
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((double)m2->columns) / blockDim.x), ceil(((double)m1->rows) / blockDim.y));

    matrix_dot_shared_memory <<< gridDim, blockDim >>> (m1->m, m2->m, res->m, m1->rows, m1->columns, m2->columns);
}

__global__
void initMatrix_kernel(double *m, int mRows, int mColumns){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    int idx = col + row * mColumns;

    if (idx < mRows * mColumns) {
        m[idx] = 1;
    }
}

void initMatrix(matrix_t *m){
    
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((double)m->columns) / blockDim.x), ceil(((double)m->rows) / blockDim.y));

    initMatrix_kernel <<< gridDim, blockDim >>> (m->m, m->rows, m->columns);

    m->columns = 100;
    m->rows = 100;
}

__global__
void hadamard_product_kernel(double *m1, double *m2, double *res, int m1rows, int m1columns, int m2columns){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    int idx = col + row * m2columns;

    if (idx < m1rows * m1columns) {
        res[idx] = m1[idx] * m2[idx];
    }
}

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((double)m2->columns) / blockDim.x), ceil(((double)m1->rows) / blockDim.y));

    hadamard_product_kernel <<< gridDim, blockDim >>> (m1->m, m2->m, res->m, m1->rows, m1->columns, m2->columns);
}

__global__
void matrix_sum_kernel(double *m1, double *m2, double *res, int m1rows, int m1columns, int m2columns){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    int idx = col + row * m2columns;

    if (idx < m1rows * m1columns) {
        res[idx] = m1[idx] + m2[idx];
    }
}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((double)m2->columns) / blockDim.x), ceil(((double)m1->rows) / blockDim.y));

    matrix_sum_kernel <<< gridDim, blockDim >>> (m1->m, m2->m, res->m, m1->rows, m1->columns, m2->columns);
}

__global__
void matrix_function_kernel(double (*f)(double), double *m1, double *res, int m1rows, int m1columns){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    int idx = col + row * m1columns;

    if (idx < m1rows * m1columns) {
        res[idx] = f(m1[idx]);
    }
}

void matrix_function0(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((double)m1->columns) / blockDim.x), ceil(((double)m1->rows) / blockDim.y));

    matrix_function_kernel <<< gridDim, blockDim >>> (sqrt, m1->m, res->m, m1->rows, m1->columns);
}

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    double m[m1->rows * m1->columns * sizeof(double)];
    double r[m1->rows * m1->columns * sizeof(double)];
    cudaMemcpy(m, m1->m, m1->rows * m1->columns * sizeof(double), cudaMemcpyDeviceToHost);

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        r[idx] = f(m[idx]);
    }
    cudaMemcpy(res->m, r, m1->rows * m1->columns * sizeof(double), cudaMemcpyHostToDevice);
}

int main(){
    matrix_t *m1 = alloc_matrix(100,100);
    matrix_t *m2 = alloc_matrix(100,100);
    matrix_t *m3 = alloc_matrix(100,100);
    initMatrix(m1);
    initMatrix(m2);
    matrix_function(m1,sqrt,m3);
    double m[m3->rows * m3->columns * sizeof(double)];
    cudaMemcpy(m, m3->m, m3->rows * m3->columns * sizeof(double), cudaMemcpyDeviceToHost);
    printf("m[5] = %d", (int)m[9999]);
    printf("try = %d\n", (int)sqrt(1));
    return 0;
}