#include "matrix.h"
#include <stdlib.h>
#include <string.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

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

void destroy_matrix(matrix_t *m)
{
    //printf("free %p %p\n", m, m->m);
    cudaFree(m->m);
    free(m);
}

void print_matrix(matrix_t *m, bool is_short){
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row ++)
    {
        for (int col = 0; col < lim_col; col ++)
        {
            printf("%.2lf ", m->m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns) printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows) printf("...\n");
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
void matrix_minus_kernel(double *m1, double *m2, double *res, int m1rows, int m1columns, int m2columns){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    int idx = col + row * m2columns;

    if (idx < m1rows * m1columns) {
        res[idx] = m1[idx] - m2[idx];
    }
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));
             
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((double)m2->columns) / blockDim.x), ceil(((double)m1->rows) / blockDim.y));

    matrix_minus_kernel <<< gridDim, blockDim >>> (m1->m, m2->m, res->m, m1->rows, m1->columns, m2->columns);
}



__global__
void matrix_dot_kernel(double *m1, double *m2, double *res, int m1rows, int m1columns, int m2columns){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < m1rows && col < m2columns) {
        int idx = col + row * m2columns;
        double var = 0.0;
        for (int ii = 0; ii < m1columns; ii++){
            var += m1[ii + row * m1columns] * m2[col + ii * m2columns];
            }
        res[idx] = var;
    }
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

    matrix_function_kernel <<< gridDim, blockDim >>> (f, m1->m, res->m, m1->rows, m1->columns);
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

__global__
void matrix_transpose_kernel(double *m1, double *res, int m1rows, int m1columns){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < m1rows && col < m1columns) {
        res[row + col * m1rows] = m1[col + row * m1columns];
    }
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));
    
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((double)m1->columns) / blockDim.x), ceil(((double)m1->rows) / blockDim.y));

    matrix_transpose_kernel <<< gridDim, blockDim >>> (m1->m, res->m, m1->rows, m1->columns);
}

__global__
void matrix_scalar_kernel(double *m1, double s, double *res, int m1rows, int m1columns){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    int idx = col + row * m1columns;

    if (idx < m1columns * m1rows) {
        res[idx] = m1[idx] * s;
    }
}

void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((double)m1->columns) / blockDim.x), ceil(((double)m1->rows) / blockDim.y));

    matrix_scalar_kernel <<< gridDim, blockDim >>> (m1->m, s, res->m, m1->rows, m1->columns);
}



void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    cudaMemcpy(dest->m, src->m, src->columns * src->rows * sizeof(double), cudaMemcpyDeviceToDevice);     
}