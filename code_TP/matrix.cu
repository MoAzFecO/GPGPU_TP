#include "matrix.h"
#include <stdlib.h>
#include <string.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

matrix_t * alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
    res->m = (double *) calloc(columns * rows, sizeof(double));
    res->columns = columns;
    res->rows = rows;
    return res;
}

void destroy_matrix(matrix_t *m)
{
    //printf("free %p %p\n", m, m->m);
    free(m->m);
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

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
            res->m[idx] = m1->m[idx] * m2->m[idx];
    }
}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    { 
        res->m[idx] = m1->m[idx] + m2->m[idx];
    }
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));
             
    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = m1->m[idx] - m2->m[idx];
    }
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

    if (0 /*m1->rows*m2->columns < 310*310*/) {
        for (int row = 0; row < m1->rows; row ++){
            for (int col = 0; col < m2->columns; col ++)
            {
                int idx = col + row * m2->columns;
                double var = 0.0;

                for (int ii = 0; ii < m1->columns; ii++) {
                    var += m1->m[ii + row * m1->columns] * m2->m[col + ii * m2->columns];
                }

                res->m[idx] = var;
            }
        }
    }
    else{  
        double *d1;
        double *d2;
        double *dres;

        cudaMalloc((void **)&d1,  m1->rows * m1->columns * sizeof(double));
        cudaMalloc((void **)&d2, m2->rows * m2->columns * sizeof(double));
        cudaMalloc((void **)&dres, res->rows * res->columns * sizeof(double));

        cudaMemcpy(d1, m1->m, m1->rows * m1->columns * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d2, m2->m, m2->rows * m2->columns * sizeof(double), cudaMemcpyHostToDevice);
    
        dim3 blockDim(16, 16);
        dim3 gridDim(ceil(((double)m2->columns) / blockDim.x), ceil(((double)m1->rows) / blockDim.y));

        matrix_dot_shared_memory <<< gridDim, blockDim >>> (d1, d2, dres, m1->rows, m1->columns, m2->columns);

        cudaMemcpy(res->m, dres, res->rows * res->columns * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d1);
        cudaFree(d2);
        cudaFree(dres);
        }
}

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = f(m1->m[idx]);
    }
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));
    
    for (int row = 0; row < m1->rows; row++)
    {
        for (int col = 0; col < m1->columns; col ++)
        {
            res->m[row + col * m1->rows] = m1->m[col + row * m1->columns];
        }
    }
}

void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns*m1->rows; idx ++)
    {
        res->m[idx] = m1->m[idx] * s;
    }
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));     
}