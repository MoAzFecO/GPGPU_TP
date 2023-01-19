#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>

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
    for (int i=0; i < m->rows; i++){
        for (int j=0; j < m->columns; j++){
            m->m[i * m->columns + j] = 1;
        }
    }
}
const int TILE_WIDTH = 16;

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


__global__
void matrix_mul_kernel (double * A, double * B, double * C, int numARows, int numAColumns, int numBColumns )
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

    matrix_mul_kernel <<< gridDim, blockDim >>> (d1, d2, dres, m1->rows, m1->columns, m2->columns);

    cudaMemcpy(res->m, dres, res->rows * res->columns * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d1);
    cudaFree(d2);
    cudaFree(dres);
}

void matrix_dot2(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));

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

    matrix_dot_kernel <<< gridDim, blockDim >>> (d1, d2, dres, m1->rows, m1->columns, m2->columns);

    cudaMemcpy(res->m, dres, res->rows * res->columns * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d1);
    cudaFree(d2);
    cudaFree(dres);
}

void matrix_dot_cpu(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));

    for (int row = 0; row < m1->rows; row ++)
    {
        for (int col = 0; col < m2->columns; col ++)
        {
            int idx = col + row * m2->columns;
            double var = 0.0;

            for (int ii = 0; ii < m1->columns; ii++)
            {
                var += m1->m[ii + row * m1->columns] * m2->m[col + ii * m2->columns];
            }

            res->m[idx] = var;
        }
    }
}

#include <time.h>

int main(){
    int taille = 310;
    matrix_t *m1 = alloc_matrix(taille, taille);
    matrix_t *m2 = alloc_matrix(taille, taille);
    matrix_t *m3 = alloc_matrix(taille, taille);

    initMatrix(m1);
    initMatrix(m2);

    clock_t start, end;
    start = clock();
    matrix_dot_cpu(m1,m2,m3);
    end=clock();
    double temps = (double)(end-start)/CLOCKS_PER_SEC;
    printf("cpu = %f\n", temps);

    start = clock();
    matrix_dot(m1,m2,m3);
    end=clock();
    temps = (double)(end-start)/CLOCKS_PER_SEC;
    printf("gpu = %f\n", temps);

    free(m1);
    free(m2);
    free(m3);
    return 0;
}