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
    for (int i=0; i<100; i++){
        for (int j=0; j<100; j++){
            m->m[i*100+j] = 1;
        }
    }
    m->columns = 100;
    m->rows = 100;
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

    matrix_dot_kernel <<< gridDim, blockDim >>> (d1, d2, dres, m1->rows, m1->columns, m2->columns);

    cudaMemcpy(res->m, dres, res->rows * res->columns * sizeof(double), cudaMemcpyDeviceToHost);

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

    printf("m[0] = %d\n", (int)m1->m[0]);
    printf("m[99] = %d\n", (int)m1->m[99]);
    printf("m[9600] = %d\n", (int)m1->m[9600]);

    matrix_dot(m1,m2,m3);

    for (int i =0; i<201; i=i+1){
        printf("m[%d] = %d\n", i, (int)m3->m[i]);
    }


    /*printf("m[0] = %d\n", (int)m3->m[0]);
    printf("m[99] = %d\n", (int)m3->m[99]);
    printf("m[9600] = %d\n", (int)m3->m[9600]);*/
    return 0;
}