#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cblas.h"

#define MYTYPE float
#define MIN_LEN 1
#define MAX_LEN 10
#define TILE_LEN 16

int genrangerandint(int max_len, int min_len){
    return (rand() % (max_len - min_len + 1)) + min_len;
}

void genrandmat(int (*fptr)(int, int), MYTYPE* mat, int nrow, int ncol){
    size_t nelems = nrow*ncol;
    for (size_t i = 0; i < nelems; ++i) {
        mat[i] = (MYTYPE)fptr(MAX_LEN, MIN_LEN);
    }
}

void showmat(MYTYPE* mat, int nrow, int ncol, const char mat_name[]){
    printf("Matrice %s is %d x %d\n", mat_name, nrow, ncol);
    for(int i=0; i<nrow; i++){
        for(int j=0; j<ncol; j++){
            printf("%f\t", mat[i*ncol+j]);
        }
        printf("\n");
    }
}

// C is M x K Row-major
__global__ void matmul(int M, int N, int K, MYTYPE* deviceA, MYTYPE* deviceB, MYTYPE* deviceC){
    const int col = threadIdx.x + blockIdx.x * blockDim.x; // column ID (grid level)
    const int row = threadIdx.y + blockIdx.y * blockDim.y; // row ID (grid level)
    //const int GID = col + (gridDim.x * blockDim.x) * row; // global (grid level) thread ID)
    const int MID = col + row * K; // thread ID at matrix level

    if(col < K && row < M){
        MYTYPE dotproduct = (MYTYPE)0;
        for(int n=0; n<N; ++n){
            dotproduct += deviceA[n + row * N] * deviceB[n * K + col];
        }
        deviceC[MID] = dotproduct;
    }
}

// C is M x K Row-major
__global__ void tiled_matmul(int M, int N, int K, MYTYPE* deviceA, MYTYPE* deviceB, MYTYPE* deviceC){
    __shared__ MYTYPE sharedA[TILE_LEN][TILE_LEN];
    __shared__ MYTYPE sharedB[TILE_LEN][TILE_LEN];
    const int col = threadIdx.x + blockIdx.x * blockDim.x; // column ID (grid level)
    const int row = threadIdx.y + blockIdx.y * blockDim.y; // row ID (grid level)
    //const int GID = col + (gridDim.x * blockDim.x) * row; // global (grid level) thread ID)
    const int MID = col + row * K; // thread ID at matrix level

    const int local_tile_col = threadIdx.x;
    const int local_tile_row = threadIdx.y;

    const int numtile = (N + (TILE_LEN - 1)) / TILE_LEN;

    MYTYPE tiled_dotproduct = (MYTYPE)0;

    for(int tile_idx=0; tile_idx<numtile; ++tile_idx){
        const int tile_offset = tile_idx * TILE_LEN;
        // load sharedA with A tile by tile
        const int A_col_idx = tile_offset + local_tile_col;
        if(row < M && A_col_idx < N){
            sharedA[local_tile_row][local_tile_col]=deviceA[A_col_idx + row * N];
        }
        // load sharedB with B tile by tile
        const int B_row_idx = tile_offset + local_tile_row;
        if(B_row_idx < N && col < K){
            sharedB[local_tile_row][local_tile_col]=deviceB[col + B_row_idx * K];
        }
        __syncthreads();
        // compute tiled dotproduct
        for(int t=0; t<TILE_LEN; ++t){
            tiled_dotproduct += sharedA[local_tile_row][t] * sharedB[t][local_tile_col];
        }
        __syncthreads();
    }
    if(row < M && col < K){
        deviceC[MID] = tiled_dotproduct;
    }
}

/*
 * Row-major
 * A is M x N
 * B is N x K
 * C is M x K
 */
int main(){
    srand(time(0));

    int M = genrangerandint(MAX_LEN, MIN_LEN);
    int N = genrangerandint(MAX_LEN, MIN_LEN);
    size_t SizeA = sizeof(MYTYPE) * M * N;
    size_t SizeTransA = sizeof(MYTYPE) * N * M;

    MYTYPE* A = (MYTYPE*) malloc(SizeA);
    MYTYPE* TransA = (MYTYPE*) malloc(SizeTransA);
    MYTYPE* HostRef = (MYTYPE*) malloc(SizeTransA);

    MYTYPE* dA = NULL;
    MYTYPE* dTransA = NULL;
    cudaMalloc((void**)&dA, SizeA);
    cudaMalloc((void**)&dTransA, SizeTransA);

    genrandmat(genrangerandint, A, M, N);
    memset(TransA, 0, SizeTransA);
    memset(HostRef, 0, SizeTransA);

    printf("M=%d N=%d\n", M,N);
    showmat(A, M, N, "A");
    showmat(TransA, N, M, "TransA");
    showmat(HostRef, N, M, "HostRef TransA");

    cblas_somatcopy(CblasRowMajor, CblasTrans, M, N, (MYTYPE)1.0, A, N, HostRef, M);
    showmat(HostRef, N, M, "HostRef C");
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            if(A[j+i*N]!=HostRef[i+j*M])printf("Missmatch transpose A[%d][%d]!=TranA[%d][%d]\n",i,j,j,i);
        }
    }

    cudaMemcpy(dA, A, SizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dTransA, TransA, SizeTransA, cudaMemcpyHostToDevice);

    dim3 BD(TILE_LEN,TILE_LEN); // blockDim is the number of threads per block in each direction TILE_LEN x TILE_LEN
    dim3 GD((N + BD.x - 1) / BD.x, (M + BD.y - 1) / BD.y); // gridDim is the number of block in each direction

    //matmul<<<GD, BD>>>(M, N, K, dA, dB, dC);
    //cudaDeviceSynchronize();
    cudaMemcpy(TransA, dTransA, SizeTransA, cudaMemcpyDeviceToHost);
    showmat(TransA, N, M, "device TransA");

    cudaFree(dA);dA = NULL;
    cudaFree(dTransA);dTransA = NULL;
    free(A);A = NULL;
    free(TransA);TransA = NULL;
    free(HostRef);HostRef = NULL;
    return 0;
}
