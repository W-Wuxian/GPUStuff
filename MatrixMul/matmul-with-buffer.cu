/*
 * A is M x N
 * B is N x K
 * C is M x K
 * C = A x B
 * All matrices are stored in row-major format.
 */
#include <cstddef>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
//#include <sys/time.h>
//#include <math.h>
//#include <omp.h>
//#include <cuda.h>
#include <cuda_runtime.h>
#include <cblas.h>
//#include <cuda_profiler_api.h>
#define MIN_LEN 1
#define MAX_LEN 10
//#define MAX_LEN 8192
#define MYTYPE double

__global__ void matrix_multiplication(const MYTYPE* A, const MYTYPE* B, MYTYPE* C, int M, int N, int K) {
    // int intra_block = threadIdx.x + blockDim.x * threadIdx.y;
    // int block_offset = blockDim.x * blockDim.y;
    // int inter_grid = blockIdx.x + gridDim.x * blockIdx.y;
    // int ID = intra_block + block_offset * inter_grid;

    extern __shared__ MYTYPE buf[];
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int block_col = blockIdx.x * blockDim.x;
    if(tid < N && block_col < K){
        buf[tid] = B[tid * K + block_col];
    }
    __syncthreads();

    if (row < M){
        if (col < K){
            MYTYPE dotproduct = (MYTYPE)0.0;
            for(int o = 0; o < N; ++o){
                dotproduct += A[o + row * N] * buf[o];
            }
            C[col + row * K] = dotproduct;
        }
    }
}

void init_rand(void* A, int N) {
    for (int i = 0; i < N; ++i){
        MYTYPE x = ((MYTYPE)rand()/(MYTYPE)(RAND_MAX)) * N;
        ((MYTYPE*)A)[i] = x;
    }
}

int main(){

    srand(time(0));
    int M = rand() % (MAX_LEN - MIN_LEN + 1) + MIN_LEN;
    int N = rand() % (MAX_LEN - MIN_LEN + 1) + MIN_LEN;
    int K = rand() % (MAX_LEN - MIN_LEN + 1) + MIN_LEN;
    size_t ASize = sizeof(MYTYPE) * M * N;
    size_t BSize = sizeof(MYTYPE) * N * K;
    size_t CSize = sizeof(MYTYPE) * M * K;
    size_t Ns = sizeof(MYTYPE) * N; /*< specifies the number of bytes in shared memory that is dynamcally allocated per block */

    // Allocate and initialize arrays on the Host
    MYTYPE* A = (MYTYPE*) malloc(ASize);
    MYTYPE* B = (MYTYPE*) malloc(BSize);
    MYTYPE* C = (MYTYPE*) malloc(CSize);
    init_rand(A, M * N);
    init_rand(B, N * K);
    memset(C, 0, CSize);

    // Allocate and initialize arrays on the device
    // The initialization step is a copy from host to device
    MYTYPE* dA=NULL;
    MYTYPE* dB=NULL;
    MYTYPE* dC=NULL;
    cudaMalloc((void**)&dA, ASize);
    cudaMalloc((void**)&dB, BSize);
    cudaMalloc((void**)&dC, CSize);
    cudaMemcpy(dA, A, ASize, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, BSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, CSize, cudaMemcpyHostToDevice);

    // Perform the vector add on the device
    dim3 threadsPerBlock(4, 4);
    dim3 numBlocks((K + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_multiplication<<<numBlocks, threadsPerBlock, Ns>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    // copy back the result from device to host
    cudaMemcpy(C, dC, CSize, cudaMemcpyDeviceToHost);
    printf("M=%d N=%d K=%d:\n", M, N, K);
    printf("Result on Device:\n");
    for(int i = 0 ; i < M; i++){
        int offset = i * K;
        for (int j = 0; j < K; ++j) {
            printf("%f\t", C[j + offset]);
        }
        printf("\n");
    }

    memset(C, 0, CSize);
    MYTYPE alpha = 1.0;
    MYTYPE beta = 1.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, alpha, A, N, B, K, beta, C, K);
    printf("Result on Host:\n");
    for(int i = 0 ; i < M; i++){
        int offset = i * K;
        for (int j = 0; j < K; ++j) {
            printf("%f\t", C[j + offset]);
        }
        printf("\n");
    }

    free(A);A=NULL;
    free(B);B=NULL;
    free(C);C=NULL;
    cudaFree(dA);dA=NULL;
    cudaFree(dB);dB=NULL;
    cudaFree(dC);dC=NULL;
    return 0;
}



// fichier : hello_cuda.cu

/* __global__ void helloFromGPU() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    helloFromGPU<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
} */
