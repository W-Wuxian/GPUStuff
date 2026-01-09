/*
 * A is M x N
 * B is N x K
 * C is M x K
 * C = A x B
 * All matrices are stored in row-major format.
 */
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

__global__ void matrix_multiplication(const float* A, const float* B, float* C, int M, int N, int K) {
    int intra_block = threadIdx.x + blockDim.x * threadIdx.y;
    int block_offset = blockDim.x * blockDim.y;
    int inter_grid = blockIdx.x + gridDim.x * blockIdx.y;
    int ID = intra_block + block_offset * inter_grid;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (row < M){
        if (col < K){
            float dotproduct = (float)0.0;
            for(int o = 0; o < N; ++o){
                dotproduct += A[o + row * N] * B[o * K + col];
            }
            C[col + row * K] = dotproduct;
        }
    }
    /*if (ID < N){
        //printf("ID %d \n", ID);
        C[ID] = A[ID] + B[ID];
    }*/
}

void init_rand(void* A, int N) {
    for (int i = 0; i < N; ++i){
        float x = ((float)rand()/(float)(RAND_MAX)) * N;
        ((float*)A)[i] = x;
    }
}

int main(){

    srand(time(0));
    int M = rand() % (MAX_LEN - MIN_LEN + 1) + MIN_LEN;
    int N = rand() % (MAX_LEN - MIN_LEN + 1) + MIN_LEN;
    int K = rand() % (MAX_LEN - MIN_LEN + 1) + MIN_LEN;
    size_t ASize = sizeof(float) * M * N;
    size_t BSize = sizeof(float) * N * K;
    size_t CSize = sizeof(float) * M * K;

    // Allocate and initialize arrays on the Host
    float* A = (float*) malloc(ASize);
    float* B = (float*) malloc(BSize);
    float* C = (float*) malloc(CSize);
    init_rand(A, M * N);
    init_rand(B, N * K);
    memset(C, 0, CSize);

    // Allocate and initialize arrays on the device
    // The initialization step is a copy from host to device
    float* dA=NULL;
    float* dB=NULL;
    float* dC=NULL;
    cudaMalloc((void**)&dA, ASize);
    cudaMalloc((void**)&dB, BSize);
    cudaMalloc((void**)&dC, CSize);
    cudaMemcpy(dA, A, ASize, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, BSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, CSize, cudaMemcpyHostToDevice);

    // Perform the vector add on the device
    dim3 threadsPerBlock(5, 5);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (K + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_multiplication<<<numBlocks, threadsPerBlock>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    // copy back the result from device to host
    cudaMemcpy(C, dC, CSize, cudaMemcpyDeviceToHost);
    for(int i = 0 ; i < M; i++){
        int offset = i * K;
        for (int j = 0; j < K; ++j) {
            printf("%f\t", C[j + offset]);
        }
        printf("\n");
    }
    free(A);
    free(B);
    free(C);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
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
