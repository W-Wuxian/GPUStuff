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
//#define MAX_LEN 100
#define MAX_LEN 8192
#define MYTYPE double
#define TILE_LEN 16

//https://gist.github.com/raytroop/120e2d175d95f82edbee436374293420
//https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpu_errchk(ans) { gpu_assert((ans), __FILE__, __LINE__); }

inline void gpu_assert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "gpu_assert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

//https://gitlostmurali.com/machine-learning/data-science/cuda-matmul-2
/*
 * A is M x N    B is N x K     C is M x K
 * C = A x B
 * All matrices are stored in row-major format.
 */
__global__ void matrix_multiplication(const MYTYPE* A, const MYTYPE* B, MYTYPE* C, int M, int N, int K) {
    __shared__ MYTYPE AS[TILE_LEN][TILE_LEN]; // Shared memory buffer is TILE_LEN x TILE_LEN
    __shared__ MYTYPE BS[TILE_LEN][TILE_LEN];

    const int col = threadIdx.x + blockIdx.x * blockDim.x; // global col index
    const int row = threadIdx.y + blockIdx.y * blockDim.y; // global row index
    const int ID = col + row * K; // Global ID

    const int tile_local_col = threadIdx.x; // local col thread index
    const int tile_local_row = threadIdx.y; // local row thread index

    const int num_tile = (N + (TILE_LEN - 1)) / TILE_LEN;

    MYTYPE tiled_dotproduct = (MYTYPE)0;

    for (int tile_idx = 0; tile_idx < num_tile; ++tile_idx){
        const int tile_offset = tile_idx * TILE_LEN;

        // Loading A into shared memory buffer AS
        const int a_col_idx = tile_offset + tile_local_col;
        if (row < M && a_col_idx < N){
            AS[tile_local_row][tile_local_col] = A[row * N + a_col_idx];
        } else {
            AS[tile_local_row][tile_local_col] = (MYTYPE) 0;
        }

        // Loading B into shared memory buffer BS
        const int b_row_idx = tile_offset + tile_local_row;
        if (col < K && b_row_idx < N){
            BS[tile_local_row][tile_local_col] = B[ b_row_idx * K + col];
        } else {
            BS[tile_local_row][tile_local_col] = (MYTYPE)0;
        }

        __syncthreads();

        // tiled dotproduct
        for(int n=0; n<TILE_LEN; ++n){
            tiled_dotproduct += AS[tile_local_row][n] * BS[n][tile_local_col];
        }

        __syncthreads();
    }

    if(row < M && col < K){
        C[ID] = tiled_dotproduct;
        //C[row * K + col] = tiled_dotproduct;
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

    // Allocate and initialize arrays on the Host
    MYTYPE* A = (MYTYPE*) malloc(ASize);
    MYTYPE* B = (MYTYPE*) malloc(BSize);
    MYTYPE* C = (MYTYPE*) malloc(CSize);
    MYTYPE* CC = (MYTYPE*) malloc(CSize);
    init_rand(A, M * N);
    init_rand(B, N * K);
    memset(C, 0, CSize);
    memset(CC, 0, CSize);

    // Allocate and initialize arrays on the device
    // The initialization step is a copy from host to device
    MYTYPE* dA=NULL;
    MYTYPE* dB=NULL;
    MYTYPE* dC=NULL;
    gpu_errchk(cudaMalloc((void**)&dA, ASize));
    gpu_errchk(cudaMalloc((void**)&dB, BSize));
    gpu_errchk(cudaMalloc((void**)&dC, CSize));
    gpu_errchk(cudaMemcpy(dA, A, ASize, cudaMemcpyHostToDevice));
    gpu_errchk(cudaMemcpy(dB, B, BSize, cudaMemcpyHostToDevice));
    gpu_errchk(cudaMemcpy(dC, C, CSize, cudaMemcpyHostToDevice));

    // Perform the vector add on the device
    dim3 threadsPerBlock(TILE_LEN, TILE_LEN);
    dim3 numBlocks((K + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_multiplication<<<numBlocks, threadsPerBlock>>>(dA, dB, dC, M, N, K);
    cudaError_t err1 = cudaPeekAtLastError();
    cudaDeviceSynchronize();
    printf("Got CUDA error ... %s \n", cudaGetErrorString(err1));

    // copy back the result from device to host
    gpu_errchk(cudaMemcpy(C, dC, CSize, cudaMemcpyDeviceToHost));
    printf("M=%d N=%d K=%d:\n", M, N, K);
    // printf("Result on Device:\n");
    // for(int i = 0 ; i < M; i++){
    //     int offset = i * K;
    //     for (int j = 0; j < K; ++j) {
    //         printf("%f\t", C[j + offset]);
    //     }
    //     printf("\n");
    // }

    MYTYPE alpha = 1.0;
    MYTYPE beta = 1.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, alpha, A, N, B, K, beta, CC, K);
    // printf("Result on Host:\n");
    // for(int i = 0 ; i < M; i++){
    //     int offset = i * K;
    //     for (int j = 0; j < K; ++j) {
    //         printf("%f\t", CC[j + offset]);
    //     }
    //     printf("\n");
    // }
    const MYTYPE tol = 1e-12;
    for(int i = 0 ; i < M; i++){
        int offset = i * K;
        for (int j = 0; j < K; ++j) {
            int k=j + offset;
            if (fabs(CC[k] - C[k]) > tol * (fabs(CC[k]) + fabs(C[k]) + 1e-15)) {
                printf("mismatch (%d,%d): host=%.17g device=%.17g rel_err=%g\n",
                       i, j, CC[k], C[k], fabs(CC[k]-C[k]) / (fabs(CC[k])+1e-15));
            }
            // if(CC[k] != C[k]){
            //     printf("missmatch value at (%d,%d) host=%f device=%f\n", i,j,CC[k],C[k]);
            // }
        }
    }

    free(A);A=NULL;
    free(B);B=NULL;
    free(C);C=NULL;
    free(CC);CC=NULL;
    gpu_errchk(cudaFree(dA));dA=NULL;
    gpu_errchk(cudaFree(dB));dB=NULL;
    gpu_errchk(cudaFree(dC));dC=NULL;
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
