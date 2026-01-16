#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cblas.h"
#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // Required for RUSAGE_THREAD on Linux
#endif
#include <sys/resource.h>  // struct rusage, getrusage(), RUSAGE_THREAD
#include <sys/time.h>      // struct timeval (ru_utime fields)

#define MIN_LEN 1
#define MAX_LEN 8192
#define MAX_LEN_TOBESHOW 10
#define TILE_LEN 16

#ifdef USE_FLOAT
    #define MYTYPE float
    #define MYTYPE_FLOAT  // Active le bon cas
    #define cblas_xgemm cblas_sgemm
    #define cblas_xcopy cblas_scopy
    #define cblas_xaxpy cblas_saxpy
    #define cblas_xomatcopy cblas_somatcopy
    #define WICHTYPE() printf("Using float\n")
#elif USE_DOUBLE
    #define MYTYPE double
    #define MYTYPE_DOUBLE // Active le bon cas
    #define cblas_xgemm cblas_dgemm
    #define cblas_xcopy cblas_dcopy
    #define cblas_xaxpy cblas_daxpy
    #define cblas_xomatcopy cblas_domatcopy
    #define WICHTYPE() printf("Using double\n")
#else
    #define MYTYPE double  // Default: Double
    #define cblas_xgemm cblas_dgemm
    #define cblas_xcopy cblas_dcopy
    #define cblas_xaxpy cblas_daxpy
    #define cblas_xomatcopy cblas_domatcopy
    #define WICHTYPE() printf("Using double\n")
#endif

//https://gist.github.com/raytroop/120e2d175d95f82edbee436374293420
//https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#ifdef WDEBUG
#define gpu_errchk(ans) { gpu_assert((ans), __FILE__, __LINE__); }
#else
#define gpu_errchk(ans) { (ans); }
#endif

inline void gpu_assert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "gpu_assert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

 #define MAX(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define MIN(a,b) \
({ __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
       _a < _b ? _a : _b; })


int genrangerandint(int max_len, int min_len){
    return (rand() % (max_len - min_len + 1)) + min_len;
}

void genrandmat(int (*fptr)(int, int), MYTYPE* mat, int nrow, int ncol, int r1, int r2){
    size_t nelems = nrow*ncol;
    for (size_t i = 0; i < nelems; ++i) {
        mat[i] = (MYTYPE)fptr(MAX(r1,r2), MIN(r1, r2));
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

void compareHostDevice(const MYTYPE* Host, const MYTYPE* Device, const int rows, const int cols){
    int mismatches = 0;
    // Type-specific tolerances (scale ~5-10× epsilon)
    #if defined(MYTYPE_FLOAT)
        const MYTYPE rtol = 1e-4f;//1e-6f;  // ~10× FLT_EPSILON
        const MYTYPE atol = 1e-6f;//1e-6f;  // ~10× FLT_EPSILON
    #elif defined(MYTYPE_DOUBLE)
        const MYTYPE rtol = 1e-6;//1e-14;  // ~5× DBL_EPSILON
        const MYTYPE atol = 1e-12;//1e-14;  // ~5× DBL_EPSILON
    #else
        const MYTYPE rtol = 1e-6;//1e-12;
        const MYTYPE atol = 1e-12;
    #endif
    for(int i = 0 ; i < rows; i++){
        int offset = i * cols;
        for (int j = 0; j < cols; ++j) {
            int k=j + offset;
            if (fabs(Host[k] - Device[k]) > rtol * (fabs(Host[k]) + fabs(Device[k]) + atol)) {
                printf("mismatch (%d,%d): host=%.17g device=%.17g rel_err=%g\n",
                       i, j, Host[k], Device[k], fabs(Host[k]-Device[k]) / (fabs(Host[k])+atol));
                mismatches++;
                if(mismatches >= 10) { printf("ABORT: >10 mismatches\n"); return; }
            }
        }
    }
    printf("%s: %d mismatches\n", mismatches ? "FAILED" : "PASSED", mismatches);
}

// C is M x K Row-major
__global__ void matmul(int M, int N, int K, const MYTYPE* __restrict__ deviceA, const MYTYPE* __restrict__ deviceB, MYTYPE* __restrict__ deviceC){
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
__global__ void tiled_matmul(int M, int N, int K, const MYTYPE* __restrict__ deviceA, const MYTYPE* __restrict__ deviceB, MYTYPE* __restrict__ deviceC){
    __shared__ MYTYPE sharedA[TILE_LEN][TILE_LEN+1];
    __shared__ MYTYPE sharedB[TILE_LEN][TILE_LEN+1];
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
        } else {
            sharedA[local_tile_row][local_tile_col]=(MYTYPE)0;
        }
        // load sharedB with B tile by tile
        const int B_row_idx = tile_offset + local_tile_row;
        if(B_row_idx < N && col < K){
            sharedB[local_tile_row][local_tile_col]=deviceB[col + B_row_idx * K];
        } else {
            sharedB[local_tile_row][local_tile_col]=(MYTYPE)0;
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
    #if defined(FIXSEED)
    #define WSRAND
    srand(FIXSEED);
    #elif defined(RANDSEED)
    #define WSRAND
    srand(time(0));
    #endif

    int M=MAX_LEN, N=MAX_LEN, K=MAX_LEN;
    #if defined(WSRAND)
    M = genrangerandint(MAX_LEN, MIN_LEN);
    K = genrangerandint(MAX_LEN, MIN_LEN);
    N = genrangerandint(MAX_LEN, MIN_LEN);
    #endif


    size_t SizeA = sizeof(MYTYPE) * M * N;
    size_t SizeB = sizeof(MYTYPE) * N * K;
    size_t SizeC = sizeof(MYTYPE) * M * K;

    MYTYPE* A = (MYTYPE*) malloc(SizeA);
    MYTYPE* B = (MYTYPE*) malloc(SizeB);
    MYTYPE* C = (MYTYPE*) malloc(SizeC);
    MYTYPE* HostRef = (MYTYPE*) malloc(SizeC);

    MYTYPE* dA = NULL;
    MYTYPE* dB = NULL;
    MYTYPE* dC = NULL;
    gpu_errchk(cudaMalloc((void**)&dA, SizeA));
    gpu_errchk(cudaMalloc((void**)&dB, SizeB));
    gpu_errchk(cudaMalloc((void**)&dC, SizeC));

    genrandmat(genrangerandint, A, M, N, MAX_LEN, 1);
    genrandmat(genrangerandint, B, N, K, MAX_LEN, 1);
    memset(C, 0, SizeC);
    memset(HostRef, 0, SizeC);

    printf("M=%d N=%d K=%d\n", M,N,K);
    if(M<=MAX_LEN_TOBESHOW && N<=MAX_LEN_TOBESHOW)showmat(A, M, N, "A");
    if(N<=MAX_LEN_TOBESHOW && K<=MAX_LEN_TOBESHOW)showmat(B, N, K, "B");
    if(M<=MAX_LEN_TOBESHOW && K<=MAX_LEN_TOBESHOW)showmat(C, M, K, "C");
    if(M<=MAX_LEN_TOBESHOW && K<=MAX_LEN_TOBESHOW)showmat(HostRef, M, K, "HostRef C");

    MYTYPE alpha = (MYTYPE)1.0;
    MYTYPE beta = (MYTYPE)0.0;
    openblas_set_num_threads(1);
    struct rusage start_rusage, end_rusage;
    getrusage(RUSAGE_THREAD, &start_rusage);
    cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
        M, K, N, \
        alpha, A, N, \
        B,K, beta, HostRef, K);
    getrusage(RUSAGE_THREAD, &end_rusage);
    double cpu_user_s = (end_rusage.ru_utime.tv_sec - start_rusage.ru_utime.tv_sec) +
                        (end_rusage.ru_utime.tv_usec - start_rusage.ru_utime.tv_usec)/1e6;
    double cpu_time_ms = cpu_user_s * 1000;
    if(M<=MAX_LEN_TOBESHOW && K<=MAX_LEN_TOBESHOW)showmat(HostRef, M, K, "HostRef C");

    gpu_errchk(cudaMemcpy(dA, A, SizeA, cudaMemcpyHostToDevice));
    gpu_errchk(cudaMemcpy(dB, B, SizeB, cudaMemcpyHostToDevice));
    gpu_errchk(cudaMemcpy(dC, C, SizeC, cudaMemcpyHostToDevice));

    dim3 BD(TILE_LEN,TILE_LEN); // blockDim is the number of threads per block in each direction TILE_LEN x TILE_LEN
    dim3 GD((K + BD.x - 1) / BD.x, (M + BD.y - 1) / BD.y); // gridDim is the number of block in each direction
    cudaEvent_t start, stop;
    gpu_errchk(cudaEventCreate(&start));
    gpu_errchk(cudaEventCreate(&stop));
    gpu_errchk(cudaEventRecord(start));
    matmul<<<GD, BD>>>(M, N, K, dA, dB, dC);
    gpu_errchk(cudaGetLastError());
    gpu_errchk(cudaDeviceSynchronize());
    gpu_errchk(cudaMemcpy(C, dC, SizeC, cudaMemcpyDeviceToHost));
    gpu_errchk(cudaEventRecord(stop));
    gpu_errchk(cudaEventSynchronize(stop));
    float kernel1_time_ms;
    cudaEventElapsedTime(&kernel1_time_ms, start, stop);
    compareHostDevice(HostRef, C, M, K);
    if(M<=MAX_LEN_TOBESHOW && K<=MAX_LEN_TOBESHOW)showmat(C, M, K, "device C using matmul kernel");

    memset(C, 0, SizeC);
    gpu_errchk(cudaMemcpy(dC, C, SizeC, cudaMemcpyHostToDevice));
    gpu_errchk(cudaEventRecord(start));
    tiled_matmul<<<GD, BD>>>(M, N, K, dA, dB, dC);
    gpu_errchk(cudaGetLastError());
    gpu_errchk(cudaDeviceSynchronize());
    gpu_errchk(cudaMemcpy(C, dC, SizeC, cudaMemcpyDeviceToHost));
    gpu_errchk(cudaEventRecord(stop));
    gpu_errchk(cudaEventSynchronize(stop));
    float kernel2_time_ms;
    cudaEventElapsedTime(&kernel2_time_ms, start, stop);
    compareHostDevice(HostRef, C, M, K);
    if(M<=MAX_LEN_TOBESHOW && K<=MAX_LEN_TOBESHOW)showmat(C, M, K, "device C using tiled_matmul kernel");

    WICHTYPE();
    #ifdef WSRAND
    printf("With random seed\n");
    #endif
    printf("M=%d N=%d K=%d\n", M,N,K);
    printf("CPU blas xgemm kernel time: %f ms\n", cpu_time_ms);
    printf("Simple matmul kernel time: %.3f ms\n", kernel1_time_ms);
    printf("Tiled matmul kernel time: %.3f ms\n", kernel2_time_ms);
    printf("Tiled vs CPU speedup: %.2fx\n", cpu_time_ms / kernel2_time_ms);
    printf("Simple vs CPU speedup: %.2fx\n", cpu_time_ms / kernel1_time_ms);
    printf("Tiled vs Simple speedup: %.2fx\n", kernel1_time_ms / kernel2_time_ms);

    gpu_errchk(cudaFree(dA));dA = NULL;
    gpu_errchk(cudaFree(dB));dB = NULL;
    gpu_errchk(cudaFree(dC));dC = NULL;
    free(A);A = NULL;
    free(B);B = NULL;
    free(C);C = NULL;
    free(HostRef);HostRef = NULL;
    return 0;
}
