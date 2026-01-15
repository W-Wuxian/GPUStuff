#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cblas.h"

#define MIN_LEN 1
#define MAX_LEN 8192
//#define MAX_LEN 10
#define MAX_LEN_TOBESHOW 10
#define TILE_LEN 4

#ifdef USE_FLOAT
    #define MYTYPE float
    #define MYTYPE_FLOAT  // Active le bon cas
    #define cblas_xgemm cblas_sgemm
    #define cblas_xcopy cblas_scopy
    #define cblas_xaxpy cblas_saxpy
    #define cblas_xomatcopy cblas_somatcopy
#elif USE_DOUBLE
    #define MYTYPE double
    #define MYTYPE_DOUBLE // Active le bon cas
    #define cblas_xgemm cblas_dgemm
    #define cblas_xcopy cblas_dcopy
    #define cblas_xaxpy cblas_daxpy
    #define cblas_xomatcopy cblas_domatcopy
#else
    #define MYTYPE double  // Default: Double
    #define cblas_xgemm cblas_dgemm
    #define cblas_xcopy cblas_dcopy
    #define cblas_xaxpy cblas_daxpy
    #define cblas_xomatcopy cblas_domatcopy
#endif

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

void compareHostDevice(const MYTYPE* Host, const MYTYPE* Device, const int rows, const int cols){
    // Type-specific tolerances (scale ~5-10× epsilon)
    #if defined(MYTYPE_FLOAT)
        const MYTYPE tol = 1e-6f;  // ~10× FLT_EPSILON
    #elif defined(MYTYPE_DOUBLE)
        const MYTYPE tol = 1e-14;  // ~5× DBL_EPSILON
    #else
        const MYTYPE tol = 1e-12;
    #endif
    for(int i = 0 ; i < rows; i++){
        int offset = i * cols;
        for (int j = 0; j < cols; ++j) {
            int k=j + offset;
            if (fabs(Host[k] - Device[k]) > tol * (fabs(Host[k]) + fabs(Device[k]) + 1e-15)) {
                printf("mismatch (%d,%d): host=%.17g device=%.17g rel_err=%g\n",
                       i, j, Host[k], Device[k], fabs(Host[k]-Device[k]) / (fabs(Host[k])+1e-15));
            }
        }
    }
}
void checktranspose(const MYTYPE* input, MYTYPE* output, int rows, int cols){
    // Type-specific tolerances (scale ~5-10× epsilon)
    #if defined(MYTYPE_FLOAT)
        const MYTYPE tol = 1e-6f;  // ~10× FLT_EPSILON
    #elif defined(MYTYPE_DOUBLE)
        const MYTYPE tol = 1e-14;  // ~5× DBL_EPSILON
    #else
        const MYTYPE tol = 1e-12;
    #endif
    for(int i=0; i<rows; i++){
        int in_offset = i * cols;
        for(int j=0; j<cols; j++){
            int in_ID = in_offset + j;
            int out_offset = j*rows;
            int out_ID = out_offset + i;
            if (fabs(input[in_ID] - output[out_ID]) > tol * (fabs(input[in_ID]) + fabs(output[out_ID]) + 1e-15)) {
                printf("mismatch (%d,%d): host=%.17g device=%.17g rel_err=%g\n",
                       i, j, input[in_ID], output[out_ID], fabs(input[in_ID]-output[out_ID]) / (fabs(input[in_ID])+1e-15));
            }
            //if(input[j+i*cols]!=output[i+j*rows])printf("Missmatch transpose A[%d][%d]!=TranA[%d][%d]\n",i,j,j,i);
        }
    }
}

// input rows x cols Row-major
__global__ void matrix_transpose_kernel(const MYTYPE* input, MYTYPE* output, int rows, int cols){
    const int col = threadIdx.x + blockIdx.x * blockDim.x; // column ID (grid level)
    const int row = threadIdx.y + blockIdx.y * blockDim.y; // row ID (grid level)
    //const int GID = col + (gridDim.x * blockDim.x) * row; // global (grid level) thread ID)
    const int in_ID = col + row * cols; // thread ID at input matrix level
    const int out_ID = row + col * rows; // thread ID at output matrix level

    if(col < cols && row < rows){
        output[out_ID] = input[in_ID];
    }
}

// input rows x cols Row-major
__global__ void tiled_matrix_transpose_kernel(const MYTYPE* input, MYTYPE* output, int rows, int cols){

    __shared__ MYTYPE shared_input[TILE_LEN][TILE_LEN+1];
    const int in_x = threadIdx.x + blockIdx.x * blockDim.x; // column ID (grid level)
    const int in_y = threadIdx.y + blockIdx.y * blockDim.y; // row ID (grid level)
    const int in_cols = cols;
    const int in_rows = rows;
    const int in_ID = in_x + in_y * in_cols;

    const int out_x = threadIdx.x + blockIdx.y * blockDim.y; // swap offset
    const int out_y = threadIdx.y + blockIdx.x * blockDim.x;
    const int out_cols = rows; // transposed dim
    const int out_rows = cols;
    const int out_ID = out_x + out_y * out_cols;


    const int local_tile_col = threadIdx.x;
    const int local_tile_row = threadIdx.y;

    if(in_ID < in_rows * in_cols){
        shared_input[local_tile_row][local_tile_col] = input[in_ID]; // copy from global to shared memory
    }
    __syncthreads();

    if(out_x < out_cols && out_y < out_rows){
        output[out_ID] = shared_input[local_tile_col][local_tile_row];
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
    #if defined(FIXSEED)
    #define WSRAND
    srand(FIXSEED);
    #elif defined(RANDSEED)
    #define WSRAND
    srand(time(0));
    #endif

    int M = MAX_LEN, N=MAX_LEN;
    #if defined(WSRAND)
    M = genrangerandint(MAX_LEN, MIN_LEN);
    N = genrangerandint(MAX_LEN, MIN_LEN);
    #endif
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

    printf("A is M=%d x N=%d\n", M,N);
    if(M<=MAX_LEN_TOBESHOW && N<=MAX_LEN_TOBESHOW)showmat(A, M, N, "A");
    struct timespec start_cpu, end_cpu;
    clock_gettime(CLOCK_MONOTONIC, &start_cpu);
    cblas_xomatcopy(CblasRowMajor, CblasTrans, M, N, (MYTYPE)1.0, A, N, HostRef, M);
    clock_gettime(CLOCK_MONOTONIC, &end_cpu);
    double cpu_time_ms = (end_cpu.tv_sec - start_cpu.tv_sec) * 1000.0 + (end_cpu.tv_nsec - start_cpu.tv_nsec) / 1e6;
    if(M<=MAX_LEN_TOBESHOW && N<=MAX_LEN_TOBESHOW)showmat(HostRef, N, M, "HostRef Transpose(A)");
    checktranspose(A, HostRef, M, N);

    cudaMemcpy(dA, A, SizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dTransA, TransA, SizeTransA, cudaMemcpyHostToDevice);

    dim3 BD(TILE_LEN,TILE_LEN); // blockDim is the number of threads per block in each direction TILE_LEN x TILE_LEN
    dim3 GD((N + BD.x - 1) / BD.x, (M + BD.y - 1) / BD.y); // gridDim is the number of block in each direction

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrix_transpose_kernel<<<GD, BD>>>(dA, dTransA, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(TransA, dTransA, SizeTransA, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel1_time_ms;
    cudaEventElapsedTime(&kernel1_time_ms, start, stop);
    if(M<=MAX_LEN_TOBESHOW && N<=MAX_LEN_TOBESHOW)showmat(TransA, N, M, "device Transpose(A)");
    checktranspose(A, TransA, M, N);

    cudaEventRecord(start);
    tiled_matrix_transpose_kernel<<<GD, BD>>>(dA, dTransA, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(TransA, dTransA, SizeTransA, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel2_time_ms;
    cudaEventElapsedTime(&kernel2_time_ms, start, stop);
    if(M<=MAX_LEN_TOBESHOW && N<=MAX_LEN_TOBESHOW)showmat(TransA, N, M, "device Transpose(A)");
    checktranspose(A, TransA, M, N);

    printf("CPU (BLAS) time: %.3f ms\n", cpu_time_ms);
    printf("Simple kernel time: %.3f ms\n", kernel1_time_ms);
    printf("Tiled kernel time: %.3f ms\n", kernel2_time_ms);
    printf("Tiled vs CPU speedup: %.2fx\n", cpu_time_ms / kernel2_time_ms);
    printf("Simple vs CPU speedup: %.2fx\n", cpu_time_ms / kernel1_time_ms);
    printf("Tiled vs Simple speedup: %.2fx\n", kernel1_time_ms / kernel2_time_ms);

    cudaFree(dA);dA = NULL;
    cudaFree(dTransA);dTransA = NULL;
    free(A);A = NULL;
    free(TransA);TransA = NULL;
    free(HostRef);HostRef = NULL;
    return 0;
}
