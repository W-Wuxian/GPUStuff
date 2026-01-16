#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cblas.h"

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
void checktranspose(const MYTYPE* input, MYTYPE* output, int rows, int cols){
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
    for(int i=0; i<rows; i++){
        int in_offset = i * cols;
        for(int j=0; j<cols; j++){
            int in_ID = in_offset + j;
            int out_offset = j*rows;
            int out_ID = out_offset + i;
            if (fabs(input[in_ID] - output[out_ID]) > rtol * (fabs(input[in_ID]) + fabs(output[out_ID]) + atol)) {
                printf("mismatch (%d,%d): host=%.17g device=%.17g rel_err=%g\n",
                       i, j, input[in_ID], output[out_ID], fabs(input[in_ID]-output[out_ID]) / (fabs(input[in_ID])+atol));
                mismatches++;
                if(mismatches >= 10) { printf("ABORT: >10 mismatches\n"); return; }
            }
        }
    }
    printf("%s: %d mismatches\n", mismatches ? "FAILED" : "PASSED", mismatches);
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
    } else {
        shared_input[local_tile_row][local_tile_col] = (MYTYPE)0;
    }
    __syncthreads();

    if(out_x < out_cols && out_y < out_rows){
        output[out_ID] = shared_input[local_tile_col][local_tile_row];
    }

}

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
    gpu_errchk(cudaMalloc((void**)&dA, SizeA));
    gpu_errchk(cudaMalloc((void**)&dTransA, SizeTransA));

    genrandmat(genrangerandint, A, M, N, MAX_LEN, 1);
    memset(TransA, 0, SizeTransA);
    memset(HostRef, 0, SizeTransA);

    printf("A is M=%d x N=%d\n", M,N);
    if(M<=MAX_LEN_TOBESHOW && N<=MAX_LEN_TOBESHOW)showmat(A, M, N, "A");
    openblas_set_num_threads(1);
    MYTYPE alpha = (MYTYPE)1.0;
    struct timespec start_cpu, end_cpu;
    clock_gettime(CLOCK_MONOTONIC, &start_cpu);
    cblas_xomatcopy(CblasRowMajor, CblasTrans, M, N, alpha, A, N, HostRef, M);
    clock_gettime(CLOCK_MONOTONIC, &end_cpu);
    double cpu_time_ms = (end_cpu.tv_sec - start_cpu.tv_sec) * 1000.0 + (end_cpu.tv_nsec - start_cpu.tv_nsec) / 1e6;
    if(M<=MAX_LEN_TOBESHOW && N<=MAX_LEN_TOBESHOW)showmat(HostRef, N, M, "HostRef Transpose(A)");
    checktranspose(A, HostRef, M, N);

    gpu_errchk(cudaMemcpy(dA, A, SizeA, cudaMemcpyHostToDevice));
    gpu_errchk(cudaMemcpy(dTransA, TransA, SizeTransA, cudaMemcpyHostToDevice));

    dim3 BD(TILE_LEN,TILE_LEN); // blockDim is the number of threads per block in each direction TILE_LEN x TILE_LEN
    dim3 GD((N + BD.x - 1) / BD.x, (M + BD.y - 1) / BD.y); // gridDim is the number of block in each direction

    cudaEvent_t start, stop;
    gpu_errchk(cudaEventCreate(&start));
    gpu_errchk(cudaEventCreate(&stop));

    gpu_errchk(cudaEventRecord(start));
    matrix_transpose_kernel<<<GD, BD>>>(dA, dTransA, M, N);
    gpu_errchk(cudaGetLastError());
    gpu_errchk(cudaDeviceSynchronize());
    gpu_errchk(cudaMemcpy(TransA, dTransA, SizeTransA, cudaMemcpyDeviceToHost));
    gpu_errchk(cudaEventRecord(stop));
    gpu_errchk(cudaEventSynchronize(stop));
    float kernel1_time_ms;
    cudaEventElapsedTime(&kernel1_time_ms, start, stop);
    if(M<=MAX_LEN_TOBESHOW && N<=MAX_LEN_TOBESHOW)showmat(TransA, N, M, "device Transpose(A)");
    checktranspose(A, TransA, M, N);

    gpu_errchk(cudaEventRecord(start));
    tiled_matrix_transpose_kernel<<<GD, BD>>>(dA, dTransA, M, N);
    gpu_errchk(cudaGetLastError());
    gpu_errchk(cudaDeviceSynchronize());
    gpu_errchk(cudaMemcpy(TransA, dTransA, SizeTransA, cudaMemcpyDeviceToHost));
    gpu_errchk(cudaEventRecord(stop));
    gpu_errchk(cudaEventSynchronize(stop));
    float kernel2_time_ms;
    cudaEventElapsedTime(&kernel2_time_ms, start, stop);
    if(M<=MAX_LEN_TOBESHOW && N<=MAX_LEN_TOBESHOW)showmat(TransA, N, M, "device Transpose(A)");
    checktranspose(A, TransA, M, N);

    WICHTYPE();
    #ifdef WSRAND
    printf("With random seed\n");
    #endif
    printf("A is M=%d x N=%d\n", M,N);
    printf("CPU (BLAS) time: %.3f ms\n", cpu_time_ms);
    printf("Simple kernel time: %.3f ms\n", kernel1_time_ms);
    printf("Tiled kernel time: %.3f ms\n", kernel2_time_ms);
    printf("Tiled vs CPU speedup: %.2fx\n", cpu_time_ms / kernel2_time_ms);
    printf("Simple vs CPU speedup: %.2fx\n", cpu_time_ms / kernel1_time_ms);
    printf("Tiled vs Simple speedup: %.2fx\n", kernel1_time_ms / kernel2_time_ms);

    gpu_errchk(cudaFree(dA));dA = NULL;
    gpu_errchk(cudaFree(dTransA));dTransA = NULL;
    free(A);A = NULL;
    free(TransA);TransA = NULL;
    free(HostRef);HostRef = NULL;
    return 0;
}
