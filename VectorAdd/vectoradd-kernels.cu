/*
 * A is N
 * B is N
 * C is N
 * C = A + B
 */
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cblas.h"
#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // Required for RUSAGE_THREAD on Linux
#endif
#include <sys/resource.h>  // struct rusage, getrusage(), RUSAGE_THREAD
#include <sys/time.h>      // struct timeval (ru_utime fields)


#define MIN_LEN 1
#define MAX_LEN 100000000

#ifdef USE_FLOAT
    #define MYTYPE float
    #define MYTYPE_FLOAT  // Active le bon cas
    #define cblas_xgemm cblas_sgemm
    #define cblas_xcopy cblas_scopy
    #define cblas_xaxpy cblas_saxpy
    #define WICHTYPE() printf("Using float\n")
#elif USE_DOUBLE
    #define MYTYPE double
    #define MYTYPE_DOUBLE // Active le bon cas
    #define cblas_xgemm cblas_dgemm
    #define cblas_xcopy cblas_dcopy
    #define cblas_xaxpy cblas_daxpy
    #define WICHTYPE() printf("Using double\n")
#else
    #define MYTYPE double  // Default: Double
    #define cblas_xgemm cblas_dgemm
    #define cblas_xcopy cblas_dcopy
    #define cblas_xaxpy cblas_daxpy
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

int genrangerandint(int max_len, int min_len){
    return (rand() % (max_len - min_len + 1)) + min_len;
}

__global__ void vector_add(const MYTYPE* A, const MYTYPE* B, MYTYPE* C, int N) {
    int ID = threadIdx.x + blockDim.x * blockIdx.x;
    if (ID < N){
        //printf("ID %d \n", ID);
        C[ID] = A[ID] + B[ID];
    }
}

void init_rand(void* A, int N) {
    for (int i = 0; i < N; ++i){
        MYTYPE x = ((MYTYPE)rand()/(MYTYPE)(RAND_MAX)) * N;
        ((MYTYPE*)A)[i] = x;
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

    int N = MAX_LEN;
    #if defined(WSRAND)
    N = genrangerandint(MAX_LEN, MIN_LEN);
    #endif
    size_t arraySize = sizeof(MYTYPE)*N;

    // Allocate and initialize arrays on the Host
    MYTYPE* A = (MYTYPE*) malloc(arraySize);
    MYTYPE* B = (MYTYPE*) malloc(arraySize);
    MYTYPE* C = (MYTYPE*) malloc(arraySize);
    MYTYPE* HostRef = (MYTYPE*) malloc(arraySize);
    init_rand(A, N);
    init_rand(B, N);
    memset(C, 0, arraySize);
    memset(HostRef, 0, arraySize);

    // HostRef compute with cblas
    openblas_set_num_threads(1);
    cblas_xcopy(N, B, 1, HostRef, 1);
    MYTYPE alpha = (MYTYPE)1;
    struct rusage start_rusage, end_rusage;
    getrusage(RUSAGE_THREAD, &start_rusage);
    cblas_xaxpy(N, alpha, A, 1, HostRef, 1);
    getrusage(RUSAGE_THREAD, &end_rusage);
    double cpu_user_s = (end_rusage.ru_utime.tv_sec - start_rusage.ru_utime.tv_sec) +
                        (end_rusage.ru_utime.tv_usec - start_rusage.ru_utime.tv_usec)/1e6;

    // Allocate and initialize arrays on the device
    // The initialization step is a copy from host to device
    MYTYPE* dA=NULL;
    MYTYPE* dB=NULL;
    MYTYPE* dC=NULL;
    gpu_errchk(cudaMalloc((void**)&dA, arraySize));
    gpu_errchk(cudaMalloc((void**)&dB, arraySize));
    gpu_errchk(cudaMalloc((void**)&dC, arraySize));
    gpu_errchk(cudaMemcpy(dA, A, arraySize, cudaMemcpyHostToDevice));
    gpu_errchk(cudaMemcpy(dB, B, arraySize, cudaMemcpyHostToDevice));
    gpu_errchk(cudaMemcpy(dC, C, arraySize, cudaMemcpyHostToDevice));

    // Perform the vector add on the device
    dim3 threadsPerBlock = (16);
    dim3 numBlocks = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
    cudaEvent_t start, stop;
    gpu_errchk(cudaEventCreate(&start));
    gpu_errchk(cudaEventCreate(&stop));
    gpu_errchk(cudaEventRecord(start));
    vector_add<<<numBlocks, threadsPerBlock>>>(dA, dB, dC, N);
    gpu_errchk(cudaGetLastError());
    gpu_errchk(cudaDeviceSynchronize());
    // copy back the result from device to host
    gpu_errchk(cudaMemcpy(C, dC, arraySize, cudaMemcpyDeviceToHost));
    gpu_errchk(cudaEventSynchronize(stop));
    float kernel1_time_ms;
    cudaEventElapsedTime(&kernel1_time_ms, start, stop);
    compareHostDevice(HostRef, C, 1, N);

    WICHTYPE();
    #ifdef WSRAND
    printf("With random seed\n");
    #endif
    printf("VecAdd N=%d\n", N);
    printf("CPU kernel, blas ?axpy time: %f s\n", cpu_user_s);
    printf("GPU kernel time: %f ms\n", kernel1_time_ms);

    free(A);A=NULL;
    free(B);B=NULL;
    free(C);C=NULL;
    free(HostRef);HostRef=NULL;
    gpu_errchk(cudaFree(dA));dA=NULL;
    gpu_errchk(cudaFree(dB));dB=NULL;
    gpu_errchk(cudaFree(dC));dC=NULL;
    return 0;
}
