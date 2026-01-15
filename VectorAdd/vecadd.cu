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
#define MIN_LEN 1
#define MAX_LEN 30
//#define MAX_LEN 100000000

#ifdef USE_FLOAT
    #define MYTYPE float
    #define MYTYPE_FLOAT  // Active le bon cas
    #define cblas_xgemm cblas_sgemm
    #define cblas_xcopy cblas_scopy
    #define cblas_xaxpy cblas_saxpy
#elif USE_DOUBLE
    #define MYTYPE double
    #define MYTYPE_DOUBLE // Active le bon cas
    #define cblas_xgemm cblas_dgemm
    #define cblas_xcopy cblas_dcopy
    #define cblas_xaxpy cblas_daxpy
#else
    #define MYTYPE double  // Default: Double
    #define cblas_xgemm cblas_dgemm
    #define cblas_xcopy cblas_dcopy
    #define cblas_xaxpy cblas_daxpy
#endif

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

    srand(time(0));
    int N = genrangerandint(MAX_LEN, MIN_LEN);
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
    cblas_xcopy(N, B, 1, HostRef, 1);
    MYTYPE alpha = (MYTYPE)1;
    cblas_xaxpy(N, alpha, A, 1, HostRef, 1);

    // Allocate and initialize arrays on the device
    // The initialization step is a copy from host to device
    MYTYPE* dA=NULL;
    MYTYPE* dB=NULL;
    MYTYPE* dC=NULL;
    cudaMalloc((void**)&dA, arraySize);
    cudaMalloc((void**)&dB, arraySize);
    cudaMalloc((void**)&dC, arraySize);
    cudaMemcpy(dA, A, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, arraySize, cudaMemcpyHostToDevice);

    // Perform the vector add on the device
    dim3 threadsPerBlock = (10);
    dim3 numBlocks = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
    vector_add<<<numBlocks, threadsPerBlock>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    // copy back the result from device to host
    cudaMemcpy(C, dC, arraySize, cudaMemcpyDeviceToHost);
    for(int i = 0 ; i < N; i++){
        if(HostRef[i]!=)
        printf("A[%i] = %f B[%i] = %f | C[%d] = %f\n", i, A[i], i, B[i], i, C[i]);
    }
    free(A);A=NULL;
    free(B);B=NULL;
    free(C);C=NULL;
    free(HostRef);HostRef=NULL;
    cudaFree(dA);dA=NULL;
    cudaFree(dB);dB=NULL;
    cudaFree(dC);dC=NULL;
    return 0;
}
