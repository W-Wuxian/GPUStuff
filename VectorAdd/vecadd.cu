#include <stdio.h>
#include <time.h>
#include <stdlib.h>
//#include <sys/time.h>
//#include <math.h>
//#include <omp.h>
//#include <cuda.h>
#include <cuda_runtime.h>
//#include <cuda_profiler_api.h>
#define MIN_LEN 1
#define MAX_LEN 30
//#define MAX_LEN 100000000

__global__ void pinfo(){
    printf("threadIdx %u %d %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N){
        //printf("idx %d \n", idx);
        C[idx] = A[idx] + B[idx];
    }
}

void init_rand(void* A, int N) {
    for (int i = 0; i < N; ++i){
        float x = ((float)rand()/(float)(RAND_MAX)) * N;
        ((float*)A)[i] = x;
    }
}

int main(){
    pinfo<<<dim3(2),dim3(3)>>>();
    cudaDeviceSynchronize();

    srand(time(0));
    int N = rand() % (MAX_LEN - MIN_LEN + 1) + MIN_LEN;
    size_t arraySize = sizeof(float)*N;

    // Allocate and initialize arrays on the Host
    float* A = (float*) malloc(arraySize);
    float* B = (float*) malloc(arraySize);
    float* C = (float*) malloc(arraySize);
    init_rand(A, N);
    init_rand(B, N);
    memset(C, 0, arraySize);

    // Allocate and initialize arrays on the device
    // The initialization step is a copy from host to device
    float* dA=NULL;
    float* dB=NULL;
    float* dC=NULL;
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
    for(int i = 0 ; i < N; i++)printf("A[%i] = %f B[%i] = %f | C[%d] = %f\n", i, A[i], i, B[i], i, C[i]);
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
