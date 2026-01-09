#include <stdio.h>
//#include <stdlib.h>
//#include <sys/time.h>
//#include <math.h>
//#include <omp.h>
//#include <cuda.h>
#include <cuda_runtime.h>
//#include <cuda_profiler_api.h>
#define PRINT_1U(...) printf("%u\n", __VA_ARGS__)
#define PRINT_2U(...) printf("%u %u\n", __VA_ARGS__)
#define PRINT_3U(...) printf("%u %u %u\n", __VA_ARGS__)


__global__ void pinfo(){
    int tid = threadIdx.x + blockIdx.x * blockDim.x + (threadIdx.y + blockIdx.y * blockDim.y) * (gridDim.x * blockDim.x)
    + (threadIdx.z + blockIdx.z * blockDim.z) * (gridDim.x * blockDim.x * gridDim.y * blockDim.y);
    int totalThreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    for (int i = 0; i < totalThreads; i++) {
        if (i == tid) {
            printf("\tgridDim: variable of type dim3 and contains the dimensions of the grid\t");PRINT_3U(gridDim.x, gridDim.y,gridDim.z);
            printf("\tblockDim: variable of type dim3 and contains the dimensions of the block\t");PRINT_3U(blockDim.x, blockDim.y,blockDim.z);
            printf("\tthreadIdx: variable of type uint3 and contains the thread index within the block\t");PRINT_3U(threadIdx.x, threadIdx.y, threadIdx.z);
            printf("\tblockIdx: variable of type uint3 and contains the block index within the grid\t");PRINT_3U(blockIdx.x, blockIdx.y, blockIdx.z);
            printf("Thread Hierarchy\n\tthe thread ID of a thread of index (x,y,z) is (x + yDx + zDxDy)\t");printf("Thread ID = %u\n", threadIdx.x + threadIdx.y*blockIdx.x + threadIdx.z*blockIdx.x*blockIdx.y);
            printf("Where Dx Dy Dz refer to the block size in each direction and are respectively equal to blockIdx.x  blockIdx.y blockIdx.z\n");
        }
        __syncthreads();
    }
}

int main(int argc, char **argv){

    
    dim3 Dg = dim3(1); /*< Dimension and size of the grid; Dg.x * Dg.y * Dg.z equals the number of blocks being launched>*/
    dim3 Db = dim3(1); /*< Dimension and size of each block; Db.x * Db.y * Db.z equals the number of threads per block>*/

    if (argc == 7) {
        Dg = make_uint3(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
        Db = make_uint3(atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
        //Dg = {.x=uint(atoi(argv[1])), .y=.uint(atoi(argv[2])), .z=.uint(atoi(argv[3]))};
        //Db = {.x=uint(atoi(argv[4])), .y=.uint(atoi(argv[5])), .z=.uint(atoi(argv[6]))};
    } else {
        printf("Usage: %s Dgx Dgy Dgz Dbx Dby Dbz\n", argv[0]);
        printf("Using default values (all = 1)\n");
    }

    printf("Dimension and size of the grid:\n");
    printf("Dg, will be used to set the Built-in variable 'gridDim'\n");
    printf("\tGrid size (Total number of block) is:\t");PRINT_1U(Dg.x * Dg.y *Dg.z);
    printf("\tNumber of block in each direction (x,y,z):\t");PRINT_3U(Dg.x, Dg.y, Dg.z);
    printf("Dimension and size of each block:\n");
    printf("Db, will be used to set the Built-in variable 'blockDim'\n");
    printf("\tNumber of threads per block is:\t");PRINT_1U(Db.x * Db.y *Db.z);
    printf("\tNumber of threads in each direction (x,y,z):\t");PRINT_3U(Db.x, Db.y, Db.z);
    printf("Displaying Built-In Variables:\n");
    pinfo<<<Dg,Db>>>();
    cudaDeviceSynchronize();
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
