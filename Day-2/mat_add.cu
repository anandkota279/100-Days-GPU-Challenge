#include <iostream>
#include <cuda_runtime.h>

__global__ void MatrixAddKernel(const float* matrixA, const float* matrixB, float* matrixC, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        int index = row * N + col;
        matrixC[index] = matrixA[index] + matrixB[index];
    }
}

int main() {
    const int N = 10;
    float *matrixA, *matrixB, *matrixC;

    matrixA = (float *)malloc(N * N * sizeof(float));
    matrixB = (float *)malloc(N * N * sizeof(float));
    matrixC = (float *)malloc(N * N * sizeof(float));

    // Initialize matrixA and matrixB
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            matrixA[row * N + col] = 1.0f;
            matrixB[row * N + col] = 2.0f;
            matrixC[row * N + col] = 0.0f;
        }
    }

    float *deviceMatrixA, *deviceMatrixB, *deviceMatrixC;
    cudaMalloc((void **)&deviceMatrixA, N * N * sizeof(float));
    cudaMalloc((void **)&deviceMatrixB, N * N * sizeof(float));
    cudaMalloc((void **)&deviceMatrixC, N * N * sizeof(float));

    cudaMemcpy(deviceMatrixA, matrixA, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, matrixB, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(32, 16);
    dim3 dimGrid(ceil(N / 32.0f), ceil(N / 16.0f));

    MatrixAddKernel<<<dimGrid, dimBlock>>>(deviceMatrixA, deviceMatrixB, deviceMatrixC, N);
    cudaDeviceSynchronize();

    cudaMemcpy(matrixC, deviceMatrixC, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print Result
    printf("Matrix C (A + B):\n");
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            printf("%.2f ", matrixC[row * N + col]);
        }
        printf("\n");
    }

    // Free memory
    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceMatrixC);
    free(matrixA);
    free(matrixB);
    free(matrixC);

    return 0;
}
