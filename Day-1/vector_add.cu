#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* input1, const float* input2, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = input1[i] + input2[i];
    }
}

int main() {
    const int size = 10;
    float input1[size], input2[size], output[size];

    // Initialize input arrays
    for (int i = 0; i < size; i++) {
        input1[i] = i * 1.0f;
        input2[i] = (size - i) * 1.0f;
    }

    // Print input arrays
    std::cout << "Input1: ";
    for (int i = 0; i < size; i++) std::cout << input1[i] << " ";
    std::cout << std::endl;

    std::cout << "Input2: ";
    for (int i = 0; i < size; i++) std::cout << input2[i] << " ";
    std::cout << std::endl;

    // Allocate memory on device
    float *dev_input1, *dev_input2, *dev_output;
    cudaMalloc(&dev_input1, size * sizeof(float));
    cudaMalloc(&dev_input2, size * sizeof(float));
    cudaMalloc(&dev_output, size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(dev_input1, input1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_input2, input2, size * sizeof(float), cudaMemcpyHostToDevice);

    // Define kernel launch configuration
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Launch kernel
    vectorAdd<<<gridSize, blockSize>>>(dev_input1, dev_input2, dev_output, size);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(output, dev_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Output (Input1 + Input2): ";
    for (int i = 0; i < size; i++) std::cout << output[i] << " ";
    std::cout << std::endl;

    // Free device memory
    cudaFree(dev_input1);
    cudaFree(dev_input2);
    cudaFree(dev_output);

    return 0;
}
