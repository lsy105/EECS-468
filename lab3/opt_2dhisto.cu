#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"


__global__ void HistCal(uint32_t *input, size_t width, size_t height, uint32_t *bins);
__global__ void uint32to8(uint32_t *input, uint8_t *output);

void  opt_2dhisto(uint32_t *input, size_t width, size_t height, uint32_t *bins_32, uint8_t *bins_8)
{
	dim3 DimGrid(width);
	dim3 DimBlock(256);

	HistCal<<<DimGrid, DimBlock>>>(input, width, height, bins_32);

	uint32to8<<<2, 512>>>(bins_32, bins_8);

	cudaThreadSynchronize();
}

__global__ void HistCal(uint32_t *input, size_t width, size_t height, uint32_t *bins) {

	const int threadOffset = threadIdx.x + blockIdx.x * blockDim.x;
	const int blockOffset = gridDim.x * blockDim.x;	

	if (threadOffset < 1024) {
		bins[threadOffset] = 0;
	}
	__syncthreads();

	uint32_t size = width * height;
	for (unsigned int i = threadOffset; i < size; i += blockOffset) {
		const int value = input[i];
		if (bins[value] < UINT8_MAX && value) {
			atomicAdd(&(bins[value]), 1);
		}
	}
	__syncthreads();
}

__global__ void uint32to8(uint32_t *input, uint8_t *output) {
	const int threadOffset = threadIdx.x + blockIdx.x * blockDim.x;
	output[threadOffset] = (uint8_t)((input[threadOffset] < UINT8_MAX) * input[threadOffset]) + (input[threadOffset] >= UINT8_MAX) * UINT8_MAX;
	syncthreads();
}

void* allocateDeviceMemory(size_t size) {
	void *p;
	cudaMalloc(&p, size);
	return p;
}

void copyToDeviceMemory(void* device, void* host, size_t size) {
	cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
}

void copyToHostMemory(void* host, void* device, size_t size) {
	cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
}

void freeDeviceMemory(void* device) {
	cudaFree(device);
}





