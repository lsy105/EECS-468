#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

__global__ void HistCal( uint32_t * input, size_t height, size_t width, Hist h);

void opt_2dhisto( uint32_t * input, size_t height, size_t width, Hist &h)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
    dim3 dimGrid(1, 1);
    dim3 dimBlock(32, 1);
    HistCal<<<dimGrid, dimBlock>>>(input, height, width, h);
}

/* Include below the implementation of any other functions you need */
__global__ void HistCal( uint32_t * input, size_t height, size_t width, Hist h)
{	
	__shared__ uint8_t subhist[32][1024 + 1];
	unsigned int subhistIdx = threadIdx.x;
	for (int i = 0; i < h.width; i++)
        {
	  subhist[subhistIdx][i] += 0;
	}

	//cal subhist
	for (int j = 0; j < height; j++)
    	{
	 for (int i = 0; i < width/blockDim.x; i++)
          {
	    uint32_t offset = i*blockDim.x + threadIdx.x;
	    uint32_t value = input[j*width + offset];
	    if(subhist[subhistIdx][value] < UINT8_MAX && offset < INPUT_WIDTH) {
	      subhist[subhistIdx][value] += 1;
	      }
	  }
	}
	__syncthreads();

	//merge subhist
	if(subhistIdx < 16) {
	  for(int i = 0; i < h.width; i++)
	  {
	    subhist[subhistIdx][i] += subhist[subhistIdx + 16][i];
 	    subhist[subhistIdx][i] += subhist[subhistIdx + 8][i];
	    subhist[subhistIdx][i] += subhist[subhistIdx + 4][i];
	    subhist[subhistIdx][i] += subhist[subhistIdx + 2][i];
	    subhist[subhistIdx][i] += subhist[subhistIdx + 1][i];
	  }
	}
	__syncthreads();


	for(int i = 0; i < h.width; i++)
	{
	 h.bins[i] = subhist[0][i];
	}
}

Hist AllocateDeviceHist(const Hist H)
{
  Hist Hdevice = H;
  int size = H.width * H.height * sizeof(uint8_t);
  cudaMalloc((void**)&Hdevice.bins, size);
  return Hdevice;
}
uint32_t* AllocateDeviceInput(size_t size)
{
  uint32_t* input;
  cudaMalloc((void**)&input, size);
  return input;
}

void CopyToDeviceHist(Hist Hdevice, const Hist Hhost)
{
  int size = Hhost.width * Hhost.height * sizeof(uint8_t);
  cudaMemcpy(Hdevice.bins, Hhost.bins, size, cudaMemcpyHostToDevice);
}

void CopyToDeviceInput(void* host_input, void* device_input, size_t size)
{
  cudaMemcpy(device_input, host_input, size, cudaMemcpyHostToDevice);
}

void CopyFromDeviceHist(Hist Hhost, const Hist Hdevice)
{
  int size = Hdevice.width * Hdevice.height * sizeof(uint8_t);
  cudaMemcpy(Hhost.bins, Hdevice.bins, size, cudaMemcpyDeviceToHost);
}

void FreeDeviceHist(Hist* H)
{
  cudaFree(H->bins);
  H->bins = NULL;
}

void FreeDeviceInput(uint32_t* input)
{
  cudaFree(input);
}
