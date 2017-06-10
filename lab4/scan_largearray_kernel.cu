#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 256
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif


float** Global_BlockNum;
unsigned int Global_AllocEltNum = 0;
unsigned int Global_AllocLvNum = 0;


//function declare
template <bool storeSum, bool isNotPower2>

__global__ void Prescan_Kernel(float *global_dataout, const float *global_datain, float *g_blockSums, int n, int blockIndex, int baseIndex);

__global__ void Global_Add(float *global_data, float *block_num, int n,  int blockOffset, int baseIndex);


// Lab4: Host Helper Functions (allocate your own data structure...)

void MemAllocate(unsigned int num_elements)
{

    Global_AllocEltNum = num_elements;

    unsigned int blockSize = BLOCK_SIZE;
    unsigned int numElts = num_elements;

    int iteration = 0;

    while(numElts > 1){
        unsigned int numBlocks = max(1, (int)ceil((float)numElts / (2.f * blockSize)));

        if (numBlocks > 1)
        {
            iteration++;
        }
        numElts = numBlocks;    
    }

    Global_BlockNum = (float**) malloc(iteration * sizeof(float*));
    Global_AllocLvNum = iteration;

    numElts = num_elements;
    iteration = 0;

    do
    {
        unsigned int numBlocks = max(1, (int)ceil((float)numElts / (2.f * blockSize)));

        if (numBlocks > 1)
        {
            cudaMalloc((void**) &Global_BlockNum[iteration++], numBlocks * sizeof(float));
        }
        numElts = numBlocks;
    } while (numElts > 1);
}

void MemDeAllocate()
{
    for (unsigned int i = 0; i < Global_AllocLvNum; i++)
      cudaFree(Global_BlockNum[i]);

    free((void**)Global_BlockNum);

    Global_BlockNum = 0;
    Global_AllocEltNum = 0;
    Global_AllocLvNum = 0;
}

// Because the array is large, we calculate in multiple iterations
void PrefixSum(float *outArray, const float *inArray, int numElements, int iteration)
{
    unsigned int blockSize = BLOCK_SIZE;
    unsigned int numBlocks = max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if ((numElements&(numElements-1))==0)
        numThreads = numElements / 2;
    else{
        int exp;
        frexp((float)numElements, &exp);
        numThreads = 1 << (exp - 1);
    }

    unsigned int numEltsPerBlock = numThreads * 2;
    // The Last Block may have number of elements smaller than before.
    unsigned int numEltsLastBlock = numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int LastBlock_isnPowerOfTwo = 0;
    unsigned int sMemLastBlock = 0;
    
    if (numEltsLastBlock != numEltsPerBlock)
    {
        LastBlock_isnPowerOfTwo = 1;
        
        // if the number of elements in the last block is not a power of two.
        if((numEltsLastBlock&(numEltsLastBlock-1)) != 0){
            int exp;
            frexp((float)numEltsLastBlock, &exp);
            numThreadsLastBlock = 1 << (exp - 1);
        }

        // pad is used to avoid bank conflict
        unsigned int pad = (2 * numThreadsLastBlock) / NUM_BANKS;
        sMemLastBlock = sizeof(float) * (2 * numThreadsLastBlock + pad);
    }

    unsigned int pad = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = sizeof(float) * (numEltsPerBlock + pad);

    dim3  Grid_Size(max(1, numBlocks - LastBlock_isnPowerOfTwo));
    dim3  Block_Size(numThreads);

    if (numBlocks > 1)
    {
        Prescan_Kernel<true, false><<< Grid_Size, Block_Size, sharedMemSize >>>(outArray, inArray, Global_BlockNum[iteration], numThreads * 2, 0, 0);

        if (LastBlock_isnPowerOfTwo)
        {
          Prescan_Kernel<true, true><<< 1, numThreadsLastBlock, sMemLastBlock >>> (outArray, inArray, Global_BlockNum[iteration], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }

        PrefixSum(Global_BlockNum[iteration], Global_BlockNum[iteration], numBlocks, iteration+1);

        Global_Add<<< Grid_Size, Block_Size >>>(outArray, Global_BlockNum[iteration], numElements - numEltsLastBlock, 0, 0);

        if (LastBlock_isnPowerOfTwo){
          Global_Add<<< 1, numThreadsLastBlock >>>(outArray, Global_BlockNum[iteration], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
		}

    }
    else if ((numElements&(numElements-1)) == 0)
    {
      Prescan_Kernel<false, false><<< Grid_Size, Block_Size, sharedMemSize >>>(outArray, inArray, 0, numThreads * 2, 0, 0);
    }
    else
    {
      Prescan_Kernel<false, true><<< Grid_Size, Block_Size, sharedMemSize >>>(outArray, inArray, 0, numElements, 0, 0);
    }
}


// Lab4: Device Functions

// Load data into shared memory.
template <bool isNotPower2>
__device__ void LoadShMem(float *s_data, const float *global_datain,int n, int baseIndex, int& indexA, int& indexB, int& m_indexA, int& m_indexB, int& bankOffsetA, int& bankOffsetB)
{
    int tid = threadIdx.x;

    indexA = tid;
    indexB = tid + blockDim.x;
    m_indexA = baseIndex + tid;
    m_indexB = m_indexA + blockDim.x;
    bankOffsetA = CONFLICT_FREE_OFFSET(indexA);
    bankOffsetB = CONFLICT_FREE_OFFSET(indexB);

    s_data[indexA + bankOffsetA] = global_datain[m_indexA]; 
    
    // be careful if not power of 2
    if (isNotPower2){
      s_data[indexB + bankOffsetB] = (indexB < n) ? global_datain[m_indexB] : 0; 
    }
    else
    {
      s_data[indexB + bankOffsetB] = global_datain[m_indexB]; 
    }
}


template <bool isNotPower2>
__device__ void StoreShMem(float* global_dataout, const float* s_data, int n,  int indexA, int indexB, int m_indexA, int m_indexB, int bankOffsetA, int bankOffsetB)
{
    __syncthreads();
    
    // store data back to global memory.
    global_dataout[m_indexA] = s_data[indexA + bankOffsetA]; 
    if (isNotPower2){
        if (indexB < n)
            global_dataout[m_indexB] = s_data[indexB + bankOffsetB]; 
    }
    else
    {
        global_dataout[m_indexB] = s_data[indexB + bankOffsetB]; 
    }
}

__device__ unsigned int ReductionSum(float *s_data)
{
    unsigned int tid = threadIdx.x;
    unsigned int stride = 1;
    
    for (int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();

        if (tid < d){
            int i  = __mul24(__mul24(2, stride), tid);
            int indexA = i + stride - 1;
            int indexB = indexA + stride;

            indexA += CONFLICT_FREE_OFFSET(indexA);
            indexB += CONFLICT_FREE_OFFSET(indexB);

            s_data[indexB] += s_data[indexA];
        }

        stride *= 2;
    }

    return stride;
}

template <bool storeSum>
__device__ void LastElementZero(float* s_data, float *g_blockSums, int blockIndex)
{
    if (threadIdx.x == 0)
    {
        int index = (blockDim.x << 1) - 1;
        index += CONFLICT_FREE_OFFSET(index);
        
        if (storeSum){
            g_blockSums[blockIndex] = s_data[index];
        }

        s_data[index] = 0;
    }
}

__device__ void SweepDown(float *s_data, unsigned int stride)
{
    unsigned int tid = threadIdx.x;
    for (int d = 1; d <= blockDim.x; d *= 2)
    {
        stride = stride >> 1;
        __syncthreads();

        if (tid < d)
        {
            int i  = __mul24(__mul24(2, stride), tid);
            int indexA = i + stride - 1;
            int indexB = indexA + stride;
            indexA += CONFLICT_FREE_OFFSET(indexA);
            indexB += CONFLICT_FREE_OFFSET(indexB);

            float t  = s_data[indexA];
            s_data[indexA] = s_data[indexB];
            s_data[indexB] += t;
        }
    }
}

template <bool storeSum>
__device__ void PrefixSumBlock(float *data, int blockIndex, float *blockSums)
{
    // phase1: Up-sweep -- work-efficient sum scan alg.
    int stride = ReductionSum(data);

    // phase2: Down-sweep -- work-efficient parallel sum scan alg.
    LastElementZero<storeSum>(data, blockSums, (blockIndex == 0) ? blockIdx.x : blockIndex);
    SweepDown(data, stride); 
}


// Lab4: Kernel Functions

// Do Prefix Scan
template <bool storeSum, bool isNotPower2>
__global__ void Prescan_Kernel(float *global_dataout, const float *global_datain, float *g_blockSums, int n, int blockIndex, int baseIndex)
{
    int indexA, indexB, m_indexA, m_indexB, bankOffsetA, bankOffsetB;

    // using shared memory
    extern __shared__ float s_data[];

    LoadShMem<isNotPower2>(s_data, global_datain, n, (baseIndex == 0) ?  mul24(blockIdx.x, (blockDim.x << 1)):baseIndex, indexA, indexB, m_indexA, m_indexB, bankOffsetA, bankOffsetB); 

    // do prefix scan in each block
    PrefixSumBlock<storeSum>(s_data, blockIndex, g_blockSums); 

    StoreShMem<isNotPower2>(global_dataout, s_data, n, 
                                 indexA, indexB, m_indexA, m_indexB, 
                                 bankOffsetA, bankOffsetB);  
}

// Add results from each block
__global__ void Global_Add(float *global_data, float *block_num, int n, int blockOffset, int baseIndex)
{
    __shared__ float sharedId;
    if (threadIdx.x == 0)
        sharedId = block_num[blockIdx.x + blockOffset];
    
    unsigned int addr = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x; 

    __syncthreads();
    
    global_data[addr] += sharedId;
    global_data[addr + blockDim.x] += (threadIdx.x + blockDim.x < n) * sharedId;
}



// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements)
{
	MemAllocate(numElements);
	PrefixSum(outArray, inArray, numElements, 0);
	MemDeAllocate();
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
