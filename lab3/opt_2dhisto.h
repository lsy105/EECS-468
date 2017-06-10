#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t*, size_t, size_t, uint32_t*, uint8_t*);
void* allocateDeviceMemory(size_t);
void copyToDeviceMemory(void*, void*, size_t);
void copyToHostMemory(void*, void*, size_t);
void freeDeviceMemory(void*);


/* Include below the function headers of any other functions that you implement */


#endif
