#ifndef OPT_KERNEL
#define OPT_KERNEL

typedef struct {
  uint8_t * bins;
  unsigned int width;
  unsigned int height;
} Hist;

void opt_2dhisto(uint32_t * input, size_t height, size_t width, Hist &h);

/* Include below the function headers of any other functions that you implement */
Hist AllocateDeviceHist(const Hist H);
uint32_t* AllocateDeviceInput(size_t size);
void CopyToDeviceHist(Hist Hdevice, const Hist Hhost);
void CopyToDeviceInput(void* host_input, void* device_input, size_t size);
void CopyFromDeviceHist(Hist Hhost, const Hist Hdevice);
void FreeDeviceHist(Hist* h);
void FreeDeviceInput(uint32_t* input);

#endif
