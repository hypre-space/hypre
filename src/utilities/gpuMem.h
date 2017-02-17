void MemAdviseReadOnly(const void *ptr, int device);
void MemAdviseUnSetReadOnly(const void *ptr, int device);
void MemAdviseSetPrefLocDevice(const void *ptr, int device);
void MemAdviseSetPrefLocHost(const void *ptr);
void MemPrefetch(const void *ptr,int device,cudaStream_t stream);
void MemPrefetchForce(const void *ptr,int device,cudaStream_t stream);
void MemPrefetchReadOnly(const void *ptr,int device,cudaStream_t stream);
