#ifdef HYPRE_USE_GPU
size_t mempush(const void *ptr, size_t size,int purge);
int memloc(const void *ptr, int device);
cudaStream_t getstream(int i);
#endif
