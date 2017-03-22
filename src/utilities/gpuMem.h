#if defined(HYPRE_USE_GPU) || defined(HYPRE_USE_MANAGED)
#ifndef __GPUMEM_H__
#define  __GPUMEM_H__
#include <cublas_v2.h>
#include <cusparse.h>
cudaStream_t getstream(int i);
nvtxDomainHandle_t getdomain(int i);
cudaEvent_t getevent(int i);
void MemAdviseReadOnly(const void *ptr, int device);
void MemAdviseUnSetReadOnly(const void *ptr, int device);
void MemAdviseSetPrefLocDevice(const void *ptr, int device);
void MemAdviseSetPrefLocHost(const void *ptr);
void MemPrefetch(const void *ptr,int device,cudaStream_t stream);
void MemPrefetchSized(const void *ptr,size_t size,int device,cudaStream_t stream);
void MemPrefetchForce(const void *ptr,int device,cudaStream_t stream);
cublasHandle_t getCublasHandle();
cusparseHandle_t getCusparseHandle();
void hypreGPUInit();
typedef struct node {
  const void *ptr;
  size_t size;
  struct node *next;
} node;
size_t mempush(const void *ptr, size_t size, int action);
node *memfind(node *head, const void *ptr);
void memdel(node **head, node *found);
void meminsert(node **head, const void *ptr,size_t size);
void printlist(node *head,int nc);
#define MEM_PAD_LEN 1
size_t memsize(const void *ptr);
int getsetasyncmode(int mode, int action);
void SetAsyncMode(int mode);
int GetAsyncMode();
void branchStream(int i, int j);
void joinStreams(int i, int j, int k);
#endif
#endif
