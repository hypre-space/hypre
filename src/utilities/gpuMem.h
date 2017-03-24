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
void hypre_GPUInit();
void hypre_GPUFinalize();
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

/*
 * Global struct for keeping HYPRE GPU Init state
 */

#define MAX_HGS_ELEMENTS 10
struct hypre__global_struct{
  int initd;
  int device;
  int device_count;
  cublasHandle_t cublas_handle;
  cusparseHandle_t cusparse_handle;
  cusparseMatDescr_t cusparse_mat_descr;
  cudaStream_t streams[MAX_HGS_ELEMENTS];
};

extern struct hypre__global_struct hypre__global_handle ;

/*
 * Macros for accessing the handle members
 */
#define HYPRE_GPU_HANDLE hypre__global_handle.initd
#define HYPRE_CUBLAS_HANDLE hypre__global_handle.cublas_handle
#define HYPRE_CUSPARSE_HANDLE hypre__global_handle.cusparse_handle
#define HYPRE_DEVICE hypre__global_handle.device
#define HYPRE_DEVICE_COUNT hypre__global_handle.device_count
#define HYPRE_CUSPARSE_MAT_DESCR hypre__global_handle.cusparse_mat_descr
#define HYPRE_STREAM(index) (hypre__global_handle.streams[index])






#endif
#endif
