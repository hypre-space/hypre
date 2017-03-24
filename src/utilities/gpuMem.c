#if defined(HYPRE_USE_GPU) || defined(HYPRE_USE_MANAGED)
#include "gpuErrorCheck.h"
#include "hypre_nvtx.h"
#include <stdlib.h>
#include <stdint.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include "_hypre_utilities.h"
#include "gpuMem.h"
#include "../seq_mv/gpukernels.h"
int ggc(int id);
#define FULL_WARN

/* Global struct that holds device,library handles etc */
struct hypre__global_struct hypre__global_handle = { .initd=0, .device=0, .device_count=1};



void hypre_GPUInit(){
  char pciBusId[80];
  int myid;
  int nDevices;
  int device;
  if (!HYPRE_GPU_HANDLE){
    HYPRE_GPU_HANDLE=1;
    HYPRE_DEVICE=0;
    gpuErrchk(cudaGetDeviceCount(&nDevices));
    HYPRE_DEVICE_COUNT=nDevices;
    
    if (nDevices==1){
      HYPRE_DEVICE=0;
      gpuErrchk(cudaSetDevice(HYPRE_DEVICE));
      cudaDeviceGetPCIBusId ( pciBusId, 80, HYPRE_DEVICE);
    } else if (nDevices>1) {

      hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
      
      MPI_Comm node_comm;
      MPI_Info info;
      MPI_Info_create(&info);
      MPI_Comm_split_type(hypre_MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, myid, info, &node_comm);

      int myNodeid, NodeSize;
      MPI_Comm_rank(node_comm, &myNodeid);
      MPI_Comm_size(node_comm, &NodeSize);
      
      HYPRE_DEVICE=myNodeid%nDevices; 
      gpuErrchk(cudaSetDevice(HYPRE_DEVICE));
      cudaDeviceGetPCIBusId ( pciBusId, 80, HYPRE_DEVICE);
      hypre_printf("WARNING:: Code running without mpibind or similar affinity support\n");
      hypre_printf("Global ID = %d , Node ID %d running on device %d of %d \n",myid,myNodeid,HYPRE_DEVICE,nDevices);
      MPI_Info_free(&info);
    } else {
      /* No device found  */
      hypre_fprintf(stderr,"ERROR:: NO GPUS found \n");
      exit(2);
    }
    

    /* Initialize streams */
    for(int jj=0;jj<MAX_HGS_ELEMENTS;jj++)
      gpuErrchk(cudaStreamCreateWithFlags(&(HYPRE_STREAM(jj)),cudaStreamNonBlocking));
    
      /* Initialize the library handles and streams */
    //hypre__global_handle.cublas_handle=getCublasHandle();
    //hypre__global_handle.cusparse_handle=getCusparseHandle();

    cusparseErrchk(cusparseCreate(&(HYPRE_CUSPARSE_HANDLE)));
    cusparseErrchk(cusparseSetStream(HYPRE_CUSPARSE_HANDLE,getstream(4)));
    cusparseErrchk(cusparseCreateMatDescr(&(HYPRE_CUSPARSE_MAT_DESCR))); 
    cusparseErrchk(cusparseSetMatType(HYPRE_CUSPARSE_MAT_DESCR,CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseErrchk(cusparseSetMatIndexBase(HYPRE_CUSPARSE_MAT_DESCR,CUSPARSE_INDEX_BASE_ZERO));

    cublasErrchk(cublasCreate(&(HYPRE_CUBLAS_HANDLE)));
    cublasErrchk(cublasSetStream(HYPRE_CUBLAS_HANDLE,getstream(4)));
    
    /* Check if the arch flags used for compiling the cuda kernels match the device */
    CudaCompileFlagCheck();
  }
}


void hypre_GPUFinalize(){
  
  cusparseErrchk(cusparseDestroy(HYPRE_CUSPARSE_HANDLE));
  
  cublasErrchk(cublasDestroy(HYPRE_CUBLAS_HANDLE));
  
}

void MemAdviseReadOnly(const void* ptr, int device){
  if (ptr==NULL) return;
    size_t size=mempush(ptr,0,0);
    if (size==0) printf("WARNING:: Operations with 0 size vector \n");
    gpuErrchk(cudaMemAdvise(ptr,size,cudaMemAdviseSetReadMostly,device));
}
void MemAdviseUnSetReadOnly(const void* ptr, int device){
  if (ptr==NULL) return;
    size_t size=mempush(ptr,0,0);
    if (size==0) printf("WARNING:: Operations with 0 size vector \n");
    gpuErrchk(cudaMemAdvise(ptr,size,cudaMemAdviseUnsetReadMostly,device));
}


void MemAdviseSetPrefLocDevice(const void *ptr, int device){
  if (ptr==NULL) return;
  gpuErrchk(cudaMemAdvise(ptr,mempush(ptr,0,0),cudaMemAdviseSetPreferredLocation,device));
}
void MemAdviseSetPrefLocHost(const void *ptr){
  if (ptr==NULL) return;
  gpuErrchk(cudaMemAdvise(ptr,mempush(ptr,0,0),cudaMemAdviseSetPreferredLocation,cudaCpuDeviceId));
}


void MemPrefetch(const void *ptr,int device,cudaStream_t stream){
  if (ptr==NULL) return;
  size_t size;
  size=memsize(ptr);
  PUSH_RANGE_DOMAIN("MemPreFetchForce",4,0);
  /* Do a prefetch every time until a possible UM bug is fixed */
  if (size>0){
    PrintPointerAttributes(ptr);
     gpuErrchk(cudaMemPrefetchAsync(ptr,size,device,stream));
    gpuErrchk(cudaStreamSynchronize(stream));
    POP_RANGE_DOMAIN(0);
  return;
  } 
  return;
}
void MemPrefetchForce(const void *ptr,int device,cudaStream_t stream){
  if (ptr==NULL) return;
  size_t size=memsize(ptr);
  PUSH_RANGE_PAYLOAD("MemPreFetchForce",4,size);
  gpuErrchk(cudaMemPrefetchAsync(ptr,size,device,stream));
  POP_RANGE;
  return;
}

void MemPrefetchSized(const void *ptr,size_t size,int device,cudaStream_t stream){
  if (ptr==NULL) return;
  PUSH_RANGE_DOMAIN("MemPreFetchSized",4,0);
  /* Do a prefetch every time until a possible UM bug is fixed */
  if (size>0){
    gpuErrchk(cudaMemPrefetchAsync(ptr,size,device,stream));
    POP_RANGE_DOMAIN(0);
    return;
  } 
  return;
}



/* void PrintPointerAttributesNew(const void *ptr){ */
/*   struct cudaPointerAttributes ptr_att; */
/*   if (cudaPointerGetAttributes(&ptr_att,ptr)!=cudaSuccess){ */
/*     printf("PrintPointerAttributes:: Raw pointer\n"); */
/*     return; */
/*   } */
/*   if (ptr_att.isManaged){ */
/*     printf("PrintPointerAttributes:: Managed pointer\n"); */
/*     printf("Host address = %p, Device Address = %p\n",ptr_att.hostPointer, ptr_att.devicePointer); */
/*     if (ptr_att.memoryType==cudaMemoryTypeHost) printf("Memory is located on host\n"); */
/*     if (ptr_att.memoryType==cudaMemoryTypeDevice) printf("Memory is located on device\n"); */
/*     printf("Device associated with this pointer is %d\n",ptr_att.device); */
/*   } else { */
/*     printf("PrintPointerAttributes:: Non-Managed & non-raw pointer\n Probably a device pointer\n"); */
/*   } */
/*   return; */
/* } */
/* int OnHost(void *ptr){ */
/*   return 1; */
/*   struct cudaPointerAttributes ptr_att; */
/*   if (cudaPointerGetAttributes(&ptr_att,ptr)==cudaSuccess){ */
/*     return (ptr_att.memoryType==cudaMemoryTypeHost); */
/*   } else { */
/*     printf("PrintPointerAttributes:: Raw pointer\n"); */
/*     return 0; */
/*   }  */
/* } */

/* Returns the same cublas handle with every call */
cublasHandle_t getCublasHandle(){
  cublasStatus_t stat;
  static cublasHandle_t handle;
  static int firstcall=1;
  if (firstcall){
    firstcall=0;
    stat = cublasCreate(&handle);
    if (stat!=CUBLAS_STATUS_SUCCESS) {
      printf("ERROR:: CUBLAS Library initialization failed\n");
      handle=0;
      exit(2);
    }
    cublasErrchk(cublasSetStream(handle,getstream(4)));
  } else return handle;
  return handle;
}

/* Returns the same cusparse handle with every call */
cusparseHandle_t getCusparseHandle(){
  cusparseStatus_t status;
  static cusparseHandle_t handle;
  static int firstcall=1;
  if (firstcall){
    firstcall=0;
    status= cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      printf("ERROR:: CUSPARSE Library initialization failed\n");
      handle=0;
      exit(2);
    }
    cusparseErrchk(cusparseSetStream(handle,getstream(4)));
  } else return handle;
  return handle;
}

/* C version of mempush using linked lists */

size_t mempush(const void *ptr, size_t size, int action){
  static node* head=NULL;
  static int nc=0;
  node *found=NULL;
  if (!head){
    if ((size<=0)||(action==1)) {
      fprintf(stderr,"mempush can start only with an insertion or a size call \n");
      return 0;
    }
    head = (node*)malloc(sizeof(node));
    head->ptr=ptr;
    head->size=size;
    head->next=NULL;
    nc++;
    return size;
  } else {
    // Purge an address
    if (action==1){
      found=memfind(head,ptr);
      if (found){
	memdel(&head, found);
	nc--;
	return 0;
      } else {
#ifdef FULL_WARN
	fprintf(stderr,"ERROR :: Pointer for deletion not found in linked list %p\n",ptr);
#endif
	return 0;
      }
    } // End purge
    
    // Insertion
    if (size>0){
      found=memfind(head,ptr);
      if (found){
#ifdef FULL_WARN
	fprintf(stderr,"ERROR :: Pointer for insertion already in use in linked list %p\n",ptr);
	//printlist(head,nc);
#endif
	return 0;
      } else {
	nc++;
	meminsert(&head,ptr,size);
	return 0;
      }
    }

    // Getting allocation size
    found=memfind(head,ptr);
    if (found){
      return found->size;
    } else{
#ifdef FULL_WARN
      fprintf(stderr,"ERROR :: Pointer for size check NOT found in linked list\n");
#endif
      return 0;
    }
  }
}
node *memfind(node *head, const void *ptr){
  node *next;
  next=head;
  while(next!=NULL){
    if (next->ptr==ptr) return next;
    next=next->next;
  }
  return NULL;
}
void memdel(node **head, node *found){
  node *next;
  if (found==*head){
    next=(*head)->next;
    free(*head);
    *head=next;
    return;
  }
  next=*head;
  while(next->next!=found){
    next=next->next;
  }
  next->next=next->next->next;
  free(found);
  return;
}
void meminsert(node **head, const void  *ptr,size_t size){
  node *nhead;
  nhead = (node*)malloc(sizeof(node));
  nhead->ptr=ptr;
  nhead->size=size;
  nhead->next=*head;
  *head=nhead;
  return;
}
void printlist(node *head,int nc){
  node *next;
  next=head;
  printf("Node count %d \n",nc);
  while(next!=NULL){
    printf("Address %p of size %zu \n",next->ptr,next->size);
    next=next->next;
  }
}

cudaStream_t getstream(int i){
  static int firstcall=1;
  const int MAXSTREAMS=10;
  static cudaStream_t s[MAXSTREAMS];
  if (firstcall){
    for(int jj=0;jj<MAXSTREAMS;jj++)
      gpuErrchk(cudaStreamCreateWithFlags(&s[jj],cudaStreamNonBlocking));
    //printf("Created streams ..\n");
    firstcall=0;
  }
  if (i<MAXSTREAMS) return s[i];
  fprintf(stderr,"ERROR in getstream in utilities/gpuMem.c %d is greater than MAXSTREAMS = %d\n Returning default stream",i,MAXSTREAMS);
  return 0;
}

nvtxDomainHandle_t getdomain(int i){
    static int firstcall=1;
    const int MAXDOMAINS=1;
    static nvtxDomainHandle_t h[MAXDOMAINS];
    if (firstcall){
      h[0]= nvtxDomainCreateA("HYPRE");
      firstcall=0;
    }
    if (i<MAXDOMAINS) return h[i];
    fprintf(stderr,"ERROR in getdomain in utilities/gpuMem.c %d  is greater than MAXDOMAINS = %d \n Returning default domain",i,MAXDOMAINS);
    return NULL;
  }

cudaEvent_t getevent(int i){
  static int firstcall=1;
  const int MAXEVENTS=10;
  static cudaEvent_t s[MAXEVENTS];
  if (firstcall){
    for(int jj=0;jj<MAXEVENTS;jj++)
      gpuErrchk(cudaEventCreateWithFlags(&s[jj],cudaEventDisableTiming));
    //printf("Created events ..\n");
    firstcall=0;
  }
  if (i<MAXEVENTS) return s[i];
  fprintf(stderr,"ERROR in getevent in utilities/gpuMem.c %d is greater than MAXEVENTS = %d\n Returning default stream",i,MAXEVENTS);
  return 0;
}
int getsetasyncmode(int mode, int action){
  static int async_mode=0;
  if (action==0) async_mode = mode;
  if (action==1) return async_mode;
  return async_mode;
}
void SetAsyncMode(int mode){
  getsetasyncmode(mode,0);
}
int GetAsyncMode(){
  return getsetasyncmode(0,1);
}
void branchStream(int i, int j){
  gpuErrchk(cudaEventRecord(getevent(i),getstream(i)));
  gpuErrchk(cudaStreamWaitEvent(getstream(j),getevent(i),0));
}
void joinStreams(int i, int j, int k){
  gpuErrchk(cudaEventRecord(getevent(i),getstream(i)));
  gpuErrchk(cudaEventRecord(getevent(j),getstream(j)));
  gpuErrchk(cudaStreamWaitEvent(getstream(k),getevent(i),0));
  gpuErrchk(cudaStreamWaitEvent(getstream(k),getevent(j),0));
}
#endif
