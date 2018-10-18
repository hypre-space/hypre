#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "_hypre_utilities.h"

#if defined(HYPRE_USING_UNIFIED_MEMORY)
size_t memsize(const void *ptr){
   return ((size_t*)ptr)[-HYPRE_MEM_PAD_LEN];
}
#else
size_t memsize(const void *ptr){
  return 0;
}
#endif

#if defined(HYPRE_USING_CUDA)
HYPRE_Int hypre_exec_policy = HYPRE_MEMORY_DEVICE;
#endif

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)

hypre_int ggc(hypre_int id);

/* Global struct that holds device,library handles etc */
struct hypre__global_struct hypre__global_handle = {0, 0, 1, 0};

/* Initialize GPU branch of Hypre AMG */
/* use_device =-1 */
/* Application passes device number it is using or -1 to let Hypre decide on which device to use */
void hypre_GPUInit(hypre_int use_device)
{
   char pciBusId[80];
   hypre_int myid;
   hypre_int nDevices;
   //hypre_int device;
#if defined(TRACK_MEMORY_ALLOCATIONS)
   hypre_printf("\n\n\n WARNING :: TRACK_MEMORY_ALLOCATIONS IS ON \n\n");
#endif /* TRACK_MEMORY_ALLOCATIONS */

   if (!HYPRE_GPU_HANDLE)
   {
      HYPRE_GPU_HANDLE=1;
      HYPRE_DEVICE=0;
      hypre_CheckErrorDevice(cudaGetDeviceCount(&nDevices));

      /* XXX */
      nDevices = 1; /* DO NOT COMMENT ME OUT AGAIN! nDevices does NOT WORK !!!! */
      HYPRE_DEVICE_COUNT=nDevices;

      /* TODO cannot use nDevices to check if mpibind is used, need to rewrite
       * E.g., NP=5 on 2 nodes, nDevices=1,1,1,1,4 */

      if (use_device<0)
      {
         if (nDevices<4)
         {
            /* with mpibind each process will only see 1 GPU */
            HYPRE_DEVICE=0;
            hypre_CheckErrorDevice(cudaSetDevice(HYPRE_DEVICE));
            cudaDeviceGetPCIBusId ( pciBusId, 80, HYPRE_DEVICE);
            //hypre_printf("num Devices %d\n", nDevices);
         }
         else if (nDevices==4)
         {
            // THIS IS A HACK THAT WORKS ONLY AT LLNL
            /* No mpibind or it is a single rank run */
            hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
            //affs(myid);
            hypre_MPI_Comm node_comm;
            hypre_MPI_Info info;
            hypre_MPI_Info_create(&info);
            hypre_MPI_Comm_split_type(hypre_MPI_COMM_WORLD, hypre_MPI_COMM_TYPE_SHARED, myid, info, &node_comm);
            hypre_int round_robin=1;
            hypre_int myNodeid, NodeSize;
            hypre_MPI_Comm_rank(node_comm, &myNodeid);
            hypre_MPI_Comm_size(node_comm, &NodeSize);
            if (round_robin)
            {
               /* Round robin allocation of GPUs. Does not account for affinities */
               HYPRE_DEVICE=myNodeid%nDevices;
               hypre_CheckErrorDevice(cudaSetDevice(HYPRE_DEVICE));
               cudaDeviceGetPCIBusId ( pciBusId, 80, HYPRE_DEVICE);
               //hypre_printf("WARNING:: Code running without mpibind\n");
               //hypre_printf("Global ID = %d , Node ID %d running on device %d of %d \n",myid,myNodeid,HYPRE_DEVICE,nDevices);
            }
            else
            {
               /* Try to set the GPU based on process binding */
               /* works correcly for all cases */
               hypre_MPI_Comm numa_comm;
               hypre_MPI_Comm_split(node_comm,getnuma(),myNodeid,&numa_comm);
               hypre_int myNumaId,NumaSize;
               hypre_MPI_Comm_rank(numa_comm, &myNumaId);
               hypre_MPI_Comm_size(numa_comm, &NumaSize);
               hypre_int domain_devices=nDevices/2; /* Again hardwired for 2 NUMA domains */
               HYPRE_DEVICE = getnuma()*2+myNumaId%domain_devices;
               hypre_CheckErrorDevice(cudaSetDevice(HYPRE_DEVICE));
               //hypre_printf("WARNING:: Code running without mpibind\n");
               //hypre_printf("NUMA %d GID %d , NodeID %d NumaID %d running on device %d (RR=%d) of %d \n",getnuma(),myid,myNodeid,myNumaId,HYPRE_DEVICE,myNodeid%nDevices,nDevices);

            }
            hypre_MPI_Info_free(&info);
         }
         else
         {
            /* No device found  */
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR:: NO GPUS found \n");
            exit(2);
         }
      }
      else
      {
         HYPRE_DEVICE = use_device;
         hypre_CheckErrorDevice(cudaSetDevice(HYPRE_DEVICE));
      }

#if defined(HYPRE_USING_OPENMP_OFFLOAD) || defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
      omp_set_default_device(HYPRE_DEVICE);
      //printf("Set OMP Default device to %d \n",HYPRE_DEVICE);
#endif

      /* Create NVTX domain for all the nvtx calls in HYPRE */
      HYPRE_DOMAIN=nvtxDomainCreateA("Hypre");

      /* Initialize streams */
      hypre_int jj;
      for(jj=0;jj<MAX_HGS_ELEMENTS;jj++)
         hypre_CheckErrorDevice(cudaStreamCreateWithFlags(&(HYPRE_STREAM(jj)),cudaStreamNonBlocking));

      /* Initialize the library handles and streams */

      cusparseErrchk(cusparseCreate(&(HYPRE_CUSPARSE_HANDLE)));
      cusparseErrchk(cusparseSetStream(HYPRE_CUSPARSE_HANDLE,HYPRE_STREAM(4)));
      //cusparseErrchk(cusparseSetStream(HYPRE_CUSPARSE_HANDLE,0)); // Cusparse MxV happens in default stream
      cusparseErrchk(cusparseCreateMatDescr(&(HYPRE_CUSPARSE_MAT_DESCR)));
      cusparseErrchk(cusparseSetMatType(HYPRE_CUSPARSE_MAT_DESCR,CUSPARSE_MATRIX_TYPE_GENERAL));
      cusparseErrchk(cusparseSetMatIndexBase(HYPRE_CUSPARSE_MAT_DESCR,CUSPARSE_INDEX_BASE_ZERO));

      cublasErrchk(cublasCreate(&(HYPRE_CUBLAS_HANDLE)));
      cublasErrchk(cublasSetStream(HYPRE_CUBLAS_HANDLE,HYPRE_STREAM(4)));
      //if (!checkDeviceProps()) hypre_printf("WARNING:: Concurrent memory access not allowed\n");

      /* Check if the arch flags used for compiling the cuda kernels match the device */
#if defined(HYPRE_USING_GPU)
      CudaCompileFlagCheck();
#endif
   }
}


void hypre_GPUFinalize()
{
   cusparseErrchk(cusparseDestroy(HYPRE_CUSPARSE_HANDLE));

   cublasErrchk(cublasDestroy(HYPRE_CUBLAS_HANDLE));

#if defined(HYPRE_MEASURE_GPU_HWM)
   hypre_printf("GPU Memory High Water Mark(per MPI_RANK) %f MB \n",(HYPRE_Real)HYPRE_GPU_HWM/1024/1024);
#endif
   /* Destroy streams */
   hypre_int jj;
   for(jj=0;jj<MAX_HGS_ELEMENTS;jj++)
   {
      hypre_CheckErrorDevice(cudaStreamDestroy(HYPRE_STREAM(jj)));
   }
}

void MemAdviseReadOnly(const void* ptr, hypre_int device){
  if (ptr==NULL) return;
    size_t size=mempush(ptr,0,0);
    //if (size==0) hypre_printf("WARNING:: Operations with 0 size vector \n");
    hypre_CheckErrorDevice(cudaMemAdvise(ptr,size,cudaMemAdviseSetReadMostly,device));
}

void MemAdviseUnSetReadOnly(const void* ptr, hypre_int device){
  if (ptr==NULL) return;
    size_t size=mempush(ptr,0,0);
    //if (size==0) hypre_printf("WARNING:: Operations with 0 size vector \n");
    hypre_CheckErrorDevice(cudaMemAdvise(ptr,size,cudaMemAdviseUnsetReadMostly,device));
}


void MemAdviseSetPrefLocDevice(const void *ptr, hypre_int device){
  if (ptr==NULL) return;
  hypre_CheckErrorDevice(cudaMemAdvise(ptr,mempush(ptr,0,0),cudaMemAdviseSetPreferredLocation,device));
}

void MemAdviseSetPrefLocHost(const void *ptr){
  if (ptr==NULL) return;
  hypre_CheckErrorDevice(cudaMemAdvise(ptr,mempush(ptr,0,0),cudaMemAdviseSetPreferredLocation,cudaCpuDeviceId));
}

void MemPrefetch(const void *ptr,hypre_int device,cudaStream_t stream){
  if (ptr==NULL) return;
  size_t size;
  size=memsize(ptr);
  PUSH_RANGE("MemPreFetchForce",4);
  /* Do a prefetch every time until a possible UM bug is fixed */
  if (size>0){
     PrintPointerAttributes(ptr);
     hypre_CheckErrorDevice(cudaMemPrefetchAsync(ptr,size,device,stream));
     hypre_CheckErrorDevice(cudaStreamSynchronize(stream));
     POP_RANGE;
     return;
  }
  return;
}


void MemPrefetchForce(const void *ptr,hypre_int device,cudaStream_t stream){
  if (ptr==NULL) return;
  size_t size=memsize(ptr);
  PUSH_RANGE_PAYLOAD("MemPreFetchForce",4,size);
  hypre_CheckErrorDevice(cudaMemPrefetchAsync(ptr,size,device,stream));
  POP_RANGE;
  return;
}

void MemPrefetchSized(const void *ptr,size_t size,hypre_int device,cudaStream_t stream){
  if (ptr==NULL) return;
  PUSH_RANGE_DOMAIN("MemPreFetchSized",4,0);
  /* Do a prefetch every time until a possible UM bug is fixed */
  if (size>0){
    hypre_CheckErrorDevice(cudaMemPrefetchAsync(ptr,size,device,stream));
    POP_RANGE_DOMAIN(0);
    return;
  }
  return;
}


/* Returns the same cublas handle with every call */
cublasHandle_t getCublasHandle(){
  cublasStatus_t stat;
  static cublasHandle_t handle;
  static hypre_int firstcall=1;
  if (firstcall){
    firstcall=0;
    stat = cublasCreate(&handle);
    if (stat!=CUBLAS_STATUS_SUCCESS) {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR:: CUBLAS Library initialization failed\n");
      handle=0;
      exit(2);
    }
    cublasErrchk(cublasSetStream(handle,HYPRE_STREAM(4)));
  } else return handle;
  return handle;
}

/* Returns the same cusparse handle with every call */
cusparseHandle_t getCusparseHandle(){
  cusparseStatus_t status;
  static cusparseHandle_t handle;
  static hypre_int firstcall=1;
  if (firstcall){
    firstcall=0;
    status= cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR:: CUSPARSE Library initialization failed\n");
      handle=0;
      exit(2);
    }
    cusparseErrchk(cusparseSetStream(handle,HYPRE_STREAM(4)));
  } else return handle;
  return handle;
}


cudaStream_t getstreamOlde(hypre_int i){
  static hypre_int firstcall=1;
  const hypre_int MAXSTREAMS=10;
  static cudaStream_t s[MAXSTREAMS];
  if (firstcall){
    hypre_int jj;
    for(jj=0;jj<MAXSTREAMS;jj++)
      hypre_CheckErrorDevice(cudaStreamCreateWithFlags(&s[jj],cudaStreamNonBlocking));
    //printf("Created streams ..\n");
    firstcall=0;
  }
  if (i<MAXSTREAMS) return s[i];
  //fprintf(stderr,"ERROR in HYPRE_STREAM in utilities/gpuMem.c %d is greater than MAXSTREAMS = %d\n Returning default stream",i,MAXSTREAMS);
  return 0;
}

nvtxDomainHandle_t getdomain(hypre_int i){
    static hypre_int firstcall=1;
    const hypre_int MAXDOMAINS=1;
    static nvtxDomainHandle_t h[MAXDOMAINS];
    if (firstcall){
      h[0]= nvtxDomainCreateA("HYPRE_A");
      firstcall=0;
    }
    if (i<MAXDOMAINS) return h[i];
   // fprintf(stderr,"ERROR in getdomain in utilities/gpuMem.c %d  is greater than MAXDOMAINS = %d \n Returning default domain",i,MAXDOMAINS);
    return NULL;
  }

cudaEvent_t getevent(hypre_int i){
  static hypre_int firstcall=1;
  const hypre_int MAXEVENTS=10;
  static cudaEvent_t s[MAXEVENTS];
  if (firstcall){
    hypre_int jj;
    for(jj=0;jj<MAXEVENTS;jj++)
       hypre_CheckErrorDevice(cudaEventCreateWithFlags(&s[jj],cudaEventDisableTiming));
    //printf("Created events ..\n");
    firstcall=0;
  }
  if (i<MAXEVENTS) return s[i];
  //fprintf(stderr,"ERROR in getevent in utilities/gpuMem.c %d is greater than MAXEVENTS = %d\n Returning default stream",i,MAXEVENTS);
  return 0;
}

hypre_int getsetasyncmode(hypre_int mode, hypre_int action){
  static hypre_int async_mode=0;
  if (action==0) async_mode = mode;
  if (action==1) return async_mode;
  return async_mode;
}

void SetAsyncMode(hypre_int mode){
  getsetasyncmode(mode,0);
}

hypre_int GetAsyncMode(){
  return getsetasyncmode(0,1);
}

void branchStream(hypre_int i, hypre_int j){
   hypre_CheckErrorDevice(cudaEventRecord(getevent(i),HYPRE_STREAM(i)));
   hypre_CheckErrorDevice(cudaStreamWaitEvent(HYPRE_STREAM(j),getevent(i),0));
}

void joinStreams(hypre_int i, hypre_int j, hypre_int k){
   hypre_CheckErrorDevice(cudaEventRecord(getevent(i),HYPRE_STREAM(i)));
   hypre_CheckErrorDevice(cudaEventRecord(getevent(j),HYPRE_STREAM(j)));
   hypre_CheckErrorDevice(cudaStreamWaitEvent(HYPRE_STREAM(k),getevent(i),0));
   hypre_CheckErrorDevice(cudaStreamWaitEvent(HYPRE_STREAM(k),getevent(j),0));
}

#include <sched.h>
#include <errno.h>

void affs(hypre_int myid){
  const hypre_int NCPUS=160;
  cpu_set_t* mask = CPU_ALLOC(NCPUS);
  size_t size = CPU_ALLOC_SIZE(NCPUS);
  hypre_int cpus[NCPUS],i;
  hypre_int retval=sched_getaffinity(0, size,mask);
  if (!retval){
     for(i=0;i<NCPUS;i++){
        if (CPU_ISSET(i,mask))
           cpus[i]=1;
        else
           cpus[i]=0;
     }
     printf("Node(%d)::",myid);
     for(i=0;i<160;i++)printf("%d",cpus[i]);
     printf("\n");
  } else {
     fprintf(stderr,"sched_affinity failed\n");
     switch(errno){
        case EFAULT:
           printf("INVALID MEMORY ADDRESS\n");
           break;
        case EINVAL:
           printf("EINVAL:: NO VALID CPUS\n");
           break;
        default:
           printf("%d something else\n",errno);
     }
  }

  CPU_FREE(mask);

}
hypre_int getcore(){
  const hypre_int NCPUS=160;
  cpu_set_t* mask = CPU_ALLOC(NCPUS);
  size_t size = CPU_ALLOC_SIZE(NCPUS);
  hypre_int /*cpus[NCPUS],*/i;
  hypre_int retval=sched_getaffinity(0, size,mask);
  if (!retval){
    for(i=0;i<NCPUS;i+=20){
      if (CPU_ISSET(i,mask)) {
         CPU_FREE(mask);
         return i;
      }
    }
  } else {
    hypre_error_w_msg(HYPRE_ERROR_GENERIC,"sched_affinity failed\n");
    switch(errno){
    case EFAULT:
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"INVALID MEMORY ADDRESS\n");
      break;
    case EINVAL:
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"EINVAL:: NO VALID CPUS\n");
      break;
    default:
      hypre_error_w_msg(HYPRE_ERROR_GENERIC," something else\n");
    }
  }
  return hypre_error_flag;
  //CPU_FREE(mask);

}
hypre_int getnuma(){
  const hypre_int NCPUS=160;
  cpu_set_t* mask = CPU_ALLOC(NCPUS);
  size_t size = CPU_ALLOC_SIZE(NCPUS);
  hypre_int retval=sched_getaffinity(0, size,mask);
  /* HARDWIRED FOR 2 NUMA DOMAINS */
  if (!retval){
    hypre_int sum0=0,i;
    for(i=0;i<NCPUS/2;i++)
      if (CPU_ISSET(i,mask)) sum0++;
    hypre_int sum1=0;
    for(i=NCPUS/2;i<NCPUS;i++)
      if (CPU_ISSET(i,mask)) sum1++;
    CPU_FREE(mask);
    if (sum0>sum1) return 0;
    else return 1;
  } else {
    hypre_error_w_msg(HYPRE_ERROR_GENERIC,"sched_affinity failed\n");
    switch(errno){
    case EFAULT:
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"INVALID MEMORY ADDRESS\n");
      break;
    case EINVAL:
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"EINVAL:: NO VALID CPUS\n");
      break;
    default:
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"something else\n");
    }
  }
  return hypre_error_flag;
  //CPU_FREE(mask);

}
hypre_int checkDeviceProps(){
  struct cudaDeviceProp prop;
  hypre_CheckErrorDevice(cudaGetDeviceProperties(&prop, HYPRE_DEVICE));
  HYPRE_GPU_CMA=prop.concurrentManagedAccess;
  return HYPRE_GPU_CMA;
}
hypre_int pointerIsManaged(const void *ptr){
  struct cudaPointerAttributes ptr_att;
  if (cudaPointerGetAttributes(&ptr_att,ptr)!=cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return ptr_att.isManaged;
}

/* C version of mempush using linked lists */

size_t mempush(const void *ptr, size_t size, hypre_int action){
  static node* head=NULL;
  static hypre_int nc=0;
  node *found=NULL;
  if (!head){
    if ((size<=0)||(action==1)) {
      fprintf(stderr,"mempush can start only with an insertion or a size call \n");
      return 0;
    }
    head = hypre_TAlloc(node, 1, HYPRE_MEMORY_HOST);
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
  nhead = hypre_TAlloc(node, 1, HYPRE_MEMORY_HOST);
  nhead->ptr=ptr;
  nhead->size=size;
  nhead->next=*head;
  *head=nhead;
  return;
}

void printlist(node *head,hypre_int nc){
  node *next;
  next=head;
  printf("Node count %d \n",nc);
  while(next!=NULL){
    printf("Address %p of size %zu \n",next->ptr,next->size);
    next=next->next;
  }
}

#endif /* #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP) */

#if defined(HYPRE_USING_DEVICE_OPENMP)

#include <omp.h>

/* num: number of bytes */
HYPRE_Int HYPRE_OMPOffload(HYPRE_Int device, void *ptr, size_t num,
                           const char *type1, const char *type2) {
   hypre_omp45_offload(device, ptr, char, 0, num, type1, type2);

   return 0;
}

HYPRE_Int HYPRE_OMPPtrIsMapped(void *p, HYPRE_Int device_num)
{
   if (hypre__global_offload && !omp_target_is_present(p, device_num)) {
      printf("HYPRE mapping error: %p has not been mapped to device %d!\n", p, device_num);
      return 1;
   }
   return 0;
}

/* OMP offloading switch */
HYPRE_Int HYPRE_OMPOffloadOn()
{
   hypre__global_offload = 1;
   hypre__offload_device_num = omp_get_default_device();
   hypre__offload_host_num   = omp_get_initial_device();
   hypre_fprintf(stdout, "Hypre OMP 4.5 offloading has been turned on. Device %d\n",
                 hypre__offload_device_num);

   return 0;
}

HYPRE_Int HYPRE_OMPOffloadOff()
{
   fprintf(stdout, "Hypre OMP 4.5 offloading has been turned off\n");
   hypre__global_offload = 0;
   hypre__offload_device_num = omp_get_initial_device();
   hypre__offload_host_num   = omp_get_initial_device();

   return 0;
}

HYPRE_Int HYPRE_OMPOffloadStatPrint() {
   hypre_printf("Hypre OMP target memory stats:\n"
                "      ALLOC   %ld bytes, %ld counts\n"
                "      FREE    %ld bytes, %ld counts\n"
                "      HTOD    %ld bytes, %ld counts\n"
                "      DTOH    %ld bytes, %ld counts\n",
                hypre__target_allc_bytes, hypre__target_allc_count,
                hypre__target_free_bytes, hypre__target_free_count,
                hypre__target_htod_bytes, hypre__target_htod_count,
                hypre__target_dtoh_bytes, hypre__target_dtoh_count);

   return 0;
}

#endif /* #if defined(HYPRE_USING_DEVICE_OPENMP) */

