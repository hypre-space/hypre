#include "_hypre_utilities.h"


#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)

void CheckError(cudaError_t const err, const char* file, char const* const fun, const HYPRE_Int line)
{
    if (err)
    {
      printf("CUDA Error Code[%d]: %s\n %s(%s) Line:%d\n", err, cudaGetErrorString(err), file, fun, line);
      HYPRE_Int *p = NULL; *p = 1;
    }
}

/*
extern const char *cusparseErrorCheck(cusparseStatus_t error);
extern void gpuAssert(cudaError_t code, const char *file, int line);
extern void cusparseAssert(cusparseStatus_t code, const char *file, int line);
*/

/*
  cudaSafeFree frees Managed memory allocated in hypre_MAlloc,hypre_CAlloc and hypre_ReAlloc
  It checks if the memory is managed before freeing and emits a warning if it is not memory
  allocated using the above routines. This behaviour can be changed by defining ABORT_ON_RAW_POINTER.
  The core file can then be used to find the location of the anomalous hypre_Free.
 */
void cudaSafeFree(void *ptr,int padding)
{
  PUSH_RANGE("SAFE_FREE",3);
  struct cudaPointerAttributes ptr_att;
  size_t *sptr=(size_t*)ptr-padding;
  cudaError_t err;

  err=cudaPointerGetAttributes(&ptr_att,ptr);
  if (err!=cudaSuccess)
  {
    cudaGetLastError(); 
#ifndef ABORT_ON_RAW_POINTER
#ifdef FULL_WARN
    if (err==cudaErrorInvalidValue)
    {
       fprintf(stderr,"WARNING :: Raw pointer passed to cudaSafeFree %p\n",ptr);
    }
    else if (err==cudaErrorInvalidDevice)
    {
       fprintf(stderr,"WARNING :: cudaSafeFree :: INVALID DEVICE on ptr = %p\n",ptr);
    }
    else if (err==cudaErrorIncompatibleDriverContext)
    {
       fprintf(stderr,"WARNING :: cudaSafeFree :: Incompatible  Driver Context on ptr = %p\n",ptr);
    }
    else
    {
       fprintf(stderr,"Point Attrib check error is %d \n",err);
    }
    //PrintPointerAttributes(ptr);
#endif /* FULL_WARN */
#else
    fprintf(stderr,"ERROR:: cudaSafeFree Aborting on raw unmanaged pointer %p\n",ptr);
    raise(SIGABRT);
#endif /* ABORT_ON_RAW_POINTER */
    free(ptr); /* Free the nonManaged pointer */
    return;
  }

  if (ptr_att.isManaged)
  {
#if defined(HYPRE_MEASURE_GPU_HWM)
     size_t mfree,mtotal;
     hypre_CheckErrorDevice(cudaMemGetInfo(&mfree,&mtotal));
     HYPRE_GPU_HWM = hypre_max((mtotal-mfree),HYPRE_GPU_HWM);
#endif
    /* Code below for handling managed memory pointers not allocated using hypre_CTAlloc oir hypre_TAlooc */
    if (PointerAttributes(ptr)!=PointerAttributes(sptr)){
      //fprintf(stderr,"ERROR IN Pointer for freeing %p %p\n",ptr,sptr);
      hypre_CheckErrorDevice(cudaFree(ptr)); 
      return;
    }
    hypre_CheckErrorDevice(cudaFree(sptr)); 
  } 
  else
  {
    /* It is a pinned memory pointer */
    //printf("ERROR:: NON-managed pointer passed to cudaSafeFree\n");
    if (ptr_att.memoryType==cudaMemoryTypeHost)
    {
      hypre_CheckErrorDevice(cudaFreeHost(sptr));
    } 
    else if (ptr_att.memoryType==cudaMemoryTypeDevice)
    {
      hypre_CheckErrorDevice(cudaFree(sptr)); 
    }
  }
  POP_RANGE;

  return;
}

hypre_int PrintPointerAttributes(const void *ptr){
  struct cudaPointerAttributes ptr_att;
#if defined(TRACK_MEMORY_ALLOCATIONS)
  pattr_t *ss = patpush(ptr,NULL);
  if (ss!=NULL) fprintf(stderr,"Pointer %p from line %d of %s TYPE = %d \n",ptr,ss->line,ss->file,ss->type);
#endif /* TRACK_MEMORY_ALLOCATIONS */
  if (cudaPointerGetAttributes(&ptr_att,ptr)!=cudaSuccess){
    cudaGetLastError();  // Required to reset error flag on device
    fprintf(stderr,"PrintPointerAttributes:: Raw pointer %p\n",ptr);
    return HYPRE_HOST_POINTER;
  }
  if (ptr_att.isManaged){
    fprintf(stderr,"PrintPointerAttributes:: Managed pointer\n");
    fprintf(stderr,"Host address = %p, Device Address = %p\n",ptr_att.hostPointer, ptr_att.devicePointer);
    if (ptr_att.memoryType==cudaMemoryTypeHost) fprintf(stderr,"Memory is located on host\n");
    if (ptr_att.memoryType==cudaMemoryTypeDevice) fprintf(stderr,"Memory is located on device\n");
    fprintf(stderr,"Device associated with this pointer is %d\n",ptr_att.device);
    return HYPRE_MANAGED_POINTER;
  }
  else {
    fprintf(stderr,"PrintPointerAttributes:: Non-Managed & non-raw pointer\n Probably pinned host pointer\n");
    if (ptr_att.memoryType==cudaMemoryTypeHost) {
      fprintf(stderr,"Memory is located on host\n");
      return HYPRE_PINNED_POINTER;
    }
    if (ptr_att.memoryType==cudaMemoryTypeDevice) {
      fprintf(stderr,"Memory is located on device\n");
      return HYPRE_DEVICE_POINTER ;
    }
    return HYPRE_UNDEFINED_POINTER1;
  }
}
hypre_int PointerAttributes(const void *ptr){
  struct cudaPointerAttributes ptr_att;
  if (cudaPointerGetAttributes(&ptr_att,ptr)!=cudaSuccess){
     cudaGetLastError();  // Required to  reset error flag on device
     return HYPRE_HOST_POINTER;
  }
  if (ptr_att.isManaged){
    return HYPRE_MANAGED_POINTER; 
  }
  else {
    if (ptr_att.memoryType==cudaMemoryTypeHost) return HYPRE_PINNED_POINTER; /* Host pointer from cudaMallocHost */
    if (ptr_att.memoryType==cudaMemoryTypeDevice) return HYPRE_DEVICE_POINTER ; /* cudadevice pointer */
    return HYPRE_UNDEFINED_POINTER1; /* Shouldn't happen */
  }
}

#if defined(TRACK_MEMORY_ALLOCATIONS)
void assert_check(void *ptr, char *file, int line){
  if (ptr==NULL) return;
  pattr_t *ss = patpush(ptr,NULL);
  if (ss!=NULL)
    {
      if (ss->type!=HYPRE_MEMORY_SHARED)
	{
	  fprintf(stderr,"ASSERT_MANAGED FAILURE in line %d of file %s type = %d pomitrt = %p\n",line,file,ss->type,ptr);
	  fprintf(stderr,"ASSERT_MANAGED failed on allocation from line %d of %s \n",ss->line,ss->file);
	}
    }
  else
    {
      if ( PointerAttributes(ptr)!=HYPRE_MANAGED_POINTER){
	fprintf(stderr,"ASSERT_MANAGED FAILURE in line %d of file %s \n NO ALLOCATION INFO\n",line,file);
	PrintPointerAttributes(ptr);
      }
    }
  
}
void assert_check_host(void *ptr, char *file, int line){
  if (ptr==NULL) return;
  pattr_t *ss = patpush(ptr,NULL);
  if (ss!=NULL)
    {
      if (ss->type!=HYPRE_MEMORY_HOST)
	{
	  fprintf(stderr,"ASSERT_HOST FAILURE in line %d of file %s type = %d pomitrt = %p\n",line,file,ss->type,ptr);
	  fprintf(stderr,"ASSERT_HOST failed on allocation from line %d of %s of size %d bytes \n",ss->line,ss->file,ss->size);
	}
    }
  else
    {
      //printf("Address not in map\n Calling PrintPointerAttributes\n");
      if ( PointerAttributes(ptr)!=HYPRE_HOST_POINTER){
	fprintf(stderr,"ASSERT_HOST FAILURE in line %d of file %s \n NO ALLOCATION INFO\n",line,file);
      }
    }
  
}

#endif /* TRACK_MEMORY_ALLOCATIONS */

#endif 
