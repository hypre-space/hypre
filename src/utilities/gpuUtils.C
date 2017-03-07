#ifdef HYPRE_USE_GPU
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include "gpuErrorCheck.h"
//#include "hypre_nvtx.h"
#include "nvToolsExt.h"
extern "C"{
  // Deletes the record for size <0, returns the size for size==0 and sets it for size>0
  size_t mempush(const void *ptr, size_t size,int purge){
  static std::unordered_map<const void*,size_t> omap;
  // Error checking for  random pointers that have accidentally wandered in
  bool found=false;
  
  // if (purge==2){
  //    for (const auto &x : omap) std::cout<<"ALLOCATION "<<x.second<<"\n";
  //   return 0;
  // }
  // Delete maps if size < 0
#ifdef MEMLOG
  static int firstcall=1;
  static std::ofstream of;
  if (firstcall){
    of.open("memory.log");
    firstcall=0;
  }
#endif
  if (purge==1) {
    found=(omap.find(ptr)!=omap.end());
    if (found){
      int osize=omap[ptr];
      omap.erase(ptr);
#ifdef MEMLOG
      of<<ptr<<" "<<size<<" "<<purge<<"\n";
#endif
      return osize;
    } else {
#ifdef GPU_WARN
      std::cerr<<" ERROR :: Pointer for map deletetion not in map \n";
#endif
      return 0;
    }
  }

  // Insert into omap if size is greater than zero

  if (size>0) {
    found=(omap.find(ptr)!=omap.end());
    if (found){
#ifdef GPU_WARN
      std::cerr<<"ERROR:: Pointer for map insertion already exists :: "<<ptr<<" of size "<<omap[ptr]<<" new size = "<<size<<"\n";
#endif
      return 0;
    } else {
      omap[ptr]=size;
#ifdef MEMLOG
      of<<ptr<<" "<<size<<" "<<purge<<"\n";
#endif
    }
    return omap[ptr];
  }


  
  // Query size of pointer allocation using size=0
  found=(omap.find(ptr)!=omap.end());
  if (found)
    return omap[ptr];
  else {
    //std::cerr<<"WARNING:mempush Pointer is not mapped "<<ptr<<" size "<<size<<" purge "<<purge<<"\n";
    return 0;
  }

  
}
  int memloc(const void *ptr, int device){
    static std::unordered_map<const void*,int> omap;
    bool found=false;
    found=(omap.find(ptr)!=omap.end());
    if (found){
      if (omap[ptr]==device){
        //std::cout<<" Data already on device "<<device<<" "<<ptr<<"\n";
	return 0;
      } else {
        //std::cout<<" Strange Data not on device "<<device<<" "<<ptr<<"\n";
	omap[ptr]=device;
	return 1;
      }
    } else {
      omap[ptr]=device;  
      //std::cout<<" First move to "<<device<<" "<<ptr<<"\n";
      return 1;
    }
  }
    
}


extern "C"{
  cudaStream_t getstream(int i){
    static int firstcall=1;
    const int MAXSTREAMS=10;
    static cudaStream_t s[MAXSTREAMS];
    if (firstcall){
      for(int jj=0;jj<MAXSTREAMS;jj++)
	gpuErrchk(cudaStreamCreateWithFlags(&s[jj],cudaStreamNonBlocking));
      std::cout<<"Created streams ::";
      for(int jj=0;jj<MAXSTREAMS;jj++) std::cout<<s[jj]<<",";
      std::cout<<"\n";
      firstcall=0;
      //nvtxNameCudaStream(s[4], "HYPRE_COMPUTE_STREAM");
    }
    if (i<MAXSTREAMS) return s[i];
    std::cerr<<"ERROR in getstream in utilities/gpuUtils.C "<<i<<" is greater than MAXSTREAMS "<<MAXSTREAMS<<"\n Returning default stream";
    return 0;
  }
}

extern"C" {
  nvtxDomainHandle_t getdomain(int i){
    static int firstcall=1;
    const int MAXDOMAINS=1;
    static nvtxDomainHandle_t h[MAXDOMAINS];
    if (firstcall){
      h[0]= nvtxDomainCreateA("HYPRE");
      firstcall=0;
    }
    if (i<MAXDOMAINS) return h[i];
    std::cerr<<"ERROR in getdomain in utilities/gpuUtils.C "<<i<<" is greater than MAXDOMAINS "<<MAXDOMAINS<<"\n Returning default domain";
    return NULL;
  }
}
#endif
