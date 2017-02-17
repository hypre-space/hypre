#ifdef HYPRE_USE_GPU
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <unordered_map>
#include "gpuErrorCheck.h"
extern "C"{
  // Deletes the record for size <0, returns the size for size==0 and sets it for size>0
  size_t mempush(void *ptr, size_t size,int purge){
  static std::unordered_map<void*,size_t> map;
  // Error checking for  random pointers that have accidentally wandered in
  bool found=false;


  // Delete maps if size < 0

  if (purge) {
    found=(map.find(ptr)!=map.end());
    if (found){
      int osize=map[ptr];
      map.erase(ptr);
      return osize;
    } else {
      std::cerr<<" ERROR :: Pointer for map deletetion not in map \n";
      return 0;
    }
  }

  // Insert into map if size is greater than zero

  if (size>0) {
    found=(map.find(ptr)!=map.end());
    if (found){
      std::cerr<<"ERROR:: Pointer for map insertion already exists :: "<<ptr<<" of size "<<map[ptr]<<" new size = "<<size<<"\n";
      return 0;
    } else map[ptr]=size;
    return map[ptr];
  }

  // Query size of pointer allocation using size=0
  found=(map.find(ptr)!=map.end());
  if (found)
    return map[ptr];
  else {
    std::cerr<<"ERROR:: Pointer is not mapped "<<ptr<<" size "<<size<<" purge "<<purge<<"\n";
    return 0;
  }
}
  int memloc(void *ptr, int device){
    static std::unordered_map<void*,int> map;
    bool found=false;
    found=(map.find(ptr)!=map.end());
    if (found){
      if (map[ptr]==device){
        //std::cout<<" Data already on device "<<device<<" "<<ptr<<"\n";
	return 0;
      } else {
        //std::cout<<" Strange Data not on device "<<device<<" "<<ptr<<"\n";
	map[ptr]=device;
	return 1;
      }
    } else {
      map[ptr]=device;  
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
    }
    if (i<MAXSTREAMS) return s[i];
    std::cerr<<"ERROR in getstream in utilities/streams.C "<<i<<" is greater than MAXSTREAMS "<<MAXSTREAMS<<"\n Returning default stream";
    return 0;
  }
}
#endif
