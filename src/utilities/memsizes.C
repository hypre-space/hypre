#include <unordered_map>
#include <iostream>

//#include "_hypre_utilities.h"

using namespace std;

// Issues with patpush and TRACK_MEMORY_ALLOCATIONS
// 1. map entries need to be removed with frees
// 2. space used for pattr structs need to be de-allocated

extern "C"{
  typedef struct {
    char *file;
    size_t size;
    void *end;
    int line;
    int type;} pattr_t;
  
  pattr_t *patpush(void *ptr, pattr_t *ss){
    static std::unordered_map<void*,pattr_t *> map;
    pattr_t *retval=NULL;
#pragma omp critical
    {
      if (ss!=NULL) {
	map[ptr]=ss;
      } else {
	std::unordered_map<void*,pattr_t*>::const_iterator got = map.find (ptr);
	if (got==map.end()){
	  //std:cerr<<"ELEMENT NOT FOUND IN MAP\n";
	  // DO a range check for pointers which might be offsets 
	  for( const auto& k : map) {
	    if ((ptr>=k.first)&&(ptr<k.second->end)) {
	      //std::cerr<<"PTR found in range "<<k.first<<" "<<ptr<<" "<<k.second->end<<"\n";
	      retval=k.second;
	      break;
	    }
	  }
	  //if (!retval) std::cerr<<"PTR NOT found in range\n";
	} else
	  retval = got->second;
      }
    }
    return retval;
  }
  
}
