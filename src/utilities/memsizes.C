#include <unordered_map>
#include <iostream>

#include "_hypre_utilities.h"

using namespace std;

extern "C"{
  typedef struct {
    char *file;
    HYPRE_Int line;
    HYPRE_Int type;} pattr_t;
  
  pattr_t *patpush(void *ptr, pattr_t *ss){
    static std::unordered_map<void*,pattr_t *> map;
    if (ss!=NULL) {
      map[ptr]=ss;
    } else {
      std::unordered_map<void*,pattr_t*>::const_iterator got = map.find (ptr);
      if (got==map.end()){
	//std:cerr<<"ELEMENT NOT FOUND IN MAP\n";
	return NULL;
      } else
	return got->second;
    }
    return NULL;
  }
  
}
