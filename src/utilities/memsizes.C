#include <unordered_map>
#include <iostream>

using namespace std;

extern "C"{
  typedef struct {
    char *file;
    int line;
    int type;} pattr_t;
  
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
  }
  
}
