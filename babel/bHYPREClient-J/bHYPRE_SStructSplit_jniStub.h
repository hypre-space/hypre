/*
 * File:          bHYPRE_SStructSplit_jniStub.h
 * Symbol:        bHYPRE.SStructSplit-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side header code for bHYPRE.SStructSplit
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructSplit_jniStub_h
#define included_bHYPRE_SStructSplit_jniStub_h

/**
 * Symbol "bHYPRE.SStructSplit" (version 1.0.0)
 * 
 * 
 * The SStructSplit solver requires a SStruct matrix.
 */

#ifndef included_bHYPRE_SStructSplit_IOR_h
#include "bHYPRE_SStructSplit_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_SStructSplit__connectI

#pragma weak bHYPRE_SStructSplit__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructSplit__object*
bHYPRE_SStructSplit__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructSplit__object*
bHYPRE_SStructSplit__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
