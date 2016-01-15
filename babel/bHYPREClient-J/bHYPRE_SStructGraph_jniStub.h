/*
 * File:          bHYPRE_SStructGraph_jniStub.h
 * Symbol:        bHYPRE.SStructGraph-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side header code for bHYPRE.SStructGraph
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructGraph_jniStub_h
#define included_bHYPRE_SStructGraph_jniStub_h

/**
 * Symbol "bHYPRE.SStructGraph" (version 1.0.0)
 * 
 * The semi-structured grid graph class.
 */

#ifndef included_bHYPRE_SStructGraph_IOR_h
#include "bHYPRE_SStructGraph_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_SStructGraph__connectI

#pragma weak bHYPRE_SStructGraph__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructGraph__object*
bHYPRE_SStructGraph__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructGraph__object*
bHYPRE_SStructGraph__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
