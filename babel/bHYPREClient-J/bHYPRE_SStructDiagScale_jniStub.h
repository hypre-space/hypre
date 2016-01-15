/*
 * File:          bHYPRE_SStructDiagScale_jniStub.h
 * Symbol:        bHYPRE.SStructDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side header code for bHYPRE.SStructDiagScale
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructDiagScale_jniStub_h
#define included_bHYPRE_SStructDiagScale_jniStub_h

/**
 * Symbol "bHYPRE.SStructDiagScale" (version 1.0.0)
 */

#ifndef included_bHYPRE_SStructDiagScale_IOR_h
#include "bHYPRE_SStructDiagScale_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_SStructDiagScale__connectI

#pragma weak bHYPRE_SStructDiagScale__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructDiagScale__object*
bHYPRE_SStructDiagScale__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructDiagScale__object*
bHYPRE_SStructDiagScale__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
