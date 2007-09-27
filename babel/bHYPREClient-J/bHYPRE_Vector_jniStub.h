/*
 * File:          bHYPRE_Vector_jniStub.h
 * Symbol:        bHYPRE.Vector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.4
 * Description:   Client-side header code for bHYPRE.Vector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_Vector_jniStub_h
#define included_bHYPRE_Vector_jniStub_h

/**
 * Symbol "bHYPRE.Vector" (version 1.0.0)
 */

#ifndef included_bHYPRE_Vector_IOR_h
#include "bHYPRE_Vector_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_Vector__connectI

#pragma weak bHYPRE_Vector__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_Vector__object*
bHYPRE_Vector__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_Vector__object*
bHYPRE_Vector__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
