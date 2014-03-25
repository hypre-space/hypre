/*
 * File:          bHYPRE_StructDiagScale_jniStub.h
 * Symbol:        bHYPRE.StructDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side header code for bHYPRE.StructDiagScale
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_StructDiagScale_jniStub_h
#define included_bHYPRE_StructDiagScale_jniStub_h

/**
 * Symbol "bHYPRE.StructDiagScale" (version 1.0.0)
 * 
 * Diagonal scaling preconditioner for STruct matrix class.
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 */

#ifndef included_bHYPRE_StructDiagScale_IOR_h
#include "bHYPRE_StructDiagScale_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_StructDiagScale__connectI

#pragma weak bHYPRE_StructDiagScale__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructDiagScale__object*
bHYPRE_StructDiagScale__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_StructDiagScale__object*
bHYPRE_StructDiagScale__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
