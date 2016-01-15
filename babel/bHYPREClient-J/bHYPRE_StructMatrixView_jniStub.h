/*
 * File:          bHYPRE_StructMatrixView_jniStub.h
 * Symbol:        bHYPRE.StructMatrixView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.4
 * Description:   Client-side header code for bHYPRE.StructMatrixView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_StructMatrixView_jniStub_h
#define included_bHYPRE_StructMatrixView_jniStub_h

/**
 * Symbol "bHYPRE.StructMatrixView" (version 1.0.0)
 */

#ifndef included_bHYPRE_StructMatrixView_IOR_h
#include "bHYPRE_StructMatrixView_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_StructMatrixView__connectI

#pragma weak bHYPRE_StructMatrixView__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructMatrixView__object*
bHYPRE_StructMatrixView__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_StructMatrixView__object*
bHYPRE_StructMatrixView__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
