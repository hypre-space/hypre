/*
 * File:          bHYPRE_SStructMatrixView_jniStub.h
 * Symbol:        bHYPRE.SStructMatrixView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.4
 * Description:   Client-side header code for bHYPRE.SStructMatrixView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructMatrixView_jniStub_h
#define included_bHYPRE_SStructMatrixView_jniStub_h

/**
 * Symbol "bHYPRE.SStructMatrixView" (version 1.0.0)
 */

#ifndef included_bHYPRE_SStructMatrixView_IOR_h
#include "bHYPRE_SStructMatrixView_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_SStructMatrixView__connectI

#pragma weak bHYPRE_SStructMatrixView__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructMatrixView__object*
bHYPRE_SStructMatrixView__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructMatrixView__object*
bHYPRE_SStructMatrixView__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
