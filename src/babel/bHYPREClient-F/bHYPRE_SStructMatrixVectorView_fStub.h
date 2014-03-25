/*
 * File:          bHYPRE_SStructMatrixVectorView_fStub.h
 * Symbol:        bHYPRE.SStructMatrixVectorView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.4
 * Description:   Client-side documentation text for bHYPRE.SStructMatrixVectorView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructMatrixVectorView_fStub_h
#define included_bHYPRE_SStructMatrixVectorView_fStub_h

/**
 * Symbol "bHYPRE.SStructMatrixVectorView" (version 1.0.0)
 */

#ifndef included_bHYPRE_SStructMatrixVectorView_IOR_h
#include "bHYPRE_SStructMatrixVectorView_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_SStructMatrixVectorView__connectI

#pragma weak bHYPRE_SStructMatrixVectorView__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructMatrixVectorView__object*
bHYPRE_SStructMatrixVectorView__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructMatrixVectorView__object*
bHYPRE_SStructMatrixVectorView__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
