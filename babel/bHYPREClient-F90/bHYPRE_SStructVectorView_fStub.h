/*
 * File:          bHYPRE_SStructVectorView_fStub.h
 * Symbol:        bHYPRE.SStructVectorView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Description:   Client-side documentation text for bHYPRE.SStructVectorView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructVectorView_fStub_h
#define included_bHYPRE_SStructVectorView_fStub_h

/**
 * Symbol "bHYPRE.SStructVectorView" (version 1.0.0)
 */

#ifndef included_bHYPRE_SStructVectorView_IOR_h
#include "bHYPRE_SStructVectorView_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_SStructVectorView__connectI

#pragma weak bHYPRE_SStructVectorView__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructVectorView__object*
bHYPRE_SStructVectorView__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructVectorView__object*
bHYPRE_SStructVectorView__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
