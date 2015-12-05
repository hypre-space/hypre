/*
 * File:          bHYPRE_StructVectorView_fStub.h
 * Symbol:        bHYPRE.StructVectorView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Description:   Client-side documentation text for bHYPRE.StructVectorView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_StructVectorView_fStub_h
#define included_bHYPRE_StructVectorView_fStub_h

/**
 * Symbol "bHYPRE.StructVectorView" (version 1.0.0)
 */

#ifndef included_bHYPRE_StructVectorView_IOR_h
#include "bHYPRE_StructVectorView_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_StructVectorView__connectI

#pragma weak bHYPRE_StructVectorView__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructVectorView__object*
bHYPRE_StructVectorView__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_StructVectorView__object*
bHYPRE_StructVectorView__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
