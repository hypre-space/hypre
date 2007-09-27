/*
 * File:          bHYPRE_IJVectorView_fStub.h
 * Symbol:        bHYPRE.IJVectorView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.4
 * Description:   Client-side documentation text for bHYPRE.IJVectorView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_IJVectorView_fStub_h
#define included_bHYPRE_IJVectorView_fStub_h

/**
 * Symbol "bHYPRE.IJVectorView" (version 1.0.0)
 */

#ifndef included_bHYPRE_IJVectorView_IOR_h
#include "bHYPRE_IJVectorView_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_IJVectorView__connectI

#pragma weak bHYPRE_IJVectorView__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_IJVectorView__object*
bHYPRE_IJVectorView__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_IJVectorView__object*
bHYPRE_IJVectorView__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
