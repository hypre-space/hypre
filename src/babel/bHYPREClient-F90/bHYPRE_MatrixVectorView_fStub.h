/*
 * File:          bHYPRE_MatrixVectorView_fStub.h
 * Symbol:        bHYPRE.MatrixVectorView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Description:   Client-side documentation text for bHYPRE.MatrixVectorView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_MatrixVectorView_fStub_h
#define included_bHYPRE_MatrixVectorView_fStub_h

/**
 * Symbol "bHYPRE.MatrixVectorView" (version 1.0.0)
 * 
 * This interface is defined to express the conceptual structure of the object
 * system.  Derived interfaces and classes have similar functions such as
 * SetValues and Print, but the functions are not declared here because the
 * function argument lists vary
 */

#ifndef included_bHYPRE_MatrixVectorView_IOR_h
#include "bHYPRE_MatrixVectorView_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_MatrixVectorView__connectI

#pragma weak bHYPRE_MatrixVectorView__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_MatrixVectorView__object*
bHYPRE_MatrixVectorView__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_MatrixVectorView__object*
bHYPRE_MatrixVectorView__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
