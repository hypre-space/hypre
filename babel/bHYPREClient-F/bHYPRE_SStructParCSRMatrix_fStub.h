/*
 * File:          bHYPRE_SStructParCSRMatrix_fStub.h
 * Symbol:        bHYPRE.SStructParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side documentation text for bHYPRE.SStructParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructParCSRMatrix_fStub_h
#define included_bHYPRE_SStructParCSRMatrix_fStub_h

/**
 * Symbol "bHYPRE.SStructParCSRMatrix" (version 1.0.0)
 * 
 * The SStructParCSR matrix class.
 * 
 * Objects of this type can be cast to SStructMatrixView or
 * Operator objects using the {\tt \_\_cast} methods.
 */

#ifndef included_bHYPRE_SStructParCSRMatrix_IOR_h
#include "bHYPRE_SStructParCSRMatrix_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_SStructParCSRMatrix__connectI

#pragma weak bHYPRE_SStructParCSRMatrix__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructParCSRMatrix__object*
bHYPRE_SStructParCSRMatrix__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructParCSRMatrix__object*
bHYPRE_SStructParCSRMatrix__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
