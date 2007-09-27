/*
 * File:          bHYPRE_SStructMatrix_fStub.h
 * Symbol:        bHYPRE.SStructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side documentation text for bHYPRE.SStructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructMatrix_fStub_h
#define included_bHYPRE_SStructMatrix_fStub_h

/**
 * Symbol "bHYPRE.SStructMatrix" (version 1.0.0)
 * 
 * The semi-structured grid matrix class.
 * 
 * Objects of this type can be cast to SStructMatrixView or
 * Operator objects using the {\tt \_\_cast} methods.
 */

#ifndef included_bHYPRE_SStructMatrix_IOR_h
#include "bHYPRE_SStructMatrix_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_SStructMatrix__connectI

#pragma weak bHYPRE_SStructMatrix__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructMatrix__object*
bHYPRE_SStructMatrix__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructMatrix__object*
bHYPRE_SStructMatrix__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
