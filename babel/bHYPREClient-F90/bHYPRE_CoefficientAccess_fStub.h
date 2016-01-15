/*
 * File:          bHYPRE_CoefficientAccess_fStub.h
 * Symbol:        bHYPRE.CoefficientAccess-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Description:   Client-side documentation text for bHYPRE.CoefficientAccess
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_CoefficientAccess_fStub_h
#define included_bHYPRE_CoefficientAccess_fStub_h

/**
 * Symbol "bHYPRE.CoefficientAccess" (version 1.0.0)
 */

#ifndef included_bHYPRE_CoefficientAccess_IOR_h
#include "bHYPRE_CoefficientAccess_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_CoefficientAccess__connectI

#pragma weak bHYPRE_CoefficientAccess__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_CoefficientAccess__object*
bHYPRE_CoefficientAccess__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_CoefficientAccess__object*
bHYPRE_CoefficientAccess__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
