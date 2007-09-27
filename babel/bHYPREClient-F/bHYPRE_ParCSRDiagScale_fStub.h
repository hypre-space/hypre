/*
 * File:          bHYPRE_ParCSRDiagScale_fStub.h
 * Symbol:        bHYPRE.ParCSRDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side documentation text for bHYPRE.ParCSRDiagScale
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_ParCSRDiagScale_fStub_h
#define included_bHYPRE_ParCSRDiagScale_fStub_h

/**
 * Symbol "bHYPRE.ParCSRDiagScale" (version 1.0.0)
 * 
 * Diagonal scaling preconditioner for ParCSR matrix class.
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 */

#ifndef included_bHYPRE_ParCSRDiagScale_IOR_h
#include "bHYPRE_ParCSRDiagScale_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_ParCSRDiagScale__connectI

#pragma weak bHYPRE_ParCSRDiagScale__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_ParCSRDiagScale__object*
bHYPRE_ParCSRDiagScale__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_ParCSRDiagScale__object*
bHYPRE_ParCSRDiagScale__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
