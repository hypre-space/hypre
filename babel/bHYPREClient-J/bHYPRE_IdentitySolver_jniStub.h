/*
 * File:          bHYPRE_IdentitySolver_jniStub.h
 * Symbol:        bHYPRE.IdentitySolver-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side header code for bHYPRE.IdentitySolver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_IdentitySolver_jniStub_h
#define included_bHYPRE_IdentitySolver_jniStub_h

/**
 * Symbol "bHYPRE.IdentitySolver" (version 1.0.0)
 * 
 * Identity solver, just solves an identity matrix, for when you don't really
 * want a preconditioner
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 */

#ifndef included_bHYPRE_IdentitySolver_IOR_h
#include "bHYPRE_IdentitySolver_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_IdentitySolver__connectI

#pragma weak bHYPRE_IdentitySolver__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_IdentitySolver__object*
bHYPRE_IdentitySolver__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_IdentitySolver__object*
bHYPRE_IdentitySolver__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
