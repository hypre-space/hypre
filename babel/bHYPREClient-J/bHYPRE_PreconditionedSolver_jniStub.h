/*
 * File:          bHYPRE_PreconditionedSolver_jniStub.h
 * Symbol:        bHYPRE.PreconditionedSolver-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.4
 * Description:   Client-side header code for bHYPRE.PreconditionedSolver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_PreconditionedSolver_jniStub_h
#define included_bHYPRE_PreconditionedSolver_jniStub_h

/**
 * Symbol "bHYPRE.PreconditionedSolver" (version 1.0.0)
 */

#ifndef included_bHYPRE_PreconditionedSolver_IOR_h
#include "bHYPRE_PreconditionedSolver_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_PreconditionedSolver__connectI

#pragma weak bHYPRE_PreconditionedSolver__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_PreconditionedSolver__object*
bHYPRE_PreconditionedSolver__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_PreconditionedSolver__object*
bHYPRE_PreconditionedSolver__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
