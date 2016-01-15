/*
 * File:          bHYPRE_Solver_fStub.h
 * Symbol:        bHYPRE.Solver-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.4
 * Description:   Client-side documentation text for bHYPRE.Solver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_Solver_fStub_h
#define included_bHYPRE_Solver_fStub_h

/**
 * Symbol "bHYPRE.Solver" (version 1.0.0)
 */

#ifndef included_bHYPRE_Solver_IOR_h
#include "bHYPRE_Solver_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_Solver__connectI

#pragma weak bHYPRE_Solver__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_Solver__object*
bHYPRE_Solver__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_Solver__object*
bHYPRE_Solver__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
