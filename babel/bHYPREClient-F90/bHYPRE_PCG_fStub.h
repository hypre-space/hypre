/*
 * File:          bHYPRE_PCG_fStub.h
 * Symbol:        bHYPRE.PCG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side documentation text for bHYPRE.PCG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_PCG_fStub_h
#define included_bHYPRE_PCG_fStub_h

/**
 * Symbol "bHYPRE.PCG" (version 1.0.0)
 * 
 * PCG solver.
 * This calls Babel-interface matrix and vector functions, so it will work
 * with any consistent matrix, vector, and preconditioner classes.
 */

#ifndef included_bHYPRE_PCG_IOR_h
#include "bHYPRE_PCG_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_PCG__connectI

#pragma weak bHYPRE_PCG__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_PCG__object*
bHYPRE_PCG__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_PCG__object*
bHYPRE_PCG__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
