/*
 * File:          bHYPRE_StructMatrix_fStub.h
 * Symbol:        bHYPRE.StructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side documentation text for bHYPRE.StructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_StructMatrix_fStub_h
#define included_bHYPRE_StructMatrix_fStub_h

/**
 * Symbol "bHYPRE.StructMatrix" (version 1.0.0)
 * 
 * A single class that implements both a view interface and an
 * operator interface.
 * A StructMatrix is a matrix on a structured grid.
 * One function unique to a StructMatrix is SetConstantEntries.
 * This declares that matrix entries corresponding to certain stencil points
 * (supplied as stencil element indices) will be constant throughout the grid.
 */

#ifndef included_bHYPRE_StructMatrix_IOR_h
#include "bHYPRE_StructMatrix_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_StructMatrix__connectI

#pragma weak bHYPRE_StructMatrix__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructMatrix__object*
bHYPRE_StructMatrix__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_StructMatrix__object*
bHYPRE_StructMatrix__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
