/*
 * File:          bHYPRE_SStructGrid_fStub.h
 * Symbol:        bHYPRE.SStructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side documentation text for bHYPRE.SStructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructGrid_fStub_h
#define included_bHYPRE_SStructGrid_fStub_h

/**
 * Symbol "bHYPRE.SStructGrid" (version 1.0.0)
 * 
 * The semi-structured grid class.
 */

#ifndef included_bHYPRE_SStructGrid_IOR_h
#include "bHYPRE_SStructGrid_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_SStructGrid__connectI

#pragma weak bHYPRE_SStructGrid__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructGrid__object*
bHYPRE_SStructGrid__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructGrid__object*
bHYPRE_SStructGrid__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
