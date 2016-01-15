/*
 * File:          bHYPRE_StructVector_fStub.h
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side documentation text for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_StructVector_fStub_h
#define included_bHYPRE_StructVector_fStub_h

/**
 * Symbol "bHYPRE.StructVector" (version 1.0.0)
 */

#ifndef included_bHYPRE_StructVector_IOR_h
#include "bHYPRE_StructVector_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_StructVector__connectI

#pragma weak bHYPRE_StructVector__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructVector__object*
bHYPRE_StructVector__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_StructVector__object*
bHYPRE_StructVector__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
