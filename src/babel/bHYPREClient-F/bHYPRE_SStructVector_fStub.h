/*
 * File:          bHYPRE_SStructVector_fStub.h
 * Symbol:        bHYPRE.SStructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side documentation text for bHYPRE.SStructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructVector_fStub_h
#define included_bHYPRE_SStructVector_fStub_h

/**
 * Symbol "bHYPRE.SStructVector" (version 1.0.0)
 * 
 * The semi-structured grid vector class.
 * 
 * Objects of this type can be cast to SStructVectorView or Vector
 * objects using the {\tt \_\_cast} methods.
 */

#ifndef included_bHYPRE_SStructVector_IOR_h
#include "bHYPRE_SStructVector_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_SStructVector__connectI

#pragma weak bHYPRE_SStructVector__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructVector__object*
bHYPRE_SStructVector__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructVector__object*
bHYPRE_SStructVector__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
