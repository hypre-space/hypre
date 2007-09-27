/*
 * File:          bHYPRE_SStructParCSRVector_jniStub.h
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side header code for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructParCSRVector_jniStub_h
#define included_bHYPRE_SStructParCSRVector_jniStub_h

/**
 * Symbol "bHYPRE.SStructParCSRVector" (version 1.0.0)
 * 
 * The SStructParCSR vector class.
 * 
 * Objects of this type can be cast to SStructVectorView or Vector
 * objects using the {\tt \_\_cast} methods.
 */

#ifndef included_bHYPRE_SStructParCSRVector_IOR_h
#include "bHYPRE_SStructParCSRVector_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_SStructParCSRVector__connectI

#pragma weak bHYPRE_SStructParCSRVector__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
