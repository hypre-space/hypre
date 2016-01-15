/*
 * File:          bHYPRE_IJParCSRVector_jniStub.h
 * Symbol:        bHYPRE.IJParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side header code for bHYPRE.IJParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_IJParCSRVector_jniStub_h
#define included_bHYPRE_IJParCSRVector_jniStub_h

/**
 * Symbol "bHYPRE.IJParCSRVector" (version 1.0.0)
 * 
 * The IJParCSR vector class.
 * 
 * Objects of this type can be cast to IJVectorView or Vector
 * objects using the {\tt \_\_cast} methods.
 */

#ifndef included_bHYPRE_IJParCSRVector_IOR_h
#include "bHYPRE_IJParCSRVector_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_IJParCSRVector__connectI

#pragma weak bHYPRE_IJParCSRVector__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_IJParCSRVector__object*
bHYPRE_IJParCSRVector__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_IJParCSRVector__object*
bHYPRE_IJParCSRVector__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
