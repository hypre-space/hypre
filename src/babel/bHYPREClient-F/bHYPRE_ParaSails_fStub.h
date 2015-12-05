/*
 * File:          bHYPRE_ParaSails_fStub.h
 * Symbol:        bHYPRE.ParaSails-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side documentation text for bHYPRE.ParaSails
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_ParaSails_fStub_h
#define included_bHYPRE_ParaSails_fStub_h

/**
 * Symbol "bHYPRE.ParaSails" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 * ParaSails requires an IJParCSR matrix
 */

#ifndef included_bHYPRE_ParaSails_IOR_h
#include "bHYPRE_ParaSails_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_ParaSails__connectI

#pragma weak bHYPRE_ParaSails__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_ParaSails__object*
bHYPRE_ParaSails__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_ParaSails__object*
bHYPRE_ParaSails__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
