/*
 * File:          bHYPRE_CGNR_fStub.h
 * Symbol:        bHYPRE.CGNR-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side documentation text for bHYPRE.CGNR
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_CGNR_fStub_h
#define included_bHYPRE_CGNR_fStub_h

/**
 * Symbol "bHYPRE.CGNR" (version 1.0.0)
 * 
 * Objects of this type can be cast to PreconditionedSolver objects
 * using the {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 * CGNR solver calls Babel-interface functions
 */

#ifndef included_bHYPRE_CGNR_IOR_h
#include "bHYPRE_CGNR_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_CGNR__connectI

#pragma weak bHYPRE_CGNR__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_CGNR__object*
bHYPRE_CGNR__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_CGNR__object*
bHYPRE_CGNR__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
