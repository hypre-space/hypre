/*
 * File:          bHYPRE_StructSMG_fStub.h
 * Symbol:        bHYPRE.StructSMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side documentation text for bHYPRE.StructSMG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_StructSMG_fStub_h
#define included_bHYPRE_StructSMG_fStub_h

/**
 * Symbol "bHYPRE.StructSMG" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects
 * using the {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 * The StructSMG solver requires a Struct matrix.
 */

#ifndef included_bHYPRE_StructSMG_IOR_h
#include "bHYPRE_StructSMG_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_StructSMG__connectI

#pragma weak bHYPRE_StructSMG__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructSMG__object*
bHYPRE_StructSMG__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_StructSMG__object*
bHYPRE_StructSMG__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
