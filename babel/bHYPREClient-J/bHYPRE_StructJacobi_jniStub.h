/*
 * File:          bHYPRE_StructJacobi_jniStub.h
 * Symbol:        bHYPRE.StructJacobi-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side header code for bHYPRE.StructJacobi
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_StructJacobi_jniStub_h
#define included_bHYPRE_StructJacobi_jniStub_h

/**
 * Symbol "bHYPRE.StructJacobi" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects
 * using the {\tt \_\_cast} methods.
 * 
 * The StructJacobi solver requires a Struct matrix.
 */

#ifndef included_bHYPRE_StructJacobi_IOR_h
#include "bHYPRE_StructJacobi_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_StructJacobi__connectI

#pragma weak bHYPRE_StructJacobi__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructJacobi__object*
bHYPRE_StructJacobi__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_StructJacobi__object*
bHYPRE_StructJacobi__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
