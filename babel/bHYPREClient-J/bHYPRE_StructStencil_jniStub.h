/*
 * File:          bHYPRE_StructStencil_jniStub.h
 * Symbol:        bHYPRE.StructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side header code for bHYPRE.StructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_StructStencil_jniStub_h
#define included_bHYPRE_StructStencil_jniStub_h

/**
 * Symbol "bHYPRE.StructStencil" (version 1.0.0)
 * 
 * Define a structured stencil for a structured problem
 * description.  More than one implementation is not envisioned,
 * thus the decision has been made to make this a class rather than
 * an interface.
 */

#ifndef included_bHYPRE_StructStencil_IOR_h
#include "bHYPRE_StructStencil_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_StructStencil__connectI

#pragma weak bHYPRE_StructStencil__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructStencil__object*
bHYPRE_StructStencil__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_StructStencil__object*
bHYPRE_StructStencil__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
