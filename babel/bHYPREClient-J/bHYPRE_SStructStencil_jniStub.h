/*
 * File:          bHYPRE_SStructStencil_jniStub.h
 * Symbol:        bHYPRE.SStructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side header code for bHYPRE.SStructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructStencil_jniStub_h
#define included_bHYPRE_SStructStencil_jniStub_h

/**
 * Symbol "bHYPRE.SStructStencil" (version 1.0.0)
 * 
 * The semi-structured grid stencil class.
 */

#ifndef included_bHYPRE_SStructStencil_IOR_h
#include "bHYPRE_SStructStencil_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_SStructStencil__connectI

#pragma weak bHYPRE_SStructStencil__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructStencil__object*
bHYPRE_SStructStencil__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructStencil__object*
bHYPRE_SStructStencil__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
