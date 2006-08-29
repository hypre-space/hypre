/*
 * File:          bHYPRE_HGMRES_fStub.h
 * Symbol:        bHYPRE.HGMRES-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side documentation text for bHYPRE.HGMRES
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_HGMRES_fStub_h
#define included_bHYPRE_HGMRES_fStub_h

/**
 * Symbol "bHYPRE.HGMRES" (version 1.0.0)
 */

#ifndef included_bHYPRE_HGMRES_IOR_h
#include "bHYPRE_HGMRES_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_HGMRES__connectI

#pragma weak bHYPRE_HGMRES__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_HGMRES__object*
bHYPRE_HGMRES__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_HGMRES__object*
bHYPRE_HGMRES__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
