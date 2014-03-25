/*
 * File:          bHYPRE_Euclid_fStub.h
 * Symbol:        bHYPRE.Euclid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side documentation text for bHYPRE.Euclid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_Euclid_fStub_h
#define included_bHYPRE_Euclid_fStub_h

/**
 * Symbol "bHYPRE.Euclid" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * Although the usual Solver SetParameter functions are available,
 * a Euclid-stype parameter-setting function is also available, SetParameters.
 */

#ifndef included_bHYPRE_Euclid_IOR_h
#include "bHYPRE_Euclid_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_Euclid__connectI

#pragma weak bHYPRE_Euclid__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_Euclid__object*
bHYPRE_Euclid__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_Euclid__object*
bHYPRE_Euclid__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
