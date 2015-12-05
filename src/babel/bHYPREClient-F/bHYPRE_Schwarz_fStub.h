/*
 * File:          bHYPRE_Schwarz_fStub.h
 * Symbol:        bHYPRE.Schwarz-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side documentation text for bHYPRE.Schwarz
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_Schwarz_fStub_h
#define included_bHYPRE_Schwarz_fStub_h

/**
 * Symbol "bHYPRE.Schwarz" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * Schwarz requires an IJParCSR matrix
 */

#ifndef included_bHYPRE_Schwarz_IOR_h
#include "bHYPRE_Schwarz_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_Schwarz__connectI

#pragma weak bHYPRE_Schwarz__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_Schwarz__object*
bHYPRE_Schwarz__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_Schwarz__object*
bHYPRE_Schwarz__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
