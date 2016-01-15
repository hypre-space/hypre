/*
 * File:          bHYPRE_Pilut_jniStub.h
 * Symbol:        bHYPRE.Pilut-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side header code for bHYPRE.Pilut
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_Pilut_jniStub_h
#define included_bHYPRE_Pilut_jniStub_h

/**
 * Symbol "bHYPRE.Pilut" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * Pilut has not been implemented yet.
 */

#ifndef included_bHYPRE_Pilut_IOR_h
#include "bHYPRE_Pilut_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_Pilut__connectI

#pragma weak bHYPRE_Pilut__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_Pilut__object*
bHYPRE_Pilut__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_Pilut__object*
bHYPRE_Pilut__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
