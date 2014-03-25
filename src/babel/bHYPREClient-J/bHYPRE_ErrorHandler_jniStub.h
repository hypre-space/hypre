/*
 * File:          bHYPRE_ErrorHandler_jniStub.h
 * Symbol:        bHYPRE.ErrorHandler-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side header code for bHYPRE.ErrorHandler
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_ErrorHandler_jniStub_h
#define included_bHYPRE_ErrorHandler_jniStub_h

/**
 * Symbol "bHYPRE.ErrorHandler" (version 1.0.0)
 * 
 * ErrorHandler class is an interface to the hypre error handling system.
 * Its methods help interpret the error flag ierr returned by hypre functions.
 */

#ifndef included_bHYPRE_ErrorHandler_IOR_h
#include "bHYPRE_ErrorHandler_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_ErrorHandler__connectI

#pragma weak bHYPRE_ErrorHandler__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_ErrorHandler__object*
bHYPRE_ErrorHandler__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_ErrorHandler__object*
bHYPRE_ErrorHandler__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
