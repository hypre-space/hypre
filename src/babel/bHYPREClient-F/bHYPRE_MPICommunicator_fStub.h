/*
 * File:          bHYPRE_MPICommunicator_fStub.h
 * Symbol:        bHYPRE.MPICommunicator-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side documentation text for bHYPRE.MPICommunicator
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_MPICommunicator_fStub_h
#define included_bHYPRE_MPICommunicator_fStub_h

/**
 * Symbol "bHYPRE.MPICommunicator" (version 1.0.0)
 * 
 * MPICommunicator class
 * - two general Create functions: use CreateC if called from C code,
 * CreateF if called from Fortran code.
 * - Create_MPICommWorld will create a MPICommunicator to represent
 * MPI_Comm_World, and can be called from any language.
 */

#ifndef included_bHYPRE_MPICommunicator_IOR_h
#include "bHYPRE_MPICommunicator_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_MPICommunicator__connectI

#pragma weak bHYPRE_MPICommunicator__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_MPICommunicator__object*
bHYPRE_MPICommunicator__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_MPICommunicator__object*
bHYPRE_MPICommunicator__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
