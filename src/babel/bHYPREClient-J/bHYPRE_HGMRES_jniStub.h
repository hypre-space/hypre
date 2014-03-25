/*
 * File:          bHYPRE_HGMRES_jniStub.h
 * Symbol:        bHYPRE.HGMRES-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side header code for bHYPRE.HGMRES
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_HGMRES_jniStub_h
#define included_bHYPRE_HGMRES_jniStub_h

/**
 * Symbol "bHYPRE.HGMRES" (version 1.0.0)
 * 
 * Objects of this type can be cast to PreconditionedSolver objects
 * using the {\tt \_\_cast} methods.
 * 
 * The regular GMRES solver calls Babel-interface matrix and vector functions.
 * The HGMRES solver calls HYPRE interface functions.
 * The regular solver will work with any consistent matrix, vector, and
 * preconditioner classes.  The HGMRES solver will work with the more common ones.
 * 
 * The HGMRES solver checks whether the matrix, vectors, and preconditioner
 * are of known types, and will not work with any other types.
 * Presently, the recognized data types are:
 * matrix, vector: IJParCSRMatrix, IJParCSRVector
 * preconditioner: BoomerAMG, ParCSRDiagScale
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
bHYPRE_HGMRES__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
