/*
 * File:          bHYPRE_IJMatrixView_fStub.h
 * Symbol:        bHYPRE.IJMatrixView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.4
 * Description:   Client-side documentation text for bHYPRE.IJMatrixView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_IJMatrixView_fStub_h
#define included_bHYPRE_IJMatrixView_fStub_h

/**
 * Symbol "bHYPRE.IJMatrixView" (version 1.0.0)
 * 
 * This interface represents a linear-algebraic conceptual view of a
 * linear system.  The 'I' and 'J' in the name are meant to be
 * mnemonic for the traditional matrix notation A(I,J).
 */

#ifndef included_bHYPRE_IJMatrixView_IOR_h
#include "bHYPRE_IJMatrixView_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_IJMatrixView__connectI

#pragma weak bHYPRE_IJMatrixView__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_IJMatrixView__object*
bHYPRE_IJMatrixView__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_IJMatrixView__object*
bHYPRE_IJMatrixView__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
