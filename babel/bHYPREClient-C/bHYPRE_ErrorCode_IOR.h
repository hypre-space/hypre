/*
 * File:          bHYPRE_ErrorCode_IOR.h
 * Symbol:        bHYPRE.ErrorCode-v1.0.0
 * Symbol Type:   enumeration
 * Babel Version: 1.0.4
 * Description:   Intermediate Object Representation for bHYPRE.ErrorCode
 * 
 * WARNING: Automatically generated; changes will be lost
 */

#ifndef included_bHYPRE_ErrorCode_IOR_h
#define included_bHYPRE_ErrorCode_IOR_h

#ifndef included_sidlType_h
#include "sidlType.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.ErrorCode" (version 1.0.0)
 * 
 * The ErrorCode enumerated type is used with methods of the ErrorHandler class.
 */


/* Opaque forward declaration of array struct */
struct bHYPRE_ErrorCode__array;

enum bHYPRE_ErrorCode__enum {
  bHYPRE_ErrorCode_HYPRE_ERROR_GENERIC = 1,

  bHYPRE_ErrorCode_HYPRE_ERROR_MEMORY  = 2,

  bHYPRE_ErrorCode_HYPRE_ERROR_ARG     = 4,

  bHYPRE_ErrorCode_HYPRE_ERROR_CONV    = 256

};

#ifdef __cplusplus
}
#endif
#endif
