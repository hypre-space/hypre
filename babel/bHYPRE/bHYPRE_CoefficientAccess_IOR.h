/*
 * File:          bHYPRE_CoefficientAccess_IOR.h
 * Symbol:        bHYPRE.CoefficientAccess-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:35 PST
 * Generated:     20030314 14:22:37 PST
 * Description:   Intermediate Object Representation for bHYPRE.CoefficientAccess
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 754
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_CoefficientAccess_IOR_h
#define included_bHYPRE_CoefficientAccess_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.CoefficientAccess" (version 1.0.0)
 */

struct bHYPRE_CoefficientAccess__array;
struct bHYPRE_CoefficientAccess__object;

extern struct bHYPRE_CoefficientAccess__object*
bHYPRE_CoefficientAccess__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_CoefficientAccess__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    void* self,
    const char* name);
  void (*f__delete)(
    void* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    void* self);
  void (*f_deleteRef)(
    void* self);
  SIDL_bool (*f_isSame)(
    void* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    void* self,
    const char* name);
  SIDL_bool (*f_isType)(
    void* self,
    const char* name);
  /* Methods introduced in bHYPRE.CoefficientAccess-v1.0.0 */
  int32_t (*f_GetRow)(
    void* self,
    int32_t row,
    int32_t* size,
    struct SIDL_int__array** col_ind,
    struct SIDL_double__array** values);
};

/*
 * Define the interface object structure.
 */

struct bHYPRE_CoefficientAccess__object {
  struct bHYPRE_CoefficientAccess__epv* d_epv;
  void*                                 d_object;
};

#ifdef __cplusplus
}
#endif
#endif
