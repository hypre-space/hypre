/*
 * File:          Hypre_CoefficientAccess_IOR.h
 * Symbol:        Hypre.CoefficientAccess-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.7.4
 * SIDL Created:  20021217 16:38:33 PST
 * Generated:     20021217 16:38:34 PST
 * Description:   Intermediate Object Representation for Hypre.CoefficientAccess
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.7.4
 * source-line   = 381
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_CoefficientAccess_IOR_h
#define included_Hypre_CoefficientAccess_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.CoefficientAccess" (version 0.1.5)
 * 
 * The GetRow method will allocate space for its two output arrays on
 * the first call.  The space will be reused on subsequent calls.
 * Thus the user must not delete them, yet must not depend on the
 * data from GetRow to persist beyond the next GetRow call.
 */

struct Hypre_CoefficientAccess__array;
struct Hypre_CoefficientAccess__object;

extern struct Hypre_CoefficientAccess__object*
Hypre_CoefficientAccess__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_CoefficientAccess__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    void* self,
    const char* name);
  void (*f__delete)(
    void* self);
  /* Methods introduced in SIDL.BaseInterface-v0.7.4 */
  void (*f_addReference)(
    void* self);
  void (*f_deleteReference)(
    void* self);
  SIDL_bool (*f_isSame)(
    void* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInterface)(
    void* self,
    const char* name);
  SIDL_bool (*f_isInstanceOf)(
    void* self,
    const char* name);
  /* Methods introduced in Hypre.CoefficientAccess-v0.1.5 */
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

struct Hypre_CoefficientAccess__object {
  struct Hypre_CoefficientAccess__epv* d_epv;
  void*                                d_object;
};

#ifdef __cplusplus
}
#endif
#endif
