/*
 * File:          bHYPRE_Vector_IOR.h
 * Symbol:        bHYPRE.Vector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:35 PST
 * Generated:     20030314 14:22:37 PST
 * Description:   Intermediate Object Representation for bHYPRE.Vector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 655
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_Vector_IOR_h
#define included_bHYPRE_Vector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.Vector" (version 1.0.0)
 */

struct bHYPRE_Vector__array;
struct bHYPRE_Vector__object;

extern struct bHYPRE_Vector__object*
bHYPRE_Vector__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_Vector__epv {
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
  /* Methods introduced in bHYPRE.Vector-v1.0.0 */
  int32_t (*f_Clear)(
    void* self);
  int32_t (*f_Copy)(
    void* self,
    struct bHYPRE_Vector__object* x);
  int32_t (*f_Clone)(
    void* self,
    struct bHYPRE_Vector__object** x);
  int32_t (*f_Scale)(
    void* self,
    double a);
  int32_t (*f_Dot)(
    void* self,
    struct bHYPRE_Vector__object* x,
    double* d);
  int32_t (*f_Axpy)(
    void* self,
    double a,
    struct bHYPRE_Vector__object* x);
};

/*
 * Define the interface object structure.
 */

struct bHYPRE_Vector__object {
  struct bHYPRE_Vector__epv* d_epv;
  void*                      d_object;
};

#ifdef __cplusplus
}
#endif
#endif
