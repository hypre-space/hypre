/*
 * File:          bHYPRE_Vector_IOR.h
 * Symbol:        bHYPRE.Vector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:41 PST
 * Generated:     20050225 15:45:43 PST
 * Description:   Intermediate Object Representation for bHYPRE.Vector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 667
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_Vector_IOR_h
#define included_bHYPRE_Vector_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
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

struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;

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
  /* Methods introduced in sidl.BaseInterface-v0.9.0 */
  void (*f_addRef)(
    void* self);
  void (*f_deleteRef)(
    void* self);
  sidl_bool (*f_isSame)(
    void* self,
    struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    void* self,
    const char* name);
  sidl_bool (*f_isType)(
    void* self,
    const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    void* self);
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
