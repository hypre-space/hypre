/*
 * File:          Hypre_Vector_IOR.h
 * Symbol:        Hypre.Vector-v0.1.6
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030210 16:05:27 PST
 * Generated:     20030210 16:05:29 PST
 * Description:   Intermediate Object Representation for Hypre.Vector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 34
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_Vector_IOR_h
#define included_Hypre_Vector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.Vector" (version 0.1.6)
 */

struct Hypre_Vector__array;
struct Hypre_Vector__object;

extern struct Hypre_Vector__object*
Hypre_Vector__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_Vector__epv {
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
  /* Methods introduced in Hypre.Vector-v0.1.6 */
  int32_t (*f_Clear)(
    void* self);
  int32_t (*f_Copy)(
    void* self,
    struct Hypre_Vector__object* x);
  int32_t (*f_Clone)(
    void* self,
    struct Hypre_Vector__object** x);
  int32_t (*f_Scale)(
    void* self,
    double a);
  int32_t (*f_Dot)(
    void* self,
    struct Hypre_Vector__object* x,
    double* d);
  int32_t (*f_Axpy)(
    void* self,
    double a,
    struct Hypre_Vector__object* x);
};

/*
 * Define the interface object structure.
 */

struct Hypre_Vector__object {
  struct Hypre_Vector__epv* d_epv;
  void*                     d_object;
};

#ifdef __cplusplus
}
#endif
#endif
