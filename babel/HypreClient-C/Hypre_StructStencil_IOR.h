/*
 * File:          Hypre_StructStencil_IOR.h
 * Symbol:        Hypre.StructStencil-v0.1.6
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030121 14:39:12 PST
 * Generated:     20030121 14:39:15 PST
 * Description:   Intermediate Object Representation for Hypre.StructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 398
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_StructStencil_IOR_h
#define included_Hypre_StructStencil_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.StructStencil" (version 0.1.6)
 * 
 * Define a structured stencil for a structured problem description.
 * More than one implementation is not envisioned, thus the decision has
 * been made to make this a class rather than an interface.
 */

struct Hypre_StructStencil__array;
struct Hypre_StructStencil__object;

extern struct Hypre_StructStencil__object*
Hypre_StructStencil__new(void);

extern struct Hypre_StructStencil__object*
Hypre_StructStencil__remote(const char *url);

extern void Hypre_StructStencil__init(
  struct Hypre_StructStencil__object* self);
extern void Hypre_StructStencil__fini(
  struct Hypre_StructStencil__object* self);
extern void Hypre_StructStencil__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_StructStencil__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_StructStencil__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_StructStencil__object* self);
  void (*f__ctor)(
    struct Hypre_StructStencil__object* self);
  void (*f__dtor)(
    struct Hypre_StructStencil__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct Hypre_StructStencil__object* self);
  void (*f_deleteRef)(
    struct Hypre_StructStencil__object* self);
  SIDL_bool (*f_isSame)(
    struct Hypre_StructStencil__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct Hypre_StructStencil__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct Hypre_StructStencil__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct Hypre_StructStencil__object* self);
  /* Methods introduced in Hypre.StructStencil-v0.1.6 */
  int32_t (*f_SetDimension)(
    struct Hypre_StructStencil__object* self,
    int32_t dim);
  int32_t (*f_SetSize)(
    struct Hypre_StructStencil__object* self,
    int32_t size);
  int32_t (*f_SetElement)(
    struct Hypre_StructStencil__object* self,
    int32_t index,
    struct SIDL_int__array* offset);
};

/*
 * Define the class object structure.
 */

struct Hypre_StructStencil__object {
  struct SIDL_BaseClass__object    d_sidl_baseclass;
  struct Hypre_StructStencil__epv* d_epv;
  void*                            d_data;
};

struct Hypre_StructStencil__external {
  struct Hypre_StructStencil__object*
  (*createObject)(void);

  struct Hypre_StructStencil__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_StructStencil__external*
Hypre_StructStencil__externals(void);

#ifdef __cplusplus
}
#endif
#endif
