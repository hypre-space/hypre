/*
 * File:          Hypre_SStructStencil_IOR.h
 * Symbol:        Hypre.SStructStencil-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:13 PST
 * Description:   Intermediate Object Representation for Hypre.SStructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1011
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_SStructStencil_IOR_h
#define included_Hypre_SStructStencil_IOR_h

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
 * Symbol "Hypre.SStructStencil" (version 0.1.7)
 * 
 * The semi-structured grid stencil class.
 * 
 */

struct Hypre_SStructStencil__array;
struct Hypre_SStructStencil__object;

extern struct Hypre_SStructStencil__object*
Hypre_SStructStencil__new(void);

extern struct Hypre_SStructStencil__object*
Hypre_SStructStencil__remote(const char *url);

extern void Hypre_SStructStencil__init(
  struct Hypre_SStructStencil__object* self);
extern void Hypre_SStructStencil__fini(
  struct Hypre_SStructStencil__object* self);
extern void Hypre_SStructStencil__IOR_version(int32_t *major, int32_t *minor);

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

struct Hypre_SStructStencil__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_SStructStencil__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_SStructStencil__object* self);
  void (*f__ctor)(
    struct Hypre_SStructStencil__object* self);
  void (*f__dtor)(
    struct Hypre_SStructStencil__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct Hypre_SStructStencil__object* self);
  void (*f_deleteRef)(
    struct Hypre_SStructStencil__object* self);
  SIDL_bool (*f_isSame)(
    struct Hypre_SStructStencil__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct Hypre_SStructStencil__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct Hypre_SStructStencil__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct Hypre_SStructStencil__object* self);
  /* Methods introduced in Hypre.SStructStencil-v0.1.7 */
  int32_t (*f_SetNumDimSize)(
    struct Hypre_SStructStencil__object* self,
    int32_t ndim,
    int32_t size);
  int32_t (*f_SetEntry)(
    struct Hypre_SStructStencil__object* self,
    int32_t entry,
    struct SIDL_int__array* offset,
    int32_t var);
};

/*
 * Define the class object structure.
 */

struct Hypre_SStructStencil__object {
  struct SIDL_BaseClass__object     d_sidl_baseclass;
  struct Hypre_SStructStencil__epv* d_epv;
  void*                             d_data;
};

struct Hypre_SStructStencil__external {
  struct Hypre_SStructStencil__object*
  (*createObject)(void);

  struct Hypre_SStructStencil__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_SStructStencil__external*
Hypre_SStructStencil__externals(void);

#ifdef __cplusplus
}
#endif
#endif
