/*
 * File:          bHYPRE_SStructStencil_IOR.h
 * Symbol:        bHYPRE.SStructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:42 PST
 * Generated:     20030314 14:22:44 PST
 * Description:   Intermediate Object Representation for bHYPRE.SStructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 989
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructStencil_IOR_h
#define included_bHYPRE_SStructStencil_IOR_h

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
 * Symbol "bHYPRE.SStructStencil" (version 1.0.0)
 * 
 * The semi-structured grid stencil class.
 * 
 */

struct bHYPRE_SStructStencil__array;
struct bHYPRE_SStructStencil__object;

extern struct bHYPRE_SStructStencil__object*
bHYPRE_SStructStencil__new(void);

extern struct bHYPRE_SStructStencil__object*
bHYPRE_SStructStencil__remote(const char *url);

extern void bHYPRE_SStructStencil__init(
  struct bHYPRE_SStructStencil__object* self);
extern void bHYPRE_SStructStencil__fini(
  struct bHYPRE_SStructStencil__object* self);
extern void bHYPRE_SStructStencil__IOR_version(int32_t *major, int32_t *minor);

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

struct bHYPRE_SStructStencil__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct bHYPRE_SStructStencil__object* self,
    const char* name);
  void (*f__delete)(
    struct bHYPRE_SStructStencil__object* self);
  void (*f__ctor)(
    struct bHYPRE_SStructStencil__object* self);
  void (*f__dtor)(
    struct bHYPRE_SStructStencil__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct bHYPRE_SStructStencil__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_SStructStencil__object* self);
  SIDL_bool (*f_isSame)(
    struct bHYPRE_SStructStencil__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_SStructStencil__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct bHYPRE_SStructStencil__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_SStructStencil__object* self);
  /* Methods introduced in bHYPRE.SStructStencil-v1.0.0 */
  int32_t (*f_SetNumDimSize)(
    struct bHYPRE_SStructStencil__object* self,
    int32_t ndim,
    int32_t size);
  int32_t (*f_SetEntry)(
    struct bHYPRE_SStructStencil__object* self,
    int32_t entry,
    struct SIDL_int__array* offset,
    int32_t var);
};

/*
 * Define the class object structure.
 */

struct bHYPRE_SStructStencil__object {
  struct SIDL_BaseClass__object      d_sidl_baseclass;
  struct bHYPRE_SStructStencil__epv* d_epv;
  void*                              d_data;
};

struct bHYPRE_SStructStencil__external {
  struct bHYPRE_SStructStencil__object*
  (*createObject)(void);

  struct bHYPRE_SStructStencil__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct bHYPRE_SStructStencil__external*
bHYPRE_SStructStencil__externals(void);

#ifdef __cplusplus
}
#endif
#endif
