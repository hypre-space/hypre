/*
 * File:          bHYPRE_StructStencil_IOR.h
 * Symbol:        bHYPRE.StructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:46 PST
 * Generated:     20050225 15:45:48 PST
 * Description:   Intermediate Object Representation for bHYPRE.StructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 1088
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructStencil_IOR_h
#define included_bHYPRE_StructStencil_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.StructStencil" (version 1.0.0)
 * 
 * Define a structured stencil for a structured problem
 * description.  More than one implementation is not envisioned,
 * thus the decision has been made to make this a class rather than
 * an interface.
 * 
 */

struct bHYPRE_StructStencil__array;
struct bHYPRE_StructStencil__object;

extern struct bHYPRE_StructStencil__object*
bHYPRE_StructStencil__new(void);

extern struct bHYPRE_StructStencil__object*
bHYPRE_StructStencil__remote(const char *url);

extern void bHYPRE_StructStencil__init(
  struct bHYPRE_StructStencil__object* self);
extern void bHYPRE_StructStencil__fini(
  struct bHYPRE_StructStencil__object* self);
extern void bHYPRE_StructStencil__IOR_version(int32_t *major, int32_t *minor);

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

struct bHYPRE_StructStencil__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct bHYPRE_StructStencil__object* self,
    const char* name);
  void (*f__delete)(
    struct bHYPRE_StructStencil__object* self);
  void (*f__ctor)(
    struct bHYPRE_StructStencil__object* self);
  void (*f__dtor)(
    struct bHYPRE_StructStencil__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.0 */
  void (*f_addRef)(
    struct bHYPRE_StructStencil__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_StructStencil__object* self);
  sidl_bool (*f_isSame)(
    struct bHYPRE_StructStencil__object* self,
    struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_StructStencil__object* self,
    const char* name);
  sidl_bool (*f_isType)(
    struct bHYPRE_StructStencil__object* self,
    const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_StructStencil__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.0 */
  /* Methods introduced in bHYPRE.StructStencil-v1.0.0 */
  int32_t (*f_SetDimension)(
    struct bHYPRE_StructStencil__object* self,
    int32_t dim);
  int32_t (*f_SetSize)(
    struct bHYPRE_StructStencil__object* self,
    int32_t size);
  int32_t (*f_SetElement)(
    struct bHYPRE_StructStencil__object* self,
    int32_t index,
    struct sidl_int__array* offset);
};

/*
 * Define the class object structure.
 */

struct bHYPRE_StructStencil__object {
  struct sidl_BaseClass__object     d_sidl_baseclass;
  struct bHYPRE_StructStencil__epv* d_epv;
  void*                             d_data;
};

struct bHYPRE_StructStencil__external {
  struct bHYPRE_StructStencil__object*
  (*createObject)(void);

  struct bHYPRE_StructStencil__object*
  (*createRemote)(const char *url);

struct sidl_BaseClass__epv*(*getSuperEPV)(void);};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_StructStencil__external*
bHYPRE_StructStencil__externals(void);

#ifdef __cplusplus
}
#endif
#endif
