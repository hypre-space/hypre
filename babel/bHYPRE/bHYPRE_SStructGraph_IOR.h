/*
 * File:          bHYPRE_SStructGraph_IOR.h
 * Symbol:        bHYPRE.SStructGraph-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050208 15:29:05 PST
 * Generated:     20050208 15:29:05 PST
 * Description:   Intermediate Object Representation for bHYPRE.SStructGraph
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 1022
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructGraph_IOR_h
#define included_bHYPRE_SStructGraph_IOR_h

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
 * Symbol "bHYPRE.SStructGraph" (version 1.0.0)
 * 
 * The semi-structured grid graph class.
 * 
 */

struct bHYPRE_SStructGraph__array;
struct bHYPRE_SStructGraph__object;

extern struct bHYPRE_SStructGraph__object*
bHYPRE_SStructGraph__new(void);

extern struct bHYPRE_SStructGraph__object*
bHYPRE_SStructGraph__remote(const char *url);

extern void bHYPRE_SStructGraph__init(
  struct bHYPRE_SStructGraph__object* self);
extern void bHYPRE_SStructGraph__fini(
  struct bHYPRE_SStructGraph__object* self);
extern void bHYPRE_SStructGraph__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_SStructGrid__array;
struct bHYPRE_SStructGrid__object;
struct bHYPRE_SStructStencil__array;
struct bHYPRE_SStructStencil__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_SStructGraph__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct bHYPRE_SStructGraph__object* self,
    const char* name);
  void (*f__delete)(
    struct bHYPRE_SStructGraph__object* self);
  void (*f__ctor)(
    struct bHYPRE_SStructGraph__object* self);
  void (*f__dtor)(
    struct bHYPRE_SStructGraph__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.0 */
  void (*f_addRef)(
    struct bHYPRE_SStructGraph__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_SStructGraph__object* self);
  sidl_bool (*f_isSame)(
    struct bHYPRE_SStructGraph__object* self,
    struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_SStructGraph__object* self,
    const char* name);
  sidl_bool (*f_isType)(
    struct bHYPRE_SStructGraph__object* self,
    const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_SStructGraph__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.0 */
  /* Methods introduced in bHYPRE.SStructGraph-v1.0.0 */
  int32_t (*f_SetGrid)(
    struct bHYPRE_SStructGraph__object* self,
    struct bHYPRE_SStructGrid__object* grid);
  int32_t (*f_SetStencil)(
    struct bHYPRE_SStructGraph__object* self,
    int32_t part,
    int32_t var,
    struct bHYPRE_SStructStencil__object* stencil);
  int32_t (*f_AddEntries)(
    struct bHYPRE_SStructGraph__object* self,
    int32_t part,
    struct sidl_int__array* index,
    int32_t var,
    int32_t to_part,
    struct sidl_int__array* to_index,
    int32_t to_var);
};

/*
 * Define the class object structure.
 */

struct bHYPRE_SStructGraph__object {
  struct sidl_BaseClass__object    d_sidl_baseclass;
  struct bHYPRE_SStructGraph__epv* d_epv;
  void*                            d_data;
};

struct bHYPRE_SStructGraph__external {
  struct bHYPRE_SStructGraph__object*
  (*createObject)(void);

  struct bHYPRE_SStructGraph__object*
  (*createRemote)(const char *url);

struct sidl_BaseClass__epv*(*getSuperEPV)(void);};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_SStructGraph__external*
bHYPRE_SStructGraph__externals(void);

#ifdef __cplusplus
}
#endif
#endif
