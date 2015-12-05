/*
 * File:          bHYPRE_SStructGraph_IOR.h
 * Symbol:        bHYPRE.SStructGraph-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:23 PST
 * Description:   Intermediate Object Representation for bHYPRE.SStructGraph
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 1022
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructGraph_IOR_h
#define included_bHYPRE_SStructGraph_IOR_h

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

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;
struct bHYPRE_SStructGrid__array;
struct bHYPRE_SStructGrid__object;
struct bHYPRE_SStructStencil__array;
struct bHYPRE_SStructStencil__object;

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
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  void (*f_addRef)(
    struct bHYPRE_SStructGraph__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_SStructGraph__object* self);
  SIDL_bool (*f_isSame)(
    struct bHYPRE_SStructGraph__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_SStructGraph__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct bHYPRE_SStructGraph__object* self,
    const char* name);
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_SStructGraph__object* self);
  /* Methods introduced in SIDL.BaseClass-v0.8.2 */
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
    struct SIDL_int__array* index,
    int32_t var,
    int32_t to_part,
    struct SIDL_int__array* to_index,
    int32_t to_var);
};

/*
 * Define the class object structure.
 */

struct bHYPRE_SStructGraph__object {
  struct SIDL_BaseClass__object    d_sidl_baseclass;
  struct bHYPRE_SStructGraph__epv* d_epv;
  void*                            d_data;
};

struct bHYPRE_SStructGraph__external {
  struct bHYPRE_SStructGraph__object*
  (*createObject)(void);

  struct bHYPRE_SStructGraph__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct bHYPRE_SStructGraph__external*
bHYPRE_SStructGraph__externals(void);

#ifdef __cplusplus
}
#endif
#endif
