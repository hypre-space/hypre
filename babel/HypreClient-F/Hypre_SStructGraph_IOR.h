/*
 * File:          Hypre_SStructGraph_IOR.h
 * Symbol:        Hypre.SStructGraph-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:23 PST
 * Generated:     20030306 17:05:25 PST
 * Description:   Intermediate Object Representation for Hypre.SStructGraph
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1032
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_SStructGraph_IOR_h
#define included_Hypre_SStructGraph_IOR_h

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
 * Symbol "Hypre.SStructGraph" (version 0.1.7)
 * 
 * The semi-structured grid graph class.
 * 
 */

struct Hypre_SStructGraph__array;
struct Hypre_SStructGraph__object;

extern struct Hypre_SStructGraph__object*
Hypre_SStructGraph__new(void);

extern struct Hypre_SStructGraph__object*
Hypre_SStructGraph__remote(const char *url);

extern void Hypre_SStructGraph__init(
  struct Hypre_SStructGraph__object* self);
extern void Hypre_SStructGraph__fini(
  struct Hypre_SStructGraph__object* self);
extern void Hypre_SStructGraph__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct Hypre_SStructGrid__array;
struct Hypre_SStructGrid__object;
struct Hypre_SStructStencil__array;
struct Hypre_SStructStencil__object;
struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_SStructGraph__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_SStructGraph__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_SStructGraph__object* self);
  void (*f__ctor)(
    struct Hypre_SStructGraph__object* self);
  void (*f__dtor)(
    struct Hypre_SStructGraph__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct Hypre_SStructGraph__object* self);
  void (*f_deleteRef)(
    struct Hypre_SStructGraph__object* self);
  SIDL_bool (*f_isSame)(
    struct Hypre_SStructGraph__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct Hypre_SStructGraph__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct Hypre_SStructGraph__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct Hypre_SStructGraph__object* self);
  /* Methods introduced in Hypre.SStructGraph-v0.1.7 */
  int32_t (*f_SetGrid)(
    struct Hypre_SStructGraph__object* self,
    struct Hypre_SStructGrid__object* grid);
  int32_t (*f_SetStencil)(
    struct Hypre_SStructGraph__object* self,
    int32_t part,
    int32_t var,
    struct Hypre_SStructStencil__object* stencil);
  int32_t (*f_AddEntries)(
    struct Hypre_SStructGraph__object* self,
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

struct Hypre_SStructGraph__object {
  struct SIDL_BaseClass__object   d_sidl_baseclass;
  struct Hypre_SStructGraph__epv* d_epv;
  void*                           d_data;
};

struct Hypre_SStructGraph__external {
  struct Hypre_SStructGraph__object*
  (*createObject)(void);

  struct Hypre_SStructGraph__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_SStructGraph__external*
Hypre_SStructGraph__externals(void);

#ifdef __cplusplus
}
#endif
#endif
