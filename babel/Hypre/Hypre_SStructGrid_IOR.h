/*
 * File:          Hypre_SStructGrid_IOR.h
 * Symbol:        Hypre.SStructGrid-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:13 PST
 * Description:   Intermediate Object Representation for Hypre.SStructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 914
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_SStructGrid_IOR_h
#define included_Hypre_SStructGrid_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_SStructVariable_IOR_h
#include "Hypre_SStructVariable_IOR.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.SStructGrid" (version 0.1.7)
 * 
 * The semi-structured grid class.
 * 
 */

struct Hypre_SStructGrid__array;
struct Hypre_SStructGrid__object;

extern struct Hypre_SStructGrid__object*
Hypre_SStructGrid__new(void);

extern struct Hypre_SStructGrid__object*
Hypre_SStructGrid__remote(const char *url);

extern void Hypre_SStructGrid__init(
  struct Hypre_SStructGrid__object* self);
extern void Hypre_SStructGrid__fini(
  struct Hypre_SStructGrid__object* self);
extern void Hypre_SStructGrid__IOR_version(int32_t *major, int32_t *minor);

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

struct Hypre_SStructGrid__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_SStructGrid__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_SStructGrid__object* self);
  void (*f__ctor)(
    struct Hypre_SStructGrid__object* self);
  void (*f__dtor)(
    struct Hypre_SStructGrid__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct Hypre_SStructGrid__object* self);
  void (*f_deleteRef)(
    struct Hypre_SStructGrid__object* self);
  SIDL_bool (*f_isSame)(
    struct Hypre_SStructGrid__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct Hypre_SStructGrid__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct Hypre_SStructGrid__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct Hypre_SStructGrid__object* self);
  /* Methods introduced in Hypre.SStructGrid-v0.1.7 */
  int32_t (*f_SetNumDimParts)(
    struct Hypre_SStructGrid__object* self,
    int32_t ndim,
    int32_t nparts);
  int32_t (*f_SetExtents)(
    struct Hypre_SStructGrid__object* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper);
  int32_t (*f_SetVariable)(
    struct Hypre_SStructGrid__object* self,
    int32_t part,
    int32_t var,
    enum Hypre_SStructVariable__enum vartype);
  int32_t (*f_AddVariable)(
    struct Hypre_SStructGrid__object* self,
    int32_t part,
    struct SIDL_int__array* index,
    int32_t var,
    enum Hypre_SStructVariable__enum vartype);
  int32_t (*f_SetNeighborBox)(
    struct Hypre_SStructGrid__object* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t nbor_part,
    struct SIDL_int__array* nbor_ilower,
    struct SIDL_int__array* nbor_iupper,
    struct SIDL_int__array* index_map);
  int32_t (*f_AddUnstructuredPart)(
    struct Hypre_SStructGrid__object* self,
    int32_t ilower,
    int32_t iupper);
  int32_t (*f_SetPeriodic)(
    struct Hypre_SStructGrid__object* self,
    int32_t part,
    struct SIDL_int__array* periodic);
  int32_t (*f_SetNumGhost)(
    struct Hypre_SStructGrid__object* self,
    struct SIDL_int__array* num_ghost);
};

/*
 * Define the class object structure.
 */

struct Hypre_SStructGrid__object {
  struct SIDL_BaseClass__object  d_sidl_baseclass;
  struct Hypre_SStructGrid__epv* d_epv;
  void*                          d_data;
};

struct Hypre_SStructGrid__external {
  struct Hypre_SStructGrid__object*
  (*createObject)(void);

  struct Hypre_SStructGrid__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_SStructGrid__external*
Hypre_SStructGrid__externals(void);

#ifdef __cplusplus
}
#endif
#endif
