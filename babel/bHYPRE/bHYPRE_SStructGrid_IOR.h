/*
 * File:          bHYPRE_SStructGrid_IOR.h
 * Symbol:        bHYPRE.SStructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050208 15:29:05 PST
 * Generated:     20050208 15:29:06 PST
 * Description:   Intermediate Object Representation for bHYPRE.SStructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 904
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructGrid_IOR_h
#define included_bHYPRE_SStructGrid_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_SStructVariable_IOR_h
#include "bHYPRE_SStructVariable_IOR.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.SStructGrid" (version 1.0.0)
 * 
 * The semi-structured grid class.
 * 
 */

struct bHYPRE_SStructGrid__array;
struct bHYPRE_SStructGrid__object;

extern struct bHYPRE_SStructGrid__object*
bHYPRE_SStructGrid__new(void);

extern struct bHYPRE_SStructGrid__object*
bHYPRE_SStructGrid__remote(const char *url);

extern void bHYPRE_SStructGrid__init(
  struct bHYPRE_SStructGrid__object* self);
extern void bHYPRE_SStructGrid__fini(
  struct bHYPRE_SStructGrid__object* self);
extern void bHYPRE_SStructGrid__IOR_version(int32_t *major, int32_t *minor);

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

struct bHYPRE_SStructGrid__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct bHYPRE_SStructGrid__object* self,
    const char* name);
  void (*f__delete)(
    struct bHYPRE_SStructGrid__object* self);
  void (*f__ctor)(
    struct bHYPRE_SStructGrid__object* self);
  void (*f__dtor)(
    struct bHYPRE_SStructGrid__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.0 */
  void (*f_addRef)(
    struct bHYPRE_SStructGrid__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_SStructGrid__object* self);
  sidl_bool (*f_isSame)(
    struct bHYPRE_SStructGrid__object* self,
    struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_SStructGrid__object* self,
    const char* name);
  sidl_bool (*f_isType)(
    struct bHYPRE_SStructGrid__object* self,
    const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_SStructGrid__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.0 */
  /* Methods introduced in bHYPRE.SStructGrid-v1.0.0 */
  int32_t (*f_SetNumDimParts)(
    struct bHYPRE_SStructGrid__object* self,
    int32_t ndim,
    int32_t nparts);
  int32_t (*f_SetExtents)(
    struct bHYPRE_SStructGrid__object* self,
    int32_t part,
    struct sidl_int__array* ilower,
    struct sidl_int__array* iupper);
  int32_t (*f_SetVariable)(
    struct bHYPRE_SStructGrid__object* self,
    int32_t part,
    int32_t var,
    enum bHYPRE_SStructVariable__enum vartype);
  int32_t (*f_AddVariable)(
    struct bHYPRE_SStructGrid__object* self,
    int32_t part,
    struct sidl_int__array* index,
    int32_t var,
    enum bHYPRE_SStructVariable__enum vartype);
  int32_t (*f_SetNeighborBox)(
    struct bHYPRE_SStructGrid__object* self,
    int32_t part,
    struct sidl_int__array* ilower,
    struct sidl_int__array* iupper,
    int32_t nbor_part,
    struct sidl_int__array* nbor_ilower,
    struct sidl_int__array* nbor_iupper,
    struct sidl_int__array* index_map);
  int32_t (*f_AddUnstructuredPart)(
    struct bHYPRE_SStructGrid__object* self,
    int32_t ilower,
    int32_t iupper);
  int32_t (*f_SetPeriodic)(
    struct bHYPRE_SStructGrid__object* self,
    int32_t part,
    struct sidl_int__array* periodic);
  int32_t (*f_SetNumGhost)(
    struct bHYPRE_SStructGrid__object* self,
    struct sidl_int__array* num_ghost);
};

/*
 * Define the class object structure.
 */

struct bHYPRE_SStructGrid__object {
  struct sidl_BaseClass__object   d_sidl_baseclass;
  struct bHYPRE_SStructGrid__epv* d_epv;
  void*                           d_data;
};

struct bHYPRE_SStructGrid__external {
  struct bHYPRE_SStructGrid__object*
  (*createObject)(void);

  struct bHYPRE_SStructGrid__object*
  (*createRemote)(const char *url);

struct sidl_BaseClass__epv*(*getSuperEPV)(void);};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_SStructGrid__external*
bHYPRE_SStructGrid__externals(void);

#ifdef __cplusplus
}
#endif
#endif
