/*
 * File:          bHYPRE_SStructGrid_IOR.h
 * Symbol:        bHYPRE.SStructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:42 PST
 * Generated:     20030314 14:22:43 PST
 * Description:   Intermediate Object Representation for bHYPRE.SStructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 892
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructGrid_IOR_h
#define included_bHYPRE_SStructGrid_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif
#ifndef included_bHYPRE_SStructVariable_IOR_h
#include "bHYPRE_SStructVariable_IOR.h"
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

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;

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
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct bHYPRE_SStructGrid__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_SStructGrid__object* self);
  SIDL_bool (*f_isSame)(
    struct bHYPRE_SStructGrid__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_SStructGrid__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct bHYPRE_SStructGrid__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_SStructGrid__object* self);
  /* Methods introduced in bHYPRE.SStructGrid-v1.0.0 */
  int32_t (*f_SetNumDimParts)(
    struct bHYPRE_SStructGrid__object* self,
    int32_t ndim,
    int32_t nparts);
  int32_t (*f_SetExtents)(
    struct bHYPRE_SStructGrid__object* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper);
  int32_t (*f_SetVariable)(
    struct bHYPRE_SStructGrid__object* self,
    int32_t part,
    int32_t var,
    enum bHYPRE_SStructVariable__enum vartype);
  int32_t (*f_AddVariable)(
    struct bHYPRE_SStructGrid__object* self,
    int32_t part,
    struct SIDL_int__array* index,
    int32_t var,
    enum bHYPRE_SStructVariable__enum vartype);
  int32_t (*f_SetNeighborBox)(
    struct bHYPRE_SStructGrid__object* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t nbor_part,
    struct SIDL_int__array* nbor_ilower,
    struct SIDL_int__array* nbor_iupper,
    struct SIDL_int__array* index_map);
  int32_t (*f_AddUnstructuredPart)(
    struct bHYPRE_SStructGrid__object* self,
    int32_t ilower,
    int32_t iupper);
  int32_t (*f_SetPeriodic)(
    struct bHYPRE_SStructGrid__object* self,
    int32_t part,
    struct SIDL_int__array* periodic);
  int32_t (*f_SetNumGhost)(
    struct bHYPRE_SStructGrid__object* self,
    struct SIDL_int__array* num_ghost);
};

/*
 * Define the class object structure.
 */

struct bHYPRE_SStructGrid__object {
  struct SIDL_BaseClass__object   d_sidl_baseclass;
  struct bHYPRE_SStructGrid__epv* d_epv;
  void*                           d_data;
};

struct bHYPRE_SStructGrid__external {
  struct bHYPRE_SStructGrid__object*
  (*createObject)(void);

  struct bHYPRE_SStructGrid__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct bHYPRE_SStructGrid__external*
bHYPRE_SStructGrid__externals(void);

#ifdef __cplusplus
}
#endif
#endif
