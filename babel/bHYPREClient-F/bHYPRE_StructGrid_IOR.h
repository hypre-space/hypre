/*
 * File:          bHYPRE_StructGrid_IOR.h
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:50 PST
 * Generated:     20050317 11:17:51 PST
 * Description:   Intermediate Object Representation for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 1106
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructGrid_IOR_h
#define included_bHYPRE_StructGrid_IOR_h

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
 * Symbol "bHYPRE.StructGrid" (version 1.0.0)
 * 
 * Define a structured grid class.
 * 
 */

struct bHYPRE_StructGrid__array;
struct bHYPRE_StructGrid__object;

extern struct bHYPRE_StructGrid__object*
bHYPRE_StructGrid__new(void);

extern struct bHYPRE_StructGrid__object*
bHYPRE_StructGrid__remote(const char *url);

extern void bHYPRE_StructGrid__init(
  struct bHYPRE_StructGrid__object* self);
extern void bHYPRE_StructGrid__fini(
  struct bHYPRE_StructGrid__object* self);
extern void bHYPRE_StructGrid__IOR_version(int32_t *major, int32_t *minor);

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

struct bHYPRE_StructGrid__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct bHYPRE_StructGrid__object* self,
    const char* name);
  void (*f__delete)(
    struct bHYPRE_StructGrid__object* self);
  void (*f__ctor)(
    struct bHYPRE_StructGrid__object* self);
  void (*f__dtor)(
    struct bHYPRE_StructGrid__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.0 */
  void (*f_addRef)(
    struct bHYPRE_StructGrid__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_StructGrid__object* self);
  sidl_bool (*f_isSame)(
    struct bHYPRE_StructGrid__object* self,
    struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_StructGrid__object* self,
    const char* name);
  sidl_bool (*f_isType)(
    struct bHYPRE_StructGrid__object* self,
    const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_StructGrid__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.0 */
  /* Methods introduced in bHYPRE.StructGrid-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    struct bHYPRE_StructGrid__object* self,
    void* mpi_comm);
  int32_t (*f_SetDimension)(
    struct bHYPRE_StructGrid__object* self,
    int32_t dim);
  int32_t (*f_SetExtents)(
    struct bHYPRE_StructGrid__object* self,
    struct sidl_int__array* ilower,
    struct sidl_int__array* iupper);
  int32_t (*f_SetPeriodic)(
    struct bHYPRE_StructGrid__object* self,
    struct sidl_int__array* periodic);
  int32_t (*f_SetNumGhost)(
    struct bHYPRE_StructGrid__object* self,
    struct sidl_int__array* num_ghost);
  int32_t (*f_Assemble)(
    struct bHYPRE_StructGrid__object* self);
};

/*
 * Define the class object structure.
 */

struct bHYPRE_StructGrid__object {
  struct sidl_BaseClass__object  d_sidl_baseclass;
  struct bHYPRE_StructGrid__epv* d_epv;
  void*                          d_data;
};

struct bHYPRE_StructGrid__external {
  struct bHYPRE_StructGrid__object*
  (*createObject)(void);

  struct bHYPRE_StructGrid__object*
  (*createRemote)(const char *url);

struct sidl_BaseClass__epv*(*getSuperEPV)(void);};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_StructGrid__external*
bHYPRE_StructGrid__externals(void);

#ifdef __cplusplus
}
#endif
#endif
