/*
 * File:          bHYPRE_StructGrid_IOR.h
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:35 PST
 * Generated:     20030401 14:47:38 PST
 * Description:   Intermediate Object Representation for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 1101
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructGrid_IOR_h
#define included_bHYPRE_StructGrid_IOR_h

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

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;

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
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  void (*f_addRef)(
    struct bHYPRE_StructGrid__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_StructGrid__object* self);
  SIDL_bool (*f_isSame)(
    struct bHYPRE_StructGrid__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_StructGrid__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct bHYPRE_StructGrid__object* self,
    const char* name);
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_StructGrid__object* self);
  /* Methods introduced in SIDL.BaseClass-v0.8.2 */
  /* Methods introduced in bHYPRE.StructGrid-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    struct bHYPRE_StructGrid__object* self,
    void* mpi_comm);
  int32_t (*f_SetDimension)(
    struct bHYPRE_StructGrid__object* self,
    int32_t dim);
  int32_t (*f_SetExtents)(
    struct bHYPRE_StructGrid__object* self,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper);
  int32_t (*f_SetPeriodic)(
    struct bHYPRE_StructGrid__object* self,
    struct SIDL_int__array* periodic);
  int32_t (*f_Assemble)(
    struct bHYPRE_StructGrid__object* self);
};

/*
 * Define the class object structure.
 */

struct bHYPRE_StructGrid__object {
  struct SIDL_BaseClass__object  d_sidl_baseclass;
  struct bHYPRE_StructGrid__epv* d_epv;
  void*                          d_data;
};

struct bHYPRE_StructGrid__external {
  struct bHYPRE_StructGrid__object*
  (*createObject)(void);

  struct bHYPRE_StructGrid__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct bHYPRE_StructGrid__external*
bHYPRE_StructGrid__externals(void);

#ifdef __cplusplus
}
#endif
#endif
