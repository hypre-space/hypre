/*
 * File:          Hypre_StructGrid_IOR.h
 * Symbol:        Hypre.StructGrid-v0.1.6
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030210 16:05:28 PST
 * Generated:     20030210 16:05:30 PST
 * Description:   Intermediate Object Representation for Hypre.StructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 408
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_StructGrid_IOR_h
#define included_Hypre_StructGrid_IOR_h

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
 * Symbol "Hypre.StructGrid" (version 0.1.6)
 * 
 * Define a structured grid class.
 */

struct Hypre_StructGrid__array;
struct Hypre_StructGrid__object;

extern struct Hypre_StructGrid__object*
Hypre_StructGrid__new(void);

extern struct Hypre_StructGrid__object*
Hypre_StructGrid__remote(const char *url);

extern void Hypre_StructGrid__init(
  struct Hypre_StructGrid__object* self);
extern void Hypre_StructGrid__fini(
  struct Hypre_StructGrid__object* self);
extern void Hypre_StructGrid__IOR_version(int32_t *major, int32_t *minor);

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

struct Hypre_StructGrid__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_StructGrid__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_StructGrid__object* self);
  void (*f__ctor)(
    struct Hypre_StructGrid__object* self);
  void (*f__dtor)(
    struct Hypre_StructGrid__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct Hypre_StructGrid__object* self);
  void (*f_deleteRef)(
    struct Hypre_StructGrid__object* self);
  SIDL_bool (*f_isSame)(
    struct Hypre_StructGrid__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct Hypre_StructGrid__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct Hypre_StructGrid__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct Hypre_StructGrid__object* self);
  /* Methods introduced in Hypre.StructGrid-v0.1.6 */
  int32_t (*f_SetCommunicator)(
    struct Hypre_StructGrid__object* self,
    void* MPI_comm);
  int32_t (*f_SetDimension)(
    struct Hypre_StructGrid__object* self,
    int32_t dim);
  int32_t (*f_SetExtents)(
    struct Hypre_StructGrid__object* self,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper);
  int32_t (*f_SetPeriodic)(
    struct Hypre_StructGrid__object* self,
    struct SIDL_int__array* periodic);
  int32_t (*f_Assemble)(
    struct Hypre_StructGrid__object* self);
};

/*
 * Define the class object structure.
 */

struct Hypre_StructGrid__object {
  struct SIDL_BaseClass__object d_sidl_baseclass;
  struct Hypre_StructGrid__epv* d_epv;
  void*                         d_data;
};

struct Hypre_StructGrid__external {
  struct Hypre_StructGrid__object*
  (*createObject)(void);

  struct Hypre_StructGrid__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_StructGrid__external*
Hypre_StructGrid__externals(void);

#ifdef __cplusplus
}
#endif
#endif
