/*
 * File:          Hypre_StructVector_IOR.h
 * Symbol:        Hypre.StructVector-v0.1.6
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030121 14:39:01 PST
 * Generated:     20030121 14:39:02 PST
 * Description:   Intermediate Object Representation for Hypre.StructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 427
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_StructVector_IOR_h
#define included_Hypre_StructVector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_ProblemDefinition_IOR_h
#include "Hypre_ProblemDefinition_IOR.h"
#endif
#ifndef included_Hypre_StructuredGridBuildVector_IOR_h
#include "Hypre_StructuredGridBuildVector_IOR.h"
#endif
#ifndef included_Hypre_Vector_IOR_h
#include "Hypre_Vector_IOR.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.StructVector" (version 0.1.6)
 */

struct Hypre_StructVector__array;
struct Hypre_StructVector__object;

extern struct Hypre_StructVector__object*
Hypre_StructVector__new(void);

extern struct Hypre_StructVector__object*
Hypre_StructVector__remote(const char *url);

extern void Hypre_StructVector__init(
  struct Hypre_StructVector__object* self);
extern void Hypre_StructVector__fini(
  struct Hypre_StructVector__object* self);
extern void Hypre_StructVector__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct Hypre_StructGrid__array;
struct Hypre_StructGrid__object;
struct Hypre_StructStencil__array;
struct Hypre_StructStencil__object;
struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_StructVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_StructVector__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_StructVector__object* self);
  void (*f__ctor)(
    struct Hypre_StructVector__object* self);
  void (*f__dtor)(
    struct Hypre_StructVector__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct Hypre_StructVector__object* self);
  void (*f_deleteRef)(
    struct Hypre_StructVector__object* self);
  SIDL_bool (*f_isSame)(
    struct Hypre_StructVector__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct Hypre_StructVector__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct Hypre_StructVector__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct Hypre_StructVector__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.ProblemDefinition-v0.1.6 */
  int32_t (*f_SetCommunicator)(
    struct Hypre_StructVector__object* self,
    void* mpi_comm);
  int32_t (*f_Initialize)(
    struct Hypre_StructVector__object* self);
  int32_t (*f_Assemble)(
    struct Hypre_StructVector__object* self);
  int32_t (*f_GetObject)(
    struct Hypre_StructVector__object* self,
    struct SIDL_BaseInterface__object** A);
  /* Methods introduced in Hypre.StructuredGridBuildVector-v0.1.6 */
  int32_t (*f_SetGrid)(
    struct Hypre_StructVector__object* self,
    struct Hypre_StructGrid__object* grid);
  int32_t (*f_SetStencil)(
    struct Hypre_StructVector__object* self,
    struct Hypre_StructStencil__object* stencil);
  int32_t (*f_SetValue)(
    struct Hypre_StructVector__object* self,
    struct SIDL_int__array* grid_index,
    double value);
  int32_t (*f_SetBoxValues)(
    struct Hypre_StructVector__object* self,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    struct SIDL_double__array* values);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.Vector-v0.1.6 */
  int32_t (*f_Clear)(
    struct Hypre_StructVector__object* self);
  int32_t (*f_Copy)(
    struct Hypre_StructVector__object* self,
    struct Hypre_Vector__object* x);
  int32_t (*f_Clone)(
    struct Hypre_StructVector__object* self,
    struct Hypre_Vector__object** x);
  int32_t (*f_Scale)(
    struct Hypre_StructVector__object* self,
    double a);
  int32_t (*f_Dot)(
    struct Hypre_StructVector__object* self,
    struct Hypre_Vector__object* x,
    double* d);
  int32_t (*f_Axpy)(
    struct Hypre_StructVector__object* self,
    double a,
    struct Hypre_Vector__object* x);
  /* Methods introduced in Hypre.StructVector-v0.1.6 */
};

/*
 * Define the class object structure.
 */

struct Hypre_StructVector__object {
  struct SIDL_BaseClass__object                  d_sidl_baseclass;
  struct Hypre_ProblemDefinition__object         d_hypre_problemdefinition;
  struct Hypre_StructuredGridBuildVector__object 
    d_hypre_structuredgridbuildvector;
  struct Hypre_Vector__object                    d_hypre_vector;
  struct Hypre_StructVector__epv*                d_epv;
  void*                                          d_data;
};

struct Hypre_StructVector__external {
  struct Hypre_StructVector__object*
  (*createObject)(void);

  struct Hypre_StructVector__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_StructVector__external*
Hypre_StructVector__externals(void);

#ifdef __cplusplus
}
#endif
#endif
