/*
 * File:          Hypre_StructToIJMatrix_IOR.h
 * Symbol:        Hypre.StructToIJMatrix-v0.1.6
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030210 16:05:50 PST
 * Generated:     20030210 16:05:52 PST
 * Description:   Intermediate Object Representation for Hypre.StructToIJMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 446
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_StructToIJMatrix_IOR_h
#define included_Hypre_StructToIJMatrix_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_ProblemDefinition_IOR_h
#include "Hypre_ProblemDefinition_IOR.h"
#endif
#ifndef included_Hypre_StructuredGridBuildMatrix_IOR_h
#include "Hypre_StructuredGridBuildMatrix_IOR.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.StructToIJMatrix" (version 0.1.6)
 * 
 * This class implements the StructuredGrid user interface, but builds
 * an unstructured matrix behind the curtain.  It does this by using
 * an IJBuildMatrix (e.g., ParCSRMatrix, PETScMatrix, ...)
 * specified by the user with an extra method ...
 */

struct Hypre_StructToIJMatrix__array;
struct Hypre_StructToIJMatrix__object;

extern struct Hypre_StructToIJMatrix__object*
Hypre_StructToIJMatrix__new(void);

extern struct Hypre_StructToIJMatrix__object*
Hypre_StructToIJMatrix__remote(const char *url);

extern void Hypre_StructToIJMatrix__init(
  struct Hypre_StructToIJMatrix__object* self);
extern void Hypre_StructToIJMatrix__fini(
  struct Hypre_StructToIJMatrix__object* self);
extern void Hypre_StructToIJMatrix__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct Hypre_IJBuildMatrix__array;
struct Hypre_IJBuildMatrix__object;
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

struct Hypre_StructToIJMatrix__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_StructToIJMatrix__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_StructToIJMatrix__object* self);
  void (*f__ctor)(
    struct Hypre_StructToIJMatrix__object* self);
  void (*f__dtor)(
    struct Hypre_StructToIJMatrix__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct Hypre_StructToIJMatrix__object* self);
  void (*f_deleteRef)(
    struct Hypre_StructToIJMatrix__object* self);
  SIDL_bool (*f_isSame)(
    struct Hypre_StructToIJMatrix__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct Hypre_StructToIJMatrix__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct Hypre_StructToIJMatrix__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct Hypre_StructToIJMatrix__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.ProblemDefinition-v0.1.6 */
  int32_t (*f_SetCommunicator)(
    struct Hypre_StructToIJMatrix__object* self,
    void* mpi_comm);
  int32_t (*f_Initialize)(
    struct Hypre_StructToIJMatrix__object* self);
  int32_t (*f_Assemble)(
    struct Hypre_StructToIJMatrix__object* self);
  int32_t (*f_GetObject)(
    struct Hypre_StructToIJMatrix__object* self,
    struct SIDL_BaseInterface__object** A);
  /* Methods introduced in Hypre.StructuredGridBuildMatrix-v0.1.6 */
  int32_t (*f_SetGrid)(
    struct Hypre_StructToIJMatrix__object* self,
    struct Hypre_StructGrid__object* grid);
  int32_t (*f_SetStencil)(
    struct Hypre_StructToIJMatrix__object* self,
    struct Hypre_StructStencil__object* stencil);
  int32_t (*f_SetValues)(
    struct Hypre_StructToIJMatrix__object* self,
    struct SIDL_int__array* index,
    int32_t num_stencil_indices,
    struct SIDL_int__array* stencil_indices,
    struct SIDL_double__array* values);
  int32_t (*f_SetBoxValues)(
    struct Hypre_StructToIJMatrix__object* self,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t num_stencil_indices,
    struct SIDL_int__array* stencil_indices,
    struct SIDL_double__array* values);
  int32_t (*f_SetNumGhost)(
    struct Hypre_StructToIJMatrix__object* self,
    struct SIDL_int__array* num_ghost);
  int32_t (*f_SetSymmetric)(
    struct Hypre_StructToIJMatrix__object* self,
    int32_t symmetric);
  /* Methods introduced in Hypre.StructToIJMatrix-v0.1.6 */
  int32_t (*f_SetIJMatrix)(
    struct Hypre_StructToIJMatrix__object* self,
    struct Hypre_IJBuildMatrix__object* I);
};

/*
 * Define the class object structure.
 */

struct Hypre_StructToIJMatrix__object {
  struct SIDL_BaseClass__object                  d_sidl_baseclass;
  struct Hypre_ProblemDefinition__object         d_hypre_problemdefinition;
  struct Hypre_StructuredGridBuildMatrix__object 
    d_hypre_structuredgridbuildmatrix;
  struct Hypre_StructToIJMatrix__epv*            d_epv;
  void*                                          d_data;
};

struct Hypre_StructToIJMatrix__external {
  struct Hypre_StructToIJMatrix__object*
  (*createObject)(void);

  struct Hypre_StructToIJMatrix__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_StructToIJMatrix__external*
Hypre_StructToIJMatrix__externals(void);

#ifdef __cplusplus
}
#endif
#endif
