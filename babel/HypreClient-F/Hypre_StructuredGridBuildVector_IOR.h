/*
 * File:          Hypre_StructuredGridBuildVector_IOR.h
 * Symbol:        Hypre.StructuredGridBuildVector-v0.1.6
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030210 16:05:49 PST
 * Generated:     20030210 16:05:52 PST
 * Description:   Intermediate Object Representation for Hypre.StructuredGridBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 137
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_StructuredGridBuildVector_IOR_h
#define included_Hypre_StructuredGridBuildVector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.StructuredGridBuildVector" (version 0.1.6)
 */

struct Hypre_StructuredGridBuildVector__array;
struct Hypre_StructuredGridBuildVector__object;

extern struct Hypre_StructuredGridBuildVector__object*
Hypre_StructuredGridBuildVector__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct Hypre_StructGrid__array;
struct Hypre_StructGrid__object;
struct Hypre_StructStencil__array;
struct Hypre_StructStencil__object;
struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_StructuredGridBuildVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    void* self,
    const char* name);
  void (*f__delete)(
    void* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    void* self);
  void (*f_deleteRef)(
    void* self);
  SIDL_bool (*f_isSame)(
    void* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    void* self,
    const char* name);
  SIDL_bool (*f_isType)(
    void* self,
    const char* name);
  /* Methods introduced in Hypre.ProblemDefinition-v0.1.6 */
  int32_t (*f_SetCommunicator)(
    void* self,
    void* mpi_comm);
  int32_t (*f_Initialize)(
    void* self);
  int32_t (*f_Assemble)(
    void* self);
  int32_t (*f_GetObject)(
    void* self,
    struct SIDL_BaseInterface__object** A);
  /* Methods introduced in Hypre.StructuredGridBuildVector-v0.1.6 */
  int32_t (*f_SetGrid)(
    void* self,
    struct Hypre_StructGrid__object* grid);
  int32_t (*f_SetStencil)(
    void* self,
    struct Hypre_StructStencil__object* stencil);
  int32_t (*f_SetValue)(
    void* self,
    struct SIDL_int__array* grid_index,
    double value);
  int32_t (*f_SetBoxValues)(
    void* self,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    struct SIDL_double__array* values);
};

/*
 * Define the interface object structure.
 */

struct Hypre_StructuredGridBuildVector__object {
  struct Hypre_StructuredGridBuildVector__epv* d_epv;
  void*                                        d_object;
};

#ifdef __cplusplus
}
#endif
#endif
