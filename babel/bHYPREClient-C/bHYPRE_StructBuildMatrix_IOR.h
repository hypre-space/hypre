/*
 * File:          bHYPRE_StructBuildMatrix_IOR.h
 * Symbol:        bHYPRE.StructBuildMatrix-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:45 PST
 * Generated:     20050317 11:17:47 PST
 * Description:   Intermediate Object Representation for bHYPRE.StructBuildMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 543
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructBuildMatrix_IOR_h
#define included_bHYPRE_StructBuildMatrix_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.StructBuildMatrix" (version 1.0.0)
 */

struct bHYPRE_StructBuildMatrix__array;
struct bHYPRE_StructBuildMatrix__object;

extern struct bHYPRE_StructBuildMatrix__object*
bHYPRE_StructBuildMatrix__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_StructGrid__array;
struct bHYPRE_StructGrid__object;
struct bHYPRE_StructStencil__array;
struct bHYPRE_StructStencil__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_StructBuildMatrix__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    void* self,
    const char* name);
  void (*f__delete)(
    void* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.0 */
  void (*f_addRef)(
    void* self);
  void (*f_deleteRef)(
    void* self);
  sidl_bool (*f_isSame)(
    void* self,
    struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    void* self,
    const char* name);
  sidl_bool (*f_isType)(
    void* self,
    const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    void* self);
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    void* self,
    void* mpi_comm);
  int32_t (*f_Initialize)(
    void* self);
  int32_t (*f_Assemble)(
    void* self);
  int32_t (*f_GetObject)(
    void* self,
    struct sidl_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.StructBuildMatrix-v1.0.0 */
  int32_t (*f_SetGrid)(
    void* self,
    struct bHYPRE_StructGrid__object* grid);
  int32_t (*f_SetStencil)(
    void* self,
    struct bHYPRE_StructStencil__object* stencil);
  int32_t (*f_SetValues)(
    void* self,
    struct sidl_int__array* index,
    int32_t num_stencil_indices,
    struct sidl_int__array* stencil_indices,
    struct sidl_double__array* values);
  int32_t (*f_SetBoxValues)(
    void* self,
    struct sidl_int__array* ilower,
    struct sidl_int__array* iupper,
    int32_t num_stencil_indices,
    struct sidl_int__array* stencil_indices,
    struct sidl_double__array* values);
  int32_t (*f_SetNumGhost)(
    void* self,
    struct sidl_int__array* num_ghost);
  int32_t (*f_SetSymmetric)(
    void* self,
    int32_t symmetric);
  int32_t (*f_SetConstantEntries)(
    void* self,
    int32_t num_stencil_constant_points,
    struct sidl_int__array* stencil_constant_points);
  int32_t (*f_SetConstantValues)(
    void* self,
    int32_t num_stencil_indices,
    struct sidl_int__array* stencil_indices,
    struct sidl_double__array* values);
};

/*
 * Define the interface object structure.
 */

struct bHYPRE_StructBuildMatrix__object {
  struct bHYPRE_StructBuildMatrix__epv* d_epv;
  void*                                 d_object;
};

#ifdef __cplusplus
}
#endif
#endif
