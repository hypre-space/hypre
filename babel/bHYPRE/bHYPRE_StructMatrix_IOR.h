/*
 * File:          bHYPRE_StructMatrix_IOR.h
 * Symbol:        bHYPRE.StructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:39 PST
 * Generated:     20050317 11:17:40 PST
 * Description:   Intermediate Object Representation for bHYPRE.StructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 1135
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructMatrix_IOR_h
#define included_bHYPRE_StructMatrix_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_Operator_IOR_h
#include "bHYPRE_Operator_IOR.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_IOR_h
#include "bHYPRE_ProblemDefinition_IOR.h"
#endif
#ifndef included_bHYPRE_StructBuildMatrix_IOR_h
#include "bHYPRE_StructBuildMatrix_IOR.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.StructMatrix" (version 1.0.0)
 * 
 * A single class that implements both a build interface and an
 * operator interface. It returns itself for GetConstructedObject.
 * A StructMatrix is a matrix on a structured grid.
 * One function unique to a StructMatrix is SetConstantEntries.
 * This declares that matrix entries corresponding to certain stencil points
 * (supplied as stencil element indices) will be constant throughout the grid.
 * 
 */

struct bHYPRE_StructMatrix__array;
struct bHYPRE_StructMatrix__object;

extern struct bHYPRE_StructMatrix__object*
bHYPRE_StructMatrix__new(void);

extern struct bHYPRE_StructMatrix__object*
bHYPRE_StructMatrix__remote(const char *url);

extern void bHYPRE_StructMatrix__init(
  struct bHYPRE_StructMatrix__object* self);
extern void bHYPRE_StructMatrix__fini(
  struct bHYPRE_StructMatrix__object* self);
extern void bHYPRE_StructMatrix__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_StructGrid__array;
struct bHYPRE_StructGrid__object;
struct bHYPRE_StructStencil__array;
struct bHYPRE_StructStencil__object;
struct bHYPRE_Vector__array;
struct bHYPRE_Vector__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_StructMatrix__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct bHYPRE_StructMatrix__object* self,
    const char* name);
  void (*f__delete)(
    struct bHYPRE_StructMatrix__object* self);
  void (*f__ctor)(
    struct bHYPRE_StructMatrix__object* self);
  void (*f__dtor)(
    struct bHYPRE_StructMatrix__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.0 */
  void (*f_addRef)(
    struct bHYPRE_StructMatrix__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_StructMatrix__object* self);
  sidl_bool (*f_isSame)(
    struct bHYPRE_StructMatrix__object* self,
    struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_StructMatrix__object* self,
    const char* name);
  sidl_bool (*f_isType)(
    struct bHYPRE_StructMatrix__object* self,
    const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_StructMatrix__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.0 */
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    struct bHYPRE_StructMatrix__object* self,
    void* mpi_comm);
  int32_t (*f_SetIntParameter)(
    struct bHYPRE_StructMatrix__object* self,
    const char* name,
    int32_t value);
  int32_t (*f_SetDoubleParameter)(
    struct bHYPRE_StructMatrix__object* self,
    const char* name,
    double value);
  int32_t (*f_SetStringParameter)(
    struct bHYPRE_StructMatrix__object* self,
    const char* name,
    const char* value);
  int32_t (*f_SetIntArray1Parameter)(
    struct bHYPRE_StructMatrix__object* self,
    const char* name,
    struct sidl_int__array* value);
  int32_t (*f_SetIntArray2Parameter)(
    struct bHYPRE_StructMatrix__object* self,
    const char* name,
    struct sidl_int__array* value);
  int32_t (*f_SetDoubleArray1Parameter)(
    struct bHYPRE_StructMatrix__object* self,
    const char* name,
    struct sidl_double__array* value);
  int32_t (*f_SetDoubleArray2Parameter)(
    struct bHYPRE_StructMatrix__object* self,
    const char* name,
    struct sidl_double__array* value);
  int32_t (*f_GetIntValue)(
    struct bHYPRE_StructMatrix__object* self,
    const char* name,
    int32_t* value);
  int32_t (*f_GetDoubleValue)(
    struct bHYPRE_StructMatrix__object* self,
    const char* name,
    double* value);
  int32_t (*f_Setup)(
    struct bHYPRE_StructMatrix__object* self,
    struct bHYPRE_Vector__object* b,
    struct bHYPRE_Vector__object* x);
  int32_t (*f_Apply)(
    struct bHYPRE_StructMatrix__object* self,
    struct bHYPRE_Vector__object* b,
    struct bHYPRE_Vector__object** x);
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_Initialize)(
    struct bHYPRE_StructMatrix__object* self);
  int32_t (*f_Assemble)(
    struct bHYPRE_StructMatrix__object* self);
  int32_t (*f_GetObject)(
    struct bHYPRE_StructMatrix__object* self,
    struct sidl_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.StructBuildMatrix-v1.0.0 */
  int32_t (*f_SetGrid)(
    struct bHYPRE_StructMatrix__object* self,
    struct bHYPRE_StructGrid__object* grid);
  int32_t (*f_SetStencil)(
    struct bHYPRE_StructMatrix__object* self,
    struct bHYPRE_StructStencil__object* stencil);
  int32_t (*f_SetValues)(
    struct bHYPRE_StructMatrix__object* self,
    struct sidl_int__array* index,
    int32_t num_stencil_indices,
    struct sidl_int__array* stencil_indices,
    struct sidl_double__array* values);
  int32_t (*f_SetBoxValues)(
    struct bHYPRE_StructMatrix__object* self,
    struct sidl_int__array* ilower,
    struct sidl_int__array* iupper,
    int32_t num_stencil_indices,
    struct sidl_int__array* stencil_indices,
    struct sidl_double__array* values);
  int32_t (*f_SetNumGhost)(
    struct bHYPRE_StructMatrix__object* self,
    struct sidl_int__array* num_ghost);
  int32_t (*f_SetSymmetric)(
    struct bHYPRE_StructMatrix__object* self,
    int32_t symmetric);
  int32_t (*f_SetConstantEntries)(
    struct bHYPRE_StructMatrix__object* self,
    int32_t num_stencil_constant_points,
    struct sidl_int__array* stencil_constant_points);
  int32_t (*f_SetConstantValues)(
    struct bHYPRE_StructMatrix__object* self,
    int32_t num_stencil_indices,
    struct sidl_int__array* stencil_indices,
    struct sidl_double__array* values);
  /* Methods introduced in bHYPRE.StructMatrix-v1.0.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE_StructMatrix__object {
  struct sidl_BaseClass__object           d_sidl_baseclass;
  struct bHYPRE_Operator__object          d_bhypre_operator;
  struct bHYPRE_ProblemDefinition__object d_bhypre_problemdefinition;
  struct bHYPRE_StructBuildMatrix__object d_bhypre_structbuildmatrix;
  struct bHYPRE_StructMatrix__epv*        d_epv;
  void*                                   d_data;
};

struct bHYPRE_StructMatrix__external {
  struct bHYPRE_StructMatrix__object*
  (*createObject)(void);

  struct bHYPRE_StructMatrix__object*
  (*createRemote)(const char *url);

struct sidl_BaseClass__epv*(*getSuperEPV)(void);};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_StructMatrix__external*
bHYPRE_StructMatrix__externals(void);

#ifdef __cplusplus
}
#endif
#endif
