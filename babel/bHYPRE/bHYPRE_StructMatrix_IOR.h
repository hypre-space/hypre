/*
 * File:          bHYPRE_StructMatrix_IOR.h
 * Symbol:        bHYPRE.StructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:24 PST
 * Description:   Intermediate Object Representation for bHYPRE.StructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 1124
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructMatrix_IOR_h
#define included_bHYPRE_StructMatrix_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
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

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.StructMatrix" (version 1.0.0)
 * 
 * A single class that implements both a build interface and an
 * operator interface. It returns itself for GetConstructedObject.
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

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;
struct bHYPRE_StructGrid__array;
struct bHYPRE_StructGrid__object;
struct bHYPRE_StructStencil__array;
struct bHYPRE_StructStencil__object;
struct bHYPRE_Vector__array;
struct bHYPRE_Vector__object;

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
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  void (*f_addRef)(
    struct bHYPRE_StructMatrix__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_StructMatrix__object* self);
  SIDL_bool (*f_isSame)(
    struct bHYPRE_StructMatrix__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_StructMatrix__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct bHYPRE_StructMatrix__object* self,
    const char* name);
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_StructMatrix__object* self);
  /* Methods introduced in SIDL.BaseClass-v0.8.2 */
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
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
    struct SIDL_int__array* value);
  int32_t (*f_SetIntArray2Parameter)(
    struct bHYPRE_StructMatrix__object* self,
    const char* name,
    struct SIDL_int__array* value);
  int32_t (*f_SetDoubleArray1Parameter)(
    struct bHYPRE_StructMatrix__object* self,
    const char* name,
    struct SIDL_double__array* value);
  int32_t (*f_SetDoubleArray2Parameter)(
    struct bHYPRE_StructMatrix__object* self,
    const char* name,
    struct SIDL_double__array* value);
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
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_Initialize)(
    struct bHYPRE_StructMatrix__object* self);
  int32_t (*f_Assemble)(
    struct bHYPRE_StructMatrix__object* self);
  int32_t (*f_GetObject)(
    struct bHYPRE_StructMatrix__object* self,
    struct SIDL_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.StructBuildMatrix-v1.0.0 */
  int32_t (*f_SetGrid)(
    struct bHYPRE_StructMatrix__object* self,
    struct bHYPRE_StructGrid__object* grid);
  int32_t (*f_SetStencil)(
    struct bHYPRE_StructMatrix__object* self,
    struct bHYPRE_StructStencil__object* stencil);
  int32_t (*f_SetValues)(
    struct bHYPRE_StructMatrix__object* self,
    struct SIDL_int__array* index,
    int32_t num_stencil_indices,
    struct SIDL_int__array* stencil_indices,
    struct SIDL_double__array* values);
  int32_t (*f_SetBoxValues)(
    struct bHYPRE_StructMatrix__object* self,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t num_stencil_indices,
    struct SIDL_int__array* stencil_indices,
    struct SIDL_double__array* values);
  int32_t (*f_SetNumGhost)(
    struct bHYPRE_StructMatrix__object* self,
    struct SIDL_int__array* num_ghost);
  int32_t (*f_SetSymmetric)(
    struct bHYPRE_StructMatrix__object* self,
    int32_t symmetric);
  /* Methods introduced in bHYPRE.StructMatrix-v1.0.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE_StructMatrix__object {
  struct SIDL_BaseClass__object           d_sidl_baseclass;
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

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct bHYPRE_StructMatrix__external*
bHYPRE_StructMatrix__externals(void);

#ifdef __cplusplus
}
#endif
#endif
