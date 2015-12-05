/*
 * File:          bHYPRE_StructVector_IOR.h
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:46 PST
 * Generated:     20030401 14:47:54 PST
 * Description:   Intermediate Object Representation for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 1129
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructVector_IOR_h
#define included_bHYPRE_StructVector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_IOR_h
#include "bHYPRE_ProblemDefinition_IOR.h"
#endif
#ifndef included_bHYPRE_StructBuildVector_IOR_h
#include "bHYPRE_StructBuildVector_IOR.h"
#endif
#ifndef included_bHYPRE_Vector_IOR_h
#include "bHYPRE_Vector_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.StructVector" (version 1.0.0)
 */

struct bHYPRE_StructVector__array;
struct bHYPRE_StructVector__object;

extern struct bHYPRE_StructVector__object*
bHYPRE_StructVector__new(void);

extern struct bHYPRE_StructVector__object*
bHYPRE_StructVector__remote(const char *url);

extern void bHYPRE_StructVector__init(
  struct bHYPRE_StructVector__object* self);
extern void bHYPRE_StructVector__fini(
  struct bHYPRE_StructVector__object* self);
extern void bHYPRE_StructVector__IOR_version(int32_t *major, int32_t *minor);

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

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_StructVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct bHYPRE_StructVector__object* self,
    const char* name);
  void (*f__delete)(
    struct bHYPRE_StructVector__object* self);
  void (*f__ctor)(
    struct bHYPRE_StructVector__object* self);
  void (*f__dtor)(
    struct bHYPRE_StructVector__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  void (*f_addRef)(
    struct bHYPRE_StructVector__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_StructVector__object* self);
  SIDL_bool (*f_isSame)(
    struct bHYPRE_StructVector__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_StructVector__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct bHYPRE_StructVector__object* self,
    const char* name);
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_StructVector__object* self);
  /* Methods introduced in SIDL.BaseClass-v0.8.2 */
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    struct bHYPRE_StructVector__object* self,
    void* mpi_comm);
  int32_t (*f_Initialize)(
    struct bHYPRE_StructVector__object* self);
  int32_t (*f_Assemble)(
    struct bHYPRE_StructVector__object* self);
  int32_t (*f_GetObject)(
    struct bHYPRE_StructVector__object* self,
    struct SIDL_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.StructBuildVector-v1.0.0 */
  int32_t (*f_SetGrid)(
    struct bHYPRE_StructVector__object* self,
    struct bHYPRE_StructGrid__object* grid);
  int32_t (*f_SetStencil)(
    struct bHYPRE_StructVector__object* self,
    struct bHYPRE_StructStencil__object* stencil);
  int32_t (*f_SetValue)(
    struct bHYPRE_StructVector__object* self,
    struct SIDL_int__array* grid_index,
    double value);
  int32_t (*f_SetBoxValues)(
    struct bHYPRE_StructVector__object* self,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    struct SIDL_double__array* values);
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  /* Methods introduced in bHYPRE.Vector-v1.0.0 */
  int32_t (*f_Clear)(
    struct bHYPRE_StructVector__object* self);
  int32_t (*f_Copy)(
    struct bHYPRE_StructVector__object* self,
    struct bHYPRE_Vector__object* x);
  int32_t (*f_Clone)(
    struct bHYPRE_StructVector__object* self,
    struct bHYPRE_Vector__object** x);
  int32_t (*f_Scale)(
    struct bHYPRE_StructVector__object* self,
    double a);
  int32_t (*f_Dot)(
    struct bHYPRE_StructVector__object* self,
    struct bHYPRE_Vector__object* x,
    double* d);
  int32_t (*f_Axpy)(
    struct bHYPRE_StructVector__object* self,
    double a,
    struct bHYPRE_Vector__object* x);
  /* Methods introduced in bHYPRE.StructVector-v1.0.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE_StructVector__object {
  struct SIDL_BaseClass__object           d_sidl_baseclass;
  struct bHYPRE_ProblemDefinition__object d_bhypre_problemdefinition;
  struct bHYPRE_StructBuildVector__object d_bhypre_structbuildvector;
  struct bHYPRE_Vector__object            d_bhypre_vector;
  struct bHYPRE_StructVector__epv*        d_epv;
  void*                                   d_data;
};

struct bHYPRE_StructVector__external {
  struct bHYPRE_StructVector__object*
  (*createObject)(void);

  struct bHYPRE_StructVector__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct bHYPRE_StructVector__external*
bHYPRE_StructVector__externals(void);

#ifdef __cplusplus
}
#endif
#endif
