/*
 * File:          bHYPRE_StructBuildVector_IOR.h
 * Symbol:        bHYPRE.StructBuildVector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:45 PST
 * Generated:     20030320 16:52:48 PST
 * Description:   Intermediate Object Representation for bHYPRE.StructBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 568
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructBuildVector_IOR_h
#define included_bHYPRE_StructBuildVector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.StructBuildVector" (version 1.0.0)
 */

struct bHYPRE_StructBuildVector__array;
struct bHYPRE_StructBuildVector__object;

extern struct bHYPRE_StructBuildVector__object*
bHYPRE_StructBuildVector__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct bHYPRE_StructGrid__array;
struct bHYPRE_StructGrid__object;
struct bHYPRE_StructStencil__array;
struct bHYPRE_StructStencil__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_StructBuildVector__epv {
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
    struct SIDL_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.StructBuildVector-v1.0.0 */
  int32_t (*f_SetGrid)(
    void* self,
    struct bHYPRE_StructGrid__object* grid);
  int32_t (*f_SetStencil)(
    void* self,
    struct bHYPRE_StructStencil__object* stencil);
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

struct bHYPRE_StructBuildVector__object {
  struct bHYPRE_StructBuildVector__epv* d_epv;
  void*                                 d_object;
};

#ifdef __cplusplus
}
#endif
#endif
