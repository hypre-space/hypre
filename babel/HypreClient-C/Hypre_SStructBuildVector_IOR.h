/*
 * File:          Hypre_SStructBuildVector_IOR.h
 * Symbol:        Hypre.SStructBuildVector-v0.1.7
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:17 PST
 * Generated:     20030306 17:05:20 PST
 * Description:   Intermediate Object Representation for Hypre.SStructBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 432
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_SStructBuildVector_IOR_h
#define included_Hypre_SStructBuildVector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.SStructBuildVector" (version 0.1.7)
 */

struct Hypre_SStructBuildVector__array;
struct Hypre_SStructBuildVector__object;

extern struct Hypre_SStructBuildVector__object*
Hypre_SStructBuildVector__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct Hypre_SStructGrid__array;
struct Hypre_SStructGrid__object;
struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_SStructBuildVector__epv {
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
  /* Methods introduced in Hypre.ProblemDefinition-v0.1.7 */
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
  /* Methods introduced in Hypre.SStructBuildVector-v0.1.7 */
  int32_t (*f_SetGrid)(
    void* self,
    struct Hypre_SStructGrid__object* grid);
  int32_t (*f_SetValues)(
    void* self,
    int32_t part,
    struct SIDL_int__array* index,
    int32_t var,
    struct SIDL_double__array* value);
  int32_t (*f_SetBoxValues)(
    void* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t var,
    struct SIDL_double__array* values);
  int32_t (*f_AddToValues)(
    void* self,
    int32_t part,
    struct SIDL_int__array* index,
    int32_t var,
    struct SIDL_double__array* value);
  int32_t (*f_AddToBoxValues)(
    void* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t var,
    struct SIDL_double__array* values);
  int32_t (*f_Gather)(
    void* self);
  int32_t (*f_GetValues)(
    void* self,
    int32_t part,
    struct SIDL_int__array* index,
    int32_t var,
    double* value);
  int32_t (*f_GetBoxValues)(
    void* self,
    int32_t part,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    int32_t var,
    struct SIDL_double__array** values);
  int32_t (*f_SetComplex)(
    void* self);
  int32_t (*f_Print)(
    void* self,
    const char* filename,
    int32_t all);
};

/*
 * Define the interface object structure.
 */

struct Hypre_SStructBuildVector__object {
  struct Hypre_SStructBuildVector__epv* d_epv;
  void*                                 d_object;
};

#ifdef __cplusplus
}
#endif
#endif
