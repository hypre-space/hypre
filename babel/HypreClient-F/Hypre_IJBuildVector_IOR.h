/*
 * File:          Hypre_IJBuildVector_IOR.h
 * Symbol:        Hypre.IJBuildVector-v0.1.6
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030121 14:39:22 PST
 * Generated:     20030121 14:39:25 PST
 * Description:   Intermediate Object Representation for Hypre.IJBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 249
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_IJBuildVector_IOR_h
#define included_Hypre_IJBuildVector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.IJBuildVector" (version 0.1.6)
 */

struct Hypre_IJBuildVector__array;
struct Hypre_IJBuildVector__object;

extern struct Hypre_IJBuildVector__object*
Hypre_IJBuildVector__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_IJBuildVector__epv {
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
  /* Methods introduced in Hypre.IJBuildVector-v0.1.6 */
  int32_t (*f_SetGlobalSize)(
    void* self,
    int32_t n);
  int32_t (*f_SetPartitioning)(
    void* self,
    struct SIDL_int__array* partitioning);
  int32_t (*f_SetLocalComponents)(
    void* self,
    int32_t num_values,
    struct SIDL_int__array* glob_vec_indices,
    struct SIDL_int__array* value_indices,
    struct SIDL_double__array* values);
  int32_t (*f_AddtoLocalComponents)(
    void* self,
    int32_t num_values,
    struct SIDL_int__array* glob_vec_indices,
    struct SIDL_int__array* value_indices,
    struct SIDL_double__array* values);
  int32_t (*f_SetLocalComponentsInBlock)(
    void* self,
    int32_t glob_vec_index_start,
    int32_t glob_vec_index_stop,
    struct SIDL_int__array* value_indices,
    struct SIDL_double__array* values);
  int32_t (*f_AddToLocalComponentsInBlock)(
    void* self,
    int32_t glob_vec_index_start,
    int32_t glob_vec_index_stop,
    struct SIDL_int__array* value_indices,
    struct SIDL_double__array* values);
  int32_t (*f_Create)(
    void* self,
    void* comm,
    int32_t jlower,
    int32_t jupper);
  int32_t (*f_SetValues)(
    void* self,
    int32_t nvalues,
    struct SIDL_int__array* indices,
    struct SIDL_double__array* values);
  int32_t (*f_AddToValues)(
    void* self,
    int32_t nvalues,
    struct SIDL_int__array* indices,
    struct SIDL_double__array* values);
  int32_t (*f_Read)(
    void* self,
    const char* filename,
    void* comm);
  int32_t (*f_Print)(
    void* self,
    const char* filename);
};

/*
 * Define the interface object structure.
 */

struct Hypre_IJBuildVector__object {
  struct Hypre_IJBuildVector__epv* d_epv;
  void*                            d_object;
};

#ifdef __cplusplus
}
#endif
#endif
