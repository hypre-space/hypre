/*
 * File:          Hypre_ParCSRVector_IOR.h
 * Symbol:        Hypre.ParCSRVector-v0.1.6
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030210 16:05:40 PST
 * Generated:     20030210 16:05:44 PST
 * Description:   Intermediate Object Representation for Hypre.ParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 437
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_ParCSRVector_IOR_h
#define included_Hypre_ParCSRVector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_CoefficientAccess_IOR_h
#include "Hypre_CoefficientAccess_IOR.h"
#endif
#ifndef included_Hypre_IJBuildVector_IOR_h
#include "Hypre_IJBuildVector_IOR.h"
#endif
#ifndef included_Hypre_ProblemDefinition_IOR_h
#include "Hypre_ProblemDefinition_IOR.h"
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
 * Symbol "Hypre.ParCSRVector" (version 0.1.6)
 */

struct Hypre_ParCSRVector__array;
struct Hypre_ParCSRVector__object;

extern struct Hypre_ParCSRVector__object*
Hypre_ParCSRVector__new(void);

extern struct Hypre_ParCSRVector__object*
Hypre_ParCSRVector__remote(const char *url);

extern void Hypre_ParCSRVector__init(
  struct Hypre_ParCSRVector__object* self);
extern void Hypre_ParCSRVector__fini(
  struct Hypre_ParCSRVector__object* self);
extern void Hypre_ParCSRVector__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_ParCSRVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_ParCSRVector__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_ParCSRVector__object* self);
  void (*f__ctor)(
    struct Hypre_ParCSRVector__object* self);
  void (*f__dtor)(
    struct Hypre_ParCSRVector__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct Hypre_ParCSRVector__object* self);
  void (*f_deleteRef)(
    struct Hypre_ParCSRVector__object* self);
  SIDL_bool (*f_isSame)(
    struct Hypre_ParCSRVector__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct Hypre_ParCSRVector__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct Hypre_ParCSRVector__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct Hypre_ParCSRVector__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.CoefficientAccess-v0.1.6 */
  int32_t (*f_GetRow)(
    struct Hypre_ParCSRVector__object* self,
    int32_t row,
    int32_t* size,
    struct SIDL_int__array** col_ind,
    struct SIDL_double__array** values);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.ProblemDefinition-v0.1.6 */
  int32_t (*f_SetCommunicator)(
    struct Hypre_ParCSRVector__object* self,
    void* mpi_comm);
  int32_t (*f_Initialize)(
    struct Hypre_ParCSRVector__object* self);
  int32_t (*f_Assemble)(
    struct Hypre_ParCSRVector__object* self);
  int32_t (*f_GetObject)(
    struct Hypre_ParCSRVector__object* self,
    struct SIDL_BaseInterface__object** A);
  /* Methods introduced in Hypre.IJBuildVector-v0.1.6 */
  int32_t (*f_SetGlobalSize)(
    struct Hypre_ParCSRVector__object* self,
    int32_t n);
  int32_t (*f_SetPartitioning)(
    struct Hypre_ParCSRVector__object* self,
    struct SIDL_int__array* partitioning);
  int32_t (*f_SetLocalComponents)(
    struct Hypre_ParCSRVector__object* self,
    int32_t num_values,
    struct SIDL_int__array* glob_vec_indices,
    struct SIDL_int__array* value_indices,
    struct SIDL_double__array* values);
  int32_t (*f_AddtoLocalComponents)(
    struct Hypre_ParCSRVector__object* self,
    int32_t num_values,
    struct SIDL_int__array* glob_vec_indices,
    struct SIDL_int__array* value_indices,
    struct SIDL_double__array* values);
  int32_t (*f_SetLocalComponentsInBlock)(
    struct Hypre_ParCSRVector__object* self,
    int32_t glob_vec_index_start,
    int32_t glob_vec_index_stop,
    struct SIDL_int__array* value_indices,
    struct SIDL_double__array* values);
  int32_t (*f_AddToLocalComponentsInBlock)(
    struct Hypre_ParCSRVector__object* self,
    int32_t glob_vec_index_start,
    int32_t glob_vec_index_stop,
    struct SIDL_int__array* value_indices,
    struct SIDL_double__array* values);
  int32_t (*f_Create)(
    struct Hypre_ParCSRVector__object* self,
    void* comm,
    int32_t jlower,
    int32_t jupper);
  int32_t (*f_SetValues)(
    struct Hypre_ParCSRVector__object* self,
    int32_t nvalues,
    struct SIDL_int__array* indices,
    struct SIDL_double__array* values);
  int32_t (*f_AddToValues)(
    struct Hypre_ParCSRVector__object* self,
    int32_t nvalues,
    struct SIDL_int__array* indices,
    struct SIDL_double__array* values);
  int32_t (*f_Read)(
    struct Hypre_ParCSRVector__object* self,
    const char* filename,
    void* comm);
  int32_t (*f_Print)(
    struct Hypre_ParCSRVector__object* self,
    const char* filename);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.Vector-v0.1.6 */
  int32_t (*f_Clear)(
    struct Hypre_ParCSRVector__object* self);
  int32_t (*f_Copy)(
    struct Hypre_ParCSRVector__object* self,
    struct Hypre_Vector__object* x);
  int32_t (*f_Clone)(
    struct Hypre_ParCSRVector__object* self,
    struct Hypre_Vector__object** x);
  int32_t (*f_Scale)(
    struct Hypre_ParCSRVector__object* self,
    double a);
  int32_t (*f_Dot)(
    struct Hypre_ParCSRVector__object* self,
    struct Hypre_Vector__object* x,
    double* d);
  int32_t (*f_Axpy)(
    struct Hypre_ParCSRVector__object* self,
    double a,
    struct Hypre_Vector__object* x);
  /* Methods introduced in Hypre.ParCSRVector-v0.1.6 */
};

/*
 * Define the class object structure.
 */

struct Hypre_ParCSRVector__object {
  struct SIDL_BaseClass__object          d_sidl_baseclass;
  struct Hypre_CoefficientAccess__object d_hypre_coefficientaccess;
  struct Hypre_IJBuildVector__object     d_hypre_ijbuildvector;
  struct Hypre_ProblemDefinition__object d_hypre_problemdefinition;
  struct Hypre_Vector__object            d_hypre_vector;
  struct Hypre_ParCSRVector__epv*        d_epv;
  void*                                  d_data;
};

struct Hypre_ParCSRVector__external {
  struct Hypre_ParCSRVector__object*
  (*createObject)(void);

  struct Hypre_ParCSRVector__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_ParCSRVector__external*
Hypre_ParCSRVector__externals(void);

#ifdef __cplusplus
}
#endif
#endif
