/*
 * File:          Hypre_IJParCSRMatrix_IOR.h
 * Symbol:        Hypre.IJParCSRMatrix-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:13 PST
 * Description:   Intermediate Object Representation for Hypre.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 799
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_IJParCSRMatrix_IOR_h
#define included_Hypre_IJParCSRMatrix_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_CoefficientAccess_IOR_h
#include "Hypre_CoefficientAccess_IOR.h"
#endif
#ifndef included_Hypre_IJBuildMatrix_IOR_h
#include "Hypre_IJBuildMatrix_IOR.h"
#endif
#ifndef included_Hypre_Operator_IOR_h
#include "Hypre_Operator_IOR.h"
#endif
#ifndef included_Hypre_ProblemDefinition_IOR_h
#include "Hypre_ProblemDefinition_IOR.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.IJParCSRMatrix" (version 0.1.7)
 * 
 * The IJParCSR matrix class.
 * 
 * Objects of this type can be cast to IJBuildMatrix, Operator, or
 * CoefficientAccess objects using the {\tt \_\_cast} methods.
 * 
 */

struct Hypre_IJParCSRMatrix__array;
struct Hypre_IJParCSRMatrix__object;

extern struct Hypre_IJParCSRMatrix__object*
Hypre_IJParCSRMatrix__new(void);

extern struct Hypre_IJParCSRMatrix__object*
Hypre_IJParCSRMatrix__remote(const char *url);

extern void Hypre_IJParCSRMatrix__init(
  struct Hypre_IJParCSRMatrix__object* self);
extern void Hypre_IJParCSRMatrix__fini(
  struct Hypre_IJParCSRMatrix__object* self);
extern void Hypre_IJParCSRMatrix__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct Hypre_Vector__array;
struct Hypre_Vector__object;
struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_IJParCSRMatrix__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_IJParCSRMatrix__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_IJParCSRMatrix__object* self);
  void (*f__ctor)(
    struct Hypre_IJParCSRMatrix__object* self);
  void (*f__dtor)(
    struct Hypre_IJParCSRMatrix__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct Hypre_IJParCSRMatrix__object* self);
  void (*f_deleteRef)(
    struct Hypre_IJParCSRMatrix__object* self);
  SIDL_bool (*f_isSame)(
    struct Hypre_IJParCSRMatrix__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct Hypre_IJParCSRMatrix__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct Hypre_IJParCSRMatrix__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct Hypre_IJParCSRMatrix__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.CoefficientAccess-v0.1.7 */
  int32_t (*f_GetRow)(
    struct Hypre_IJParCSRMatrix__object* self,
    int32_t row,
    int32_t* size,
    struct SIDL_int__array** col_ind,
    struct SIDL_double__array** values);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.ProblemDefinition-v0.1.7 */
  int32_t (*f_SetCommunicator)(
    struct Hypre_IJParCSRMatrix__object* self,
    void* mpi_comm);
  int32_t (*f_Initialize)(
    struct Hypre_IJParCSRMatrix__object* self);
  int32_t (*f_Assemble)(
    struct Hypre_IJParCSRMatrix__object* self);
  int32_t (*f_GetObject)(
    struct Hypre_IJParCSRMatrix__object* self,
    struct SIDL_BaseInterface__object** A);
  /* Methods introduced in Hypre.IJBuildMatrix-v0.1.7 */
  int32_t (*f_SetLocalRange)(
    struct Hypre_IJParCSRMatrix__object* self,
    int32_t ilower,
    int32_t iupper,
    int32_t jlower,
    int32_t jupper);
  int32_t (*f_SetValues)(
    struct Hypre_IJParCSRMatrix__object* self,
    int32_t nrows,
    struct SIDL_int__array* ncols,
    struct SIDL_int__array* rows,
    struct SIDL_int__array* cols,
    struct SIDL_double__array* values);
  int32_t (*f_AddToValues)(
    struct Hypre_IJParCSRMatrix__object* self,
    int32_t nrows,
    struct SIDL_int__array* ncols,
    struct SIDL_int__array* rows,
    struct SIDL_int__array* cols,
    struct SIDL_double__array* values);
  int32_t (*f_GetLocalRange)(
    struct Hypre_IJParCSRMatrix__object* self,
    int32_t* ilower,
    int32_t* iupper,
    int32_t* jlower,
    int32_t* jupper);
  int32_t (*f_GetRowCounts)(
    struct Hypre_IJParCSRMatrix__object* self,
    int32_t nrows,
    struct SIDL_int__array* rows,
    struct SIDL_int__array** ncols);
  int32_t (*f_GetValues)(
    struct Hypre_IJParCSRMatrix__object* self,
    int32_t nrows,
    struct SIDL_int__array* ncols,
    struct SIDL_int__array* rows,
    struct SIDL_int__array* cols,
    struct SIDL_double__array** values);
  int32_t (*f_SetRowSizes)(
    struct Hypre_IJParCSRMatrix__object* self,
    struct SIDL_int__array* sizes);
  int32_t (*f_Print)(
    struct Hypre_IJParCSRMatrix__object* self,
    const char* filename);
  int32_t (*f_Read)(
    struct Hypre_IJParCSRMatrix__object* self,
    const char* filename,
    void* comm);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.Operator-v0.1.7 */
  int32_t (*f_SetIntParameter)(
    struct Hypre_IJParCSRMatrix__object* self,
    const char* name,
    int32_t value);
  int32_t (*f_SetDoubleParameter)(
    struct Hypre_IJParCSRMatrix__object* self,
    const char* name,
    double value);
  int32_t (*f_SetStringParameter)(
    struct Hypre_IJParCSRMatrix__object* self,
    const char* name,
    const char* value);
  int32_t (*f_SetIntArrayParameter)(
    struct Hypre_IJParCSRMatrix__object* self,
    const char* name,
    struct SIDL_int__array* value);
  int32_t (*f_SetDoubleArrayParameter)(
    struct Hypre_IJParCSRMatrix__object* self,
    const char* name,
    struct SIDL_double__array* value);
  int32_t (*f_GetIntValue)(
    struct Hypre_IJParCSRMatrix__object* self,
    const char* name,
    int32_t* value);
  int32_t (*f_GetDoubleValue)(
    struct Hypre_IJParCSRMatrix__object* self,
    const char* name,
    double* value);
  int32_t (*f_Setup)(
    struct Hypre_IJParCSRMatrix__object* self,
    struct Hypre_Vector__object* b,
    struct Hypre_Vector__object* x);
  int32_t (*f_Apply)(
    struct Hypre_IJParCSRMatrix__object* self,
    struct Hypre_Vector__object* b,
    struct Hypre_Vector__object** x);
  /* Methods introduced in Hypre.IJParCSRMatrix-v0.1.7 */
  int32_t (*f_SetDiagOffdSizes)(
    struct Hypre_IJParCSRMatrix__object* self,
    struct SIDL_int__array* diag_sizes,
    struct SIDL_int__array* offdiag_sizes);
};

/*
 * Define the class object structure.
 */

struct Hypre_IJParCSRMatrix__object {
  struct SIDL_BaseClass__object          d_sidl_baseclass;
  struct Hypre_CoefficientAccess__object d_hypre_coefficientaccess;
  struct Hypre_IJBuildMatrix__object     d_hypre_ijbuildmatrix;
  struct Hypre_Operator__object          d_hypre_operator;
  struct Hypre_ProblemDefinition__object d_hypre_problemdefinition;
  struct Hypre_IJParCSRMatrix__epv*      d_epv;
  void*                                  d_data;
};

struct Hypre_IJParCSRMatrix__external {
  struct Hypre_IJParCSRMatrix__object*
  (*createObject)(void);

  struct Hypre_IJParCSRMatrix__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_IJParCSRMatrix__external*
Hypre_IJParCSRMatrix__externals(void);

#ifdef __cplusplus
}
#endif
#endif
