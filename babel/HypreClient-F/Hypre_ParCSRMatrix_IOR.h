/*
 * File:          Hypre_ParCSRMatrix_IOR.h
 * Symbol:        Hypre.ParCSRMatrix-v0.1.6
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030121 14:39:22 PST
 * Generated:     20030121 14:39:23 PST
 * Description:   Intermediate Object Representation for Hypre.ParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 433
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_ParCSRMatrix_IOR_h
#define included_Hypre_ParCSRMatrix_IOR_h

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
 * Symbol "Hypre.ParCSRMatrix" (version 0.1.6)
 * 
 * A single class that implements both a build interface and an operator
 * interface. It returns itself for <code>GetConstructedObject</code>.
 */

struct Hypre_ParCSRMatrix__array;
struct Hypre_ParCSRMatrix__object;

extern struct Hypre_ParCSRMatrix__object*
Hypre_ParCSRMatrix__new(void);

extern struct Hypre_ParCSRMatrix__object*
Hypre_ParCSRMatrix__remote(const char *url);

extern void Hypre_ParCSRMatrix__init(
  struct Hypre_ParCSRMatrix__object* self);
extern void Hypre_ParCSRMatrix__fini(
  struct Hypre_ParCSRMatrix__object* self);
extern void Hypre_ParCSRMatrix__IOR_version(int32_t *major, int32_t *minor);

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

struct Hypre_ParCSRMatrix__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_ParCSRMatrix__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_ParCSRMatrix__object* self);
  void (*f__ctor)(
    struct Hypre_ParCSRMatrix__object* self);
  void (*f__dtor)(
    struct Hypre_ParCSRMatrix__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct Hypre_ParCSRMatrix__object* self);
  void (*f_deleteRef)(
    struct Hypre_ParCSRMatrix__object* self);
  SIDL_bool (*f_isSame)(
    struct Hypre_ParCSRMatrix__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct Hypre_ParCSRMatrix__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct Hypre_ParCSRMatrix__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct Hypre_ParCSRMatrix__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.CoefficientAccess-v0.1.6 */
  int32_t (*f_GetRow)(
    struct Hypre_ParCSRMatrix__object* self,
    int32_t row,
    int32_t* size,
    struct SIDL_int__array** col_ind,
    struct SIDL_double__array** values);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.ProblemDefinition-v0.1.6 */
  int32_t (*f_SetCommunicator)(
    struct Hypre_ParCSRMatrix__object* self,
    void* mpi_comm);
  int32_t (*f_Initialize)(
    struct Hypre_ParCSRMatrix__object* self);
  int32_t (*f_Assemble)(
    struct Hypre_ParCSRMatrix__object* self);
  int32_t (*f_GetObject)(
    struct Hypre_ParCSRMatrix__object* self,
    struct SIDL_BaseInterface__object** A);
  /* Methods introduced in Hypre.IJBuildMatrix-v0.1.6 */
  int32_t (*f_Create)(
    struct Hypre_ParCSRMatrix__object* self,
    int32_t ilower,
    int32_t iupper,
    int32_t jlower,
    int32_t jupper);
  int32_t (*f_SetValues)(
    struct Hypre_ParCSRMatrix__object* self,
    int32_t nrows,
    struct SIDL_int__array* ncols,
    struct SIDL_int__array* rows,
    struct SIDL_int__array* cols,
    struct SIDL_double__array* values);
  int32_t (*f_AddToValues)(
    struct Hypre_ParCSRMatrix__object* self,
    int32_t nrows,
    struct SIDL_int__array* ncols,
    struct SIDL_int__array* rows,
    struct SIDL_int__array* cols,
    struct SIDL_double__array* values);
  int32_t (*f_SetRowSizes)(
    struct Hypre_ParCSRMatrix__object* self,
    struct SIDL_int__array* sizes);
  int32_t (*f_SetDiagOffdSizes)(
    struct Hypre_ParCSRMatrix__object* self,
    struct SIDL_int__array* diag_sizes,
    struct SIDL_int__array* offdiag_sizes);
  int32_t (*f_Read)(
    struct Hypre_ParCSRMatrix__object* self,
    const char* filename,
    void* comm);
  int32_t (*f_Print)(
    struct Hypre_ParCSRMatrix__object* self,
    const char* filename);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.Operator-v0.1.6 */
  int32_t (*f_GetDoubleValue)(
    struct Hypre_ParCSRMatrix__object* self,
    const char* name,
    double* value);
  int32_t (*f_GetIntValue)(
    struct Hypre_ParCSRMatrix__object* self,
    const char* name,
    int32_t* value);
  int32_t (*f_SetDoubleParameter)(
    struct Hypre_ParCSRMatrix__object* self,
    const char* name,
    double value);
  int32_t (*f_SetIntParameter)(
    struct Hypre_ParCSRMatrix__object* self,
    const char* name,
    int32_t value);
  int32_t (*f_SetStringParameter)(
    struct Hypre_ParCSRMatrix__object* self,
    const char* name,
    const char* value);
  int32_t (*f_SetIntArrayParameter)(
    struct Hypre_ParCSRMatrix__object* self,
    const char* name,
    struct SIDL_int__array* value);
  int32_t (*f_SetDoubleArrayParameter)(
    struct Hypre_ParCSRMatrix__object* self,
    const char* name,
    struct SIDL_double__array* value);
  int32_t (*f_Setup)(
    struct Hypre_ParCSRMatrix__object* self,
    struct Hypre_Vector__object* b,
    struct Hypre_Vector__object* x);
  int32_t (*f_Apply)(
    struct Hypre_ParCSRMatrix__object* self,
    struct Hypre_Vector__object* b,
    struct Hypre_Vector__object** x);
  /* Methods introduced in Hypre.ParCSRMatrix-v0.1.6 */
};

/*
 * Define the class object structure.
 */

struct Hypre_ParCSRMatrix__object {
  struct SIDL_BaseClass__object          d_sidl_baseclass;
  struct Hypre_CoefficientAccess__object d_hypre_coefficientaccess;
  struct Hypre_IJBuildMatrix__object     d_hypre_ijbuildmatrix;
  struct Hypre_Operator__object          d_hypre_operator;
  struct Hypre_ProblemDefinition__object d_hypre_problemdefinition;
  struct Hypre_ParCSRMatrix__epv*        d_epv;
  void*                                  d_data;
};

struct Hypre_ParCSRMatrix__external {
  struct Hypre_ParCSRMatrix__object*
  (*createObject)(void);

  struct Hypre_ParCSRMatrix__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_ParCSRMatrix__external*
Hypre_ParCSRMatrix__externals(void);

#ifdef __cplusplus
}
#endif
#endif
