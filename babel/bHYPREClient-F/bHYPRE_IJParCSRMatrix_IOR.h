/*
 * File:          bHYPRE_IJParCSRMatrix_IOR.h
 * Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:50 PST
 * Generated:     20050317 11:17:52 PST
 * Description:   Intermediate Object Representation for bHYPRE.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 794
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_IJParCSRMatrix_IOR_h
#define included_bHYPRE_IJParCSRMatrix_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_CoefficientAccess_IOR_h
#include "bHYPRE_CoefficientAccess_IOR.h"
#endif
#ifndef included_bHYPRE_IJBuildMatrix_IOR_h
#include "bHYPRE_IJBuildMatrix_IOR.h"
#endif
#ifndef included_bHYPRE_Operator_IOR_h
#include "bHYPRE_Operator_IOR.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_IOR_h
#include "bHYPRE_ProblemDefinition_IOR.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.IJParCSRMatrix" (version 1.0.0)
 * 
 * The IJParCSR matrix class.
 * 
 * Objects of this type can be cast to IJBuildMatrix, Operator, or
 * CoefficientAccess objects using the {\tt \_\_cast} methods.
 * 
 */

struct bHYPRE_IJParCSRMatrix__array;
struct bHYPRE_IJParCSRMatrix__object;

extern struct bHYPRE_IJParCSRMatrix__object*
bHYPRE_IJParCSRMatrix__new(void);

extern struct bHYPRE_IJParCSRMatrix__object*
bHYPRE_IJParCSRMatrix__remote(const char *url);

extern void bHYPRE_IJParCSRMatrix__init(
  struct bHYPRE_IJParCSRMatrix__object* self);
extern void bHYPRE_IJParCSRMatrix__fini(
  struct bHYPRE_IJParCSRMatrix__object* self);
extern void bHYPRE_IJParCSRMatrix__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_Vector__array;
struct bHYPRE_Vector__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_IJParCSRMatrix__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    const char* name);
  void (*f__delete)(
    struct bHYPRE_IJParCSRMatrix__object* self);
  void (*f__ctor)(
    struct bHYPRE_IJParCSRMatrix__object* self);
  void (*f__dtor)(
    struct bHYPRE_IJParCSRMatrix__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.0 */
  void (*f_addRef)(
    struct bHYPRE_IJParCSRMatrix__object* self);
  void (*f_deleteRef)(
    struct bHYPRE_IJParCSRMatrix__object* self);
  sidl_bool (*f_isSame)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    const char* name);
  sidl_bool (*f_isType)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    struct bHYPRE_IJParCSRMatrix__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.0 */
  /* Methods introduced in bHYPRE.CoefficientAccess-v1.0.0 */
  int32_t (*f_GetRow)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    int32_t row,
    int32_t* size,
    struct sidl_int__array** col_ind,
    struct sidl_double__array** values);
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    void* mpi_comm);
  int32_t (*f_Initialize)(
    struct bHYPRE_IJParCSRMatrix__object* self);
  int32_t (*f_Assemble)(
    struct bHYPRE_IJParCSRMatrix__object* self);
  int32_t (*f_GetObject)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    struct sidl_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.IJBuildMatrix-v1.0.0 */
  int32_t (*f_SetLocalRange)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    int32_t ilower,
    int32_t iupper,
    int32_t jlower,
    int32_t jupper);
  int32_t (*f_SetValues)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    int32_t nrows,
    struct sidl_int__array* ncols,
    struct sidl_int__array* rows,
    struct sidl_int__array* cols,
    struct sidl_double__array* values);
  int32_t (*f_AddToValues)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    int32_t nrows,
    struct sidl_int__array* ncols,
    struct sidl_int__array* rows,
    struct sidl_int__array* cols,
    struct sidl_double__array* values);
  int32_t (*f_GetLocalRange)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    int32_t* ilower,
    int32_t* iupper,
    int32_t* jlower,
    int32_t* jupper);
  int32_t (*f_GetRowCounts)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    int32_t nrows,
    struct sidl_int__array* rows,
    struct sidl_int__array** ncols);
  int32_t (*f_GetValues)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    int32_t nrows,
    struct sidl_int__array* ncols,
    struct sidl_int__array* rows,
    struct sidl_int__array* cols,
    struct sidl_double__array** values);
  int32_t (*f_SetRowSizes)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    struct sidl_int__array* sizes);
  int32_t (*f_Print)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    const char* filename);
  int32_t (*f_Read)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    const char* filename,
    void* comm);
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  int32_t (*f_SetIntParameter)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    const char* name,
    int32_t value);
  int32_t (*f_SetDoubleParameter)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    const char* name,
    double value);
  int32_t (*f_SetStringParameter)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    const char* name,
    const char* value);
  int32_t (*f_SetIntArray1Parameter)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    const char* name,
    struct sidl_int__array* value);
  int32_t (*f_SetIntArray2Parameter)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    const char* name,
    struct sidl_int__array* value);
  int32_t (*f_SetDoubleArray1Parameter)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    const char* name,
    struct sidl_double__array* value);
  int32_t (*f_SetDoubleArray2Parameter)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    const char* name,
    struct sidl_double__array* value);
  int32_t (*f_GetIntValue)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    const char* name,
    int32_t* value);
  int32_t (*f_GetDoubleValue)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    const char* name,
    double* value);
  int32_t (*f_Setup)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    struct bHYPRE_Vector__object* b,
    struct bHYPRE_Vector__object* x);
  int32_t (*f_Apply)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    struct bHYPRE_Vector__object* b,
    struct bHYPRE_Vector__object** x);
  /* Methods introduced in bHYPRE.IJParCSRMatrix-v1.0.0 */
  int32_t (*f_SetDiagOffdSizes)(
    struct bHYPRE_IJParCSRMatrix__object* self,
    struct sidl_int__array* diag_sizes,
    struct sidl_int__array* offdiag_sizes);
};

/*
 * Define the class object structure.
 */

struct bHYPRE_IJParCSRMatrix__object {
  struct sidl_BaseClass__object           d_sidl_baseclass;
  struct bHYPRE_CoefficientAccess__object d_bhypre_coefficientaccess;
  struct bHYPRE_IJBuildMatrix__object     d_bhypre_ijbuildmatrix;
  struct bHYPRE_Operator__object          d_bhypre_operator;
  struct bHYPRE_ProblemDefinition__object d_bhypre_problemdefinition;
  struct bHYPRE_IJParCSRMatrix__epv*      d_epv;
  void*                                   d_data;
};

struct bHYPRE_IJParCSRMatrix__external {
  struct bHYPRE_IJParCSRMatrix__object*
  (*createObject)(void);

  struct bHYPRE_IJParCSRMatrix__object*
  (*createRemote)(const char *url);

struct sidl_BaseClass__epv*(*getSuperEPV)(void);};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_IJParCSRMatrix__external*
bHYPRE_IJParCSRMatrix__externals(void);

#ifdef __cplusplus
}
#endif
#endif
