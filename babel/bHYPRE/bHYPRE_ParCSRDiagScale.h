/*
 * File:          bHYPRE_ParCSRDiagScale.h
 * Symbol:        bHYPRE.ParCSRDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050208 15:29:05 PST
 * Generated:     20050208 15:29:06 PST
 * Description:   Client-side glue code for bHYPRE.ParCSRDiagScale
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 1140
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_ParCSRDiagScale_h
#define included_bHYPRE_ParCSRDiagScale_h

/**
 * Symbol "bHYPRE.ParCSRDiagScale" (version 1.0.0)
 * 
 * Diagonal scaling preconditioner for ParCSR matrix class.
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 */
struct bHYPRE_ParCSRDiagScale__object;
struct bHYPRE_ParCSRDiagScale__array;
typedef struct bHYPRE_ParCSRDiagScale__object* bHYPRE_ParCSRDiagScale;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
bHYPRE_ParCSRDiagScale
bHYPRE_ParCSRDiagScale__create(void);

void
bHYPRE_ParCSRDiagScale_addRef(
  /*in*/ bHYPRE_ParCSRDiagScale self);

void
bHYPRE_ParCSRDiagScale_deleteRef(
  /*in*/ bHYPRE_ParCSRDiagScale self);

sidl_bool
bHYPRE_ParCSRDiagScale_isSame(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_ParCSRDiagScale_queryInt(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ const char* name);

sidl_bool
bHYPRE_ParCSRDiagScale_isType(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ const char* name);

sidl_ClassInfo
bHYPRE_ParCSRDiagScale_getClassInfo(
  /*in*/ bHYPRE_ParCSRDiagScale self);

/**
 * Set the MPI Communicator.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_SetCommunicator(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ void* mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_SetIntParameter(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ const char* name,
  /*in*/ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_SetDoubleParameter(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ const char* name,
  /*in*/ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_SetStringParameter(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ const char* name,
  /*in*/ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_SetIntArray1Parameter(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_SetIntArray2Parameter(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_SetDoubleArray1Parameter(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_SetDoubleArray2Parameter(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_GetIntValue(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ const char* name,
  /*out*/ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_GetDoubleValue(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ const char* name,
  /*out*/ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_Setup(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ bHYPRE_Vector b,
  /*in*/ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_Apply(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ bHYPRE_Vector b,
  /*inout*/ bHYPRE_Vector* x);

/**
 * Set the operator for the linear system being solved.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_SetOperator(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ bHYPRE_Operator A);

/**
 * (Optional) Set the convergence tolerance.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_SetTolerance(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ double tolerance);

/**
 * (Optional) Set maximum number of iterations.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_SetMaxIterations(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ int32_t max_iterations);

/**
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_SetLogging(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ int32_t level);

/**
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_SetPrintLevel(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*in*/ int32_t level);

/**
 * (Optional) Return the number of iterations taken.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_GetNumIterations(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*out*/ int32_t* num_iterations);

/**
 * (Optional) Return the norm of the relative residual.
 * 
 */
int32_t
bHYPRE_ParCSRDiagScale_GetRelResidualNorm(
  /*in*/ bHYPRE_ParCSRDiagScale self,
  /*out*/ double* norm);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_ParCSRDiagScale
bHYPRE_ParCSRDiagScale__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_ParCSRDiagScale__cast2(
  void* obj,
  const char* type);

struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_create1d(int32_t len);

struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_create1dInit(
  int32_t len, 
  bHYPRE_ParCSRDiagScale* data);

struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_borrow(
  bHYPRE_ParCSRDiagScale* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_smartCopy(
  struct bHYPRE_ParCSRDiagScale__array *array);

void
bHYPRE_ParCSRDiagScale__array_addRef(
  struct bHYPRE_ParCSRDiagScale__array* array);

void
bHYPRE_ParCSRDiagScale__array_deleteRef(
  struct bHYPRE_ParCSRDiagScale__array* array);

bHYPRE_ParCSRDiagScale
bHYPRE_ParCSRDiagScale__array_get1(
  const struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t i1);

bHYPRE_ParCSRDiagScale
bHYPRE_ParCSRDiagScale__array_get2(
  const struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_ParCSRDiagScale
bHYPRE_ParCSRDiagScale__array_get3(
  const struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_ParCSRDiagScale
bHYPRE_ParCSRDiagScale__array_get4(
  const struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_ParCSRDiagScale
bHYPRE_ParCSRDiagScale__array_get5(
  const struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_ParCSRDiagScale
bHYPRE_ParCSRDiagScale__array_get6(
  const struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_ParCSRDiagScale
bHYPRE_ParCSRDiagScale__array_get7(
  const struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_ParCSRDiagScale
bHYPRE_ParCSRDiagScale__array_get(
  const struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t indices[]);

void
bHYPRE_ParCSRDiagScale__array_set1(
  struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t i1,
  bHYPRE_ParCSRDiagScale const value);

void
bHYPRE_ParCSRDiagScale__array_set2(
  struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_ParCSRDiagScale const value);

void
bHYPRE_ParCSRDiagScale__array_set3(
  struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_ParCSRDiagScale const value);

void
bHYPRE_ParCSRDiagScale__array_set4(
  struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_ParCSRDiagScale const value);

void
bHYPRE_ParCSRDiagScale__array_set5(
  struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_ParCSRDiagScale const value);

void
bHYPRE_ParCSRDiagScale__array_set6(
  struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_ParCSRDiagScale const value);

void
bHYPRE_ParCSRDiagScale__array_set7(
  struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_ParCSRDiagScale const value);

void
bHYPRE_ParCSRDiagScale__array_set(
  struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t indices[],
  bHYPRE_ParCSRDiagScale const value);

int32_t
bHYPRE_ParCSRDiagScale__array_dimen(
  const struct bHYPRE_ParCSRDiagScale__array* array);

int32_t
bHYPRE_ParCSRDiagScale__array_lower(
  const struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t ind);

int32_t
bHYPRE_ParCSRDiagScale__array_upper(
  const struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t ind);

int32_t
bHYPRE_ParCSRDiagScale__array_length(
  const struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t ind);

int32_t
bHYPRE_ParCSRDiagScale__array_stride(
  const struct bHYPRE_ParCSRDiagScale__array* array,
  const int32_t ind);

int
bHYPRE_ParCSRDiagScale__array_isColumnOrder(
  const struct bHYPRE_ParCSRDiagScale__array* array);

int
bHYPRE_ParCSRDiagScale__array_isRowOrder(
  const struct bHYPRE_ParCSRDiagScale__array* array);

struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_slice(
  struct bHYPRE_ParCSRDiagScale__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_ParCSRDiagScale__array_copy(
  const struct bHYPRE_ParCSRDiagScale__array* src,
  struct bHYPRE_ParCSRDiagScale__array* dest);

struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_ensure(
  struct bHYPRE_ParCSRDiagScale__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
