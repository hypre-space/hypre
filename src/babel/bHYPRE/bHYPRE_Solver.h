/*
 * File:          bHYPRE_Solver.h
 * Symbol:        bHYPRE.Solver-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:25 PST
 * Description:   Client-side glue code for bHYPRE.Solver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 708
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_Solver_h
#define included_bHYPRE_Solver_h

/**
 * Symbol "bHYPRE.Solver" (version 1.0.0)
 */
struct bHYPRE_Solver__object;
struct bHYPRE_Solver__array;
typedef struct bHYPRE_Solver__object* bHYPRE_Solver;

/*
 * Includes for all header dependencies.
 */

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_SIDL_ClassInfo_h
#include "SIDL_ClassInfo.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_Solver_addRef(
  bHYPRE_Solver self);

void
bHYPRE_Solver_deleteRef(
  bHYPRE_Solver self);

SIDL_bool
bHYPRE_Solver_isSame(
  bHYPRE_Solver self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_Solver_queryInt(
  bHYPRE_Solver self,
  const char* name);

SIDL_bool
bHYPRE_Solver_isType(
  bHYPRE_Solver self,
  const char* name);

SIDL_ClassInfo
bHYPRE_Solver_getClassInfo(
  bHYPRE_Solver self);

int32_t
bHYPRE_Solver_SetCommunicator(
  bHYPRE_Solver self,
  void* mpi_comm);

int32_t
bHYPRE_Solver_SetIntParameter(
  bHYPRE_Solver self,
  const char* name,
  int32_t value);

int32_t
bHYPRE_Solver_SetDoubleParameter(
  bHYPRE_Solver self,
  const char* name,
  double value);

int32_t
bHYPRE_Solver_SetStringParameter(
  bHYPRE_Solver self,
  const char* name,
  const char* value);

int32_t
bHYPRE_Solver_SetIntArray1Parameter(
  bHYPRE_Solver self,
  const char* name,
  struct SIDL_int__array* value);

int32_t
bHYPRE_Solver_SetIntArray2Parameter(
  bHYPRE_Solver self,
  const char* name,
  struct SIDL_int__array* value);

int32_t
bHYPRE_Solver_SetDoubleArray1Parameter(
  bHYPRE_Solver self,
  const char* name,
  struct SIDL_double__array* value);

int32_t
bHYPRE_Solver_SetDoubleArray2Parameter(
  bHYPRE_Solver self,
  const char* name,
  struct SIDL_double__array* value);

int32_t
bHYPRE_Solver_GetIntValue(
  bHYPRE_Solver self,
  const char* name,
  int32_t* value);

int32_t
bHYPRE_Solver_GetDoubleValue(
  bHYPRE_Solver self,
  const char* name,
  double* value);

int32_t
bHYPRE_Solver_Setup(
  bHYPRE_Solver self,
  bHYPRE_Vector b,
  bHYPRE_Vector x);

int32_t
bHYPRE_Solver_Apply(
  bHYPRE_Solver self,
  bHYPRE_Vector b,
  bHYPRE_Vector* x);

/**
 * Set the operator for the linear system being solved.
 * 
 */
int32_t
bHYPRE_Solver_SetOperator(
  bHYPRE_Solver self,
  bHYPRE_Operator A);

/**
 * (Optional) Set the convergence tolerance.
 * 
 */
int32_t
bHYPRE_Solver_SetTolerance(
  bHYPRE_Solver self,
  double tolerance);

/**
 * (Optional) Set maximum number of iterations.
 * 
 */
int32_t
bHYPRE_Solver_SetMaxIterations(
  bHYPRE_Solver self,
  int32_t max_iterations);

/**
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * 
 */
int32_t
bHYPRE_Solver_SetLogging(
  bHYPRE_Solver self,
  int32_t level);

/**
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * 
 */
int32_t
bHYPRE_Solver_SetPrintLevel(
  bHYPRE_Solver self,
  int32_t level);

/**
 * (Optional) Return the number of iterations taken.
 * 
 */
int32_t
bHYPRE_Solver_GetNumIterations(
  bHYPRE_Solver self,
  int32_t* num_iterations);

/**
 * (Optional) Return the norm of the relative residual.
 * 
 */
int32_t
bHYPRE_Solver_GetRelResidualNorm(
  bHYPRE_Solver self,
  double* norm);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_Solver
bHYPRE_Solver__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_Solver__cast2(
  void* obj,
  const char* type);

struct bHYPRE_Solver__array*
bHYPRE_Solver__array_createCol(int32_t        dimen,
                               const int32_t lower[],
                               const int32_t upper[]);

struct bHYPRE_Solver__array*
bHYPRE_Solver__array_createRow(int32_t        dimen,
                               const int32_t lower[],
                               const int32_t upper[]);

struct bHYPRE_Solver__array*
bHYPRE_Solver__array_create1d(int32_t len);

struct bHYPRE_Solver__array*
bHYPRE_Solver__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_Solver__array*
bHYPRE_Solver__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_Solver__array*
bHYPRE_Solver__array_borrow(bHYPRE_Solver*firstElement,
                            int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_Solver__array*
bHYPRE_Solver__array_smartCopy(struct bHYPRE_Solver__array *array);

void
bHYPRE_Solver__array_addRef(struct bHYPRE_Solver__array* array);

void
bHYPRE_Solver__array_deleteRef(struct bHYPRE_Solver__array* array);

bHYPRE_Solver
bHYPRE_Solver__array_get1(const struct bHYPRE_Solver__array* array,
                          const int32_t i1);

bHYPRE_Solver
bHYPRE_Solver__array_get2(const struct bHYPRE_Solver__array* array,
                          const int32_t i1,
                          const int32_t i2);

bHYPRE_Solver
bHYPRE_Solver__array_get3(const struct bHYPRE_Solver__array* array,
                          const int32_t i1,
                          const int32_t i2,
                          const int32_t i3);

bHYPRE_Solver
bHYPRE_Solver__array_get4(const struct bHYPRE_Solver__array* array,
                          const int32_t i1,
                          const int32_t i2,
                          const int32_t i3,
                          const int32_t i4);

bHYPRE_Solver
bHYPRE_Solver__array_get(const struct bHYPRE_Solver__array* array,
                         const int32_t indices[]);

void
bHYPRE_Solver__array_set1(struct bHYPRE_Solver__array* array,
                          const int32_t i1,
                          bHYPRE_Solver const value);

void
bHYPRE_Solver__array_set2(struct bHYPRE_Solver__array* array,
                          const int32_t i1,
                          const int32_t i2,
                          bHYPRE_Solver const value);

void
bHYPRE_Solver__array_set3(struct bHYPRE_Solver__array* array,
                          const int32_t i1,
                          const int32_t i2,
                          const int32_t i3,
                          bHYPRE_Solver const value);

void
bHYPRE_Solver__array_set4(struct bHYPRE_Solver__array* array,
                          const int32_t i1,
                          const int32_t i2,
                          const int32_t i3,
                          const int32_t i4,
                          bHYPRE_Solver const value);

void
bHYPRE_Solver__array_set(struct bHYPRE_Solver__array* array,
                         const int32_t indices[],
                         bHYPRE_Solver const value);

int32_t
bHYPRE_Solver__array_dimen(const struct bHYPRE_Solver__array* array);

int32_t
bHYPRE_Solver__array_lower(const struct bHYPRE_Solver__array* array,
                           const int32_t ind);

int32_t
bHYPRE_Solver__array_upper(const struct bHYPRE_Solver__array* array,
                           const int32_t ind);

int32_t
bHYPRE_Solver__array_stride(const struct bHYPRE_Solver__array* array,
                            const int32_t ind);

int
bHYPRE_Solver__array_isColumnOrder(const struct bHYPRE_Solver__array* array);

int
bHYPRE_Solver__array_isRowOrder(const struct bHYPRE_Solver__array* array);

void
bHYPRE_Solver__array_slice(const struct bHYPRE_Solver__array* src,
                                 int32_t        dimen,
                                 const int32_t  numElem[],
                                 const int32_t  *srcStart,
                                 const int32_t  *srcStride,
                                 const int32_t  *newStart);

void
bHYPRE_Solver__array_copy(const struct bHYPRE_Solver__array* src,
                                struct bHYPRE_Solver__array* dest);

struct bHYPRE_Solver__array*
bHYPRE_Solver__array_ensure(struct bHYPRE_Solver__array* src,
                            int32_t dimen,
                            int     ordering);

#ifdef __cplusplus
}
#endif
#endif
