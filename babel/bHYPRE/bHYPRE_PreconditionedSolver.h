/*
 * File:          bHYPRE_PreconditionedSolver.h
 * Symbol:        bHYPRE.PreconditionedSolver-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:29 PST
 * Description:   Client-side glue code for bHYPRE.PreconditionedSolver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 756
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_PreconditionedSolver_h
#define included_bHYPRE_PreconditionedSolver_h

/**
 * Symbol "bHYPRE.PreconditionedSolver" (version 1.0.0)
 */
struct bHYPRE_PreconditionedSolver__object;
struct bHYPRE_PreconditionedSolver__array;
typedef struct bHYPRE_PreconditionedSolver__object* bHYPRE_PreconditionedSolver;

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
#ifndef included_bHYPRE_Solver_h
#include "bHYPRE_Solver.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_PreconditionedSolver_addRef(
  bHYPRE_PreconditionedSolver self);

void
bHYPRE_PreconditionedSolver_deleteRef(
  bHYPRE_PreconditionedSolver self);

SIDL_bool
bHYPRE_PreconditionedSolver_isSame(
  bHYPRE_PreconditionedSolver self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_PreconditionedSolver_queryInt(
  bHYPRE_PreconditionedSolver self,
  const char* name);

SIDL_bool
bHYPRE_PreconditionedSolver_isType(
  bHYPRE_PreconditionedSolver self,
  const char* name);

SIDL_ClassInfo
bHYPRE_PreconditionedSolver_getClassInfo(
  bHYPRE_PreconditionedSolver self);

int32_t
bHYPRE_PreconditionedSolver_SetCommunicator(
  bHYPRE_PreconditionedSolver self,
  void* mpi_comm);

int32_t
bHYPRE_PreconditionedSolver_SetIntParameter(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  int32_t value);

int32_t
bHYPRE_PreconditionedSolver_SetDoubleParameter(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  double value);

int32_t
bHYPRE_PreconditionedSolver_SetStringParameter(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  const char* value);

int32_t
bHYPRE_PreconditionedSolver_SetIntArray1Parameter(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  struct SIDL_int__array* value);

int32_t
bHYPRE_PreconditionedSolver_SetIntArray2Parameter(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  struct SIDL_int__array* value);

int32_t
bHYPRE_PreconditionedSolver_SetDoubleArray1Parameter(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  struct SIDL_double__array* value);

int32_t
bHYPRE_PreconditionedSolver_SetDoubleArray2Parameter(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  struct SIDL_double__array* value);

int32_t
bHYPRE_PreconditionedSolver_GetIntValue(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  int32_t* value);

int32_t
bHYPRE_PreconditionedSolver_GetDoubleValue(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  double* value);

int32_t
bHYPRE_PreconditionedSolver_Setup(
  bHYPRE_PreconditionedSolver self,
  bHYPRE_Vector b,
  bHYPRE_Vector x);

int32_t
bHYPRE_PreconditionedSolver_Apply(
  bHYPRE_PreconditionedSolver self,
  bHYPRE_Vector b,
  bHYPRE_Vector* x);

int32_t
bHYPRE_PreconditionedSolver_SetOperator(
  bHYPRE_PreconditionedSolver self,
  bHYPRE_Operator A);

int32_t
bHYPRE_PreconditionedSolver_SetTolerance(
  bHYPRE_PreconditionedSolver self,
  double tolerance);

int32_t
bHYPRE_PreconditionedSolver_SetMaxIterations(
  bHYPRE_PreconditionedSolver self,
  int32_t max_iterations);

int32_t
bHYPRE_PreconditionedSolver_SetLogging(
  bHYPRE_PreconditionedSolver self,
  int32_t level);

int32_t
bHYPRE_PreconditionedSolver_SetPrintLevel(
  bHYPRE_PreconditionedSolver self,
  int32_t level);

int32_t
bHYPRE_PreconditionedSolver_GetNumIterations(
  bHYPRE_PreconditionedSolver self,
  int32_t* num_iterations);

int32_t
bHYPRE_PreconditionedSolver_GetRelResidualNorm(
  bHYPRE_PreconditionedSolver self,
  double* norm);

/**
 * Set the preconditioner.
 * 
 */
int32_t
bHYPRE_PreconditionedSolver_SetPreconditioner(
  bHYPRE_PreconditionedSolver self,
  bHYPRE_Solver s);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_PreconditionedSolver__cast2(
  void* obj,
  const char* type);

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_createCol(int32_t        dimen,
                                             const int32_t lower[],
                                             const int32_t upper[]);

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_createRow(int32_t        dimen,
                                             const int32_t lower[],
                                             const int32_t upper[]);

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_create1d(int32_t len);

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_borrow(
  bHYPRE_PreconditionedSolver*firstElement,
                                          int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_smartCopy(struct 
  bHYPRE_PreconditionedSolver__array *array);

void
bHYPRE_PreconditionedSolver__array_addRef(struct 
  bHYPRE_PreconditionedSolver__array* array);

void
bHYPRE_PreconditionedSolver__array_deleteRef(struct 
  bHYPRE_PreconditionedSolver__array* array);

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get1(const struct 
  bHYPRE_PreconditionedSolver__array* array,
                                        const int32_t i1);

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get2(const struct 
  bHYPRE_PreconditionedSolver__array* array,
                                        const int32_t i1,
                                        const int32_t i2);

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get3(const struct 
  bHYPRE_PreconditionedSolver__array* array,
                                        const int32_t i1,
                                        const int32_t i2,
                                        const int32_t i3);

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get4(const struct 
  bHYPRE_PreconditionedSolver__array* array,
                                        const int32_t i1,
                                        const int32_t i2,
                                        const int32_t i3,
                                        const int32_t i4);

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get(const struct 
  bHYPRE_PreconditionedSolver__array* array,
                                       const int32_t indices[]);

void
bHYPRE_PreconditionedSolver__array_set1(struct 
  bHYPRE_PreconditionedSolver__array* array,
                                        const int32_t i1,
                                        bHYPRE_PreconditionedSolver const 
  value);

void
bHYPRE_PreconditionedSolver__array_set2(struct 
  bHYPRE_PreconditionedSolver__array* array,
                                        const int32_t i1,
                                        const int32_t i2,
                                        bHYPRE_PreconditionedSolver const 
  value);

void
bHYPRE_PreconditionedSolver__array_set3(struct 
  bHYPRE_PreconditionedSolver__array* array,
                                        const int32_t i1,
                                        const int32_t i2,
                                        const int32_t i3,
                                        bHYPRE_PreconditionedSolver const 
  value);

void
bHYPRE_PreconditionedSolver__array_set4(struct 
  bHYPRE_PreconditionedSolver__array* array,
                                        const int32_t i1,
                                        const int32_t i2,
                                        const int32_t i3,
                                        const int32_t i4,
                                        bHYPRE_PreconditionedSolver const 
  value);

void
bHYPRE_PreconditionedSolver__array_set(struct 
  bHYPRE_PreconditionedSolver__array* array,
                                       const int32_t indices[],
                                       bHYPRE_PreconditionedSolver const value);

int32_t
bHYPRE_PreconditionedSolver__array_dimen(const struct 
  bHYPRE_PreconditionedSolver__array* array);

int32_t
bHYPRE_PreconditionedSolver__array_lower(const struct 
  bHYPRE_PreconditionedSolver__array* array,
                                         const int32_t ind);

int32_t
bHYPRE_PreconditionedSolver__array_upper(const struct 
  bHYPRE_PreconditionedSolver__array* array,
                                         const int32_t ind);

int32_t
bHYPRE_PreconditionedSolver__array_stride(const struct 
  bHYPRE_PreconditionedSolver__array* array,
                                          const int32_t ind);

int
bHYPRE_PreconditionedSolver__array_isColumnOrder(const struct 
  bHYPRE_PreconditionedSolver__array* array);

int
bHYPRE_PreconditionedSolver__array_isRowOrder(const struct 
  bHYPRE_PreconditionedSolver__array* array);

void
bHYPRE_PreconditionedSolver__array_slice(const struct 
  bHYPRE_PreconditionedSolver__array* src,
                                               int32_t        dimen,
                                               const int32_t  numElem[],
                                               const int32_t  *srcStart,
                                               const int32_t  *srcStride,
                                               const int32_t  *newStart);

void
bHYPRE_PreconditionedSolver__array_copy(const struct 
  bHYPRE_PreconditionedSolver__array* src,
                                              struct 
  bHYPRE_PreconditionedSolver__array* dest);

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_ensure(struct 
  bHYPRE_PreconditionedSolver__array* src,
                                          int32_t dimen,
                                          int     ordering);

#ifdef __cplusplus
}
#endif
#endif
