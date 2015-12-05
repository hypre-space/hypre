/*
 * File:          bHYPRE_Pilut.h
 * Symbol:        bHYPRE.Pilut-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:28 PST
 * Description:   Client-side glue code for bHYPRE.Pilut
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 1227
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_Pilut_h
#define included_bHYPRE_Pilut_h

/**
 * Symbol "bHYPRE.Pilut" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 */
struct bHYPRE_Pilut__object;
struct bHYPRE_Pilut__array;
typedef struct bHYPRE_Pilut__object* bHYPRE_Pilut;

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

/**
 * Constructor function for the class.
 */
bHYPRE_Pilut
bHYPRE_Pilut__create(void);

void
bHYPRE_Pilut_addRef(
  bHYPRE_Pilut self);

void
bHYPRE_Pilut_deleteRef(
  bHYPRE_Pilut self);

SIDL_bool
bHYPRE_Pilut_isSame(
  bHYPRE_Pilut self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_Pilut_queryInt(
  bHYPRE_Pilut self,
  const char* name);

SIDL_bool
bHYPRE_Pilut_isType(
  bHYPRE_Pilut self,
  const char* name);

SIDL_ClassInfo
bHYPRE_Pilut_getClassInfo(
  bHYPRE_Pilut self);

/**
 * Set the MPI Communicator.
 * 
 */
int32_t
bHYPRE_Pilut_SetCommunicator(
  bHYPRE_Pilut self,
  void* mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Pilut_SetIntParameter(
  bHYPRE_Pilut self,
  const char* name,
  int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Pilut_SetDoubleParameter(
  bHYPRE_Pilut self,
  const char* name,
  double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Pilut_SetStringParameter(
  bHYPRE_Pilut self,
  const char* name,
  const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Pilut_SetIntArray1Parameter(
  bHYPRE_Pilut self,
  const char* name,
  struct SIDL_int__array* value);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Pilut_SetIntArray2Parameter(
  bHYPRE_Pilut self,
  const char* name,
  struct SIDL_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Pilut_SetDoubleArray1Parameter(
  bHYPRE_Pilut self,
  const char* name,
  struct SIDL_double__array* value);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Pilut_SetDoubleArray2Parameter(
  bHYPRE_Pilut self,
  const char* name,
  struct SIDL_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Pilut_GetIntValue(
  bHYPRE_Pilut self,
  const char* name,
  int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Pilut_GetDoubleValue(
  bHYPRE_Pilut self,
  const char* name,
  double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_Pilut_Setup(
  bHYPRE_Pilut self,
  bHYPRE_Vector b,
  bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_Pilut_Apply(
  bHYPRE_Pilut self,
  bHYPRE_Vector b,
  bHYPRE_Vector* x);

/**
 * Set the operator for the linear system being solved.
 * 
 */
int32_t
bHYPRE_Pilut_SetOperator(
  bHYPRE_Pilut self,
  bHYPRE_Operator A);

/**
 * (Optional) Set the convergence tolerance.
 * 
 */
int32_t
bHYPRE_Pilut_SetTolerance(
  bHYPRE_Pilut self,
  double tolerance);

/**
 * (Optional) Set maximum number of iterations.
 * 
 */
int32_t
bHYPRE_Pilut_SetMaxIterations(
  bHYPRE_Pilut self,
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
bHYPRE_Pilut_SetLogging(
  bHYPRE_Pilut self,
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
bHYPRE_Pilut_SetPrintLevel(
  bHYPRE_Pilut self,
  int32_t level);

/**
 * (Optional) Return the number of iterations taken.
 * 
 */
int32_t
bHYPRE_Pilut_GetNumIterations(
  bHYPRE_Pilut self,
  int32_t* num_iterations);

/**
 * (Optional) Return the norm of the relative residual.
 * 
 */
int32_t
bHYPRE_Pilut_GetRelResidualNorm(
  bHYPRE_Pilut self,
  double* norm);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_Pilut
bHYPRE_Pilut__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_Pilut__cast2(
  void* obj,
  const char* type);

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_createCol(int32_t        dimen,
                              const int32_t lower[],
                              const int32_t upper[]);

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_createRow(int32_t        dimen,
                              const int32_t lower[],
                              const int32_t upper[]);

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_create1d(int32_t len);

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_borrow(bHYPRE_Pilut*firstElement,
                           int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_smartCopy(struct bHYPRE_Pilut__array *array);

void
bHYPRE_Pilut__array_addRef(struct bHYPRE_Pilut__array* array);

void
bHYPRE_Pilut__array_deleteRef(struct bHYPRE_Pilut__array* array);

bHYPRE_Pilut
bHYPRE_Pilut__array_get1(const struct bHYPRE_Pilut__array* array,
                         const int32_t i1);

bHYPRE_Pilut
bHYPRE_Pilut__array_get2(const struct bHYPRE_Pilut__array* array,
                         const int32_t i1,
                         const int32_t i2);

bHYPRE_Pilut
bHYPRE_Pilut__array_get3(const struct bHYPRE_Pilut__array* array,
                         const int32_t i1,
                         const int32_t i2,
                         const int32_t i3);

bHYPRE_Pilut
bHYPRE_Pilut__array_get4(const struct bHYPRE_Pilut__array* array,
                         const int32_t i1,
                         const int32_t i2,
                         const int32_t i3,
                         const int32_t i4);

bHYPRE_Pilut
bHYPRE_Pilut__array_get(const struct bHYPRE_Pilut__array* array,
                        const int32_t indices[]);

void
bHYPRE_Pilut__array_set1(struct bHYPRE_Pilut__array* array,
                         const int32_t i1,
                         bHYPRE_Pilut const value);

void
bHYPRE_Pilut__array_set2(struct bHYPRE_Pilut__array* array,
                         const int32_t i1,
                         const int32_t i2,
                         bHYPRE_Pilut const value);

void
bHYPRE_Pilut__array_set3(struct bHYPRE_Pilut__array* array,
                         const int32_t i1,
                         const int32_t i2,
                         const int32_t i3,
                         bHYPRE_Pilut const value);

void
bHYPRE_Pilut__array_set4(struct bHYPRE_Pilut__array* array,
                         const int32_t i1,
                         const int32_t i2,
                         const int32_t i3,
                         const int32_t i4,
                         bHYPRE_Pilut const value);

void
bHYPRE_Pilut__array_set(struct bHYPRE_Pilut__array* array,
                        const int32_t indices[],
                        bHYPRE_Pilut const value);

int32_t
bHYPRE_Pilut__array_dimen(const struct bHYPRE_Pilut__array* array);

int32_t
bHYPRE_Pilut__array_lower(const struct bHYPRE_Pilut__array* array,
                          const int32_t ind);

int32_t
bHYPRE_Pilut__array_upper(const struct bHYPRE_Pilut__array* array,
                          const int32_t ind);

int32_t
bHYPRE_Pilut__array_stride(const struct bHYPRE_Pilut__array* array,
                           const int32_t ind);

int
bHYPRE_Pilut__array_isColumnOrder(const struct bHYPRE_Pilut__array* array);

int
bHYPRE_Pilut__array_isRowOrder(const struct bHYPRE_Pilut__array* array);

void
bHYPRE_Pilut__array_slice(const struct bHYPRE_Pilut__array* src,
                                int32_t        dimen,
                                const int32_t  numElem[],
                                const int32_t  *srcStart,
                                const int32_t  *srcStride,
                                const int32_t  *newStart);

void
bHYPRE_Pilut__array_copy(const struct bHYPRE_Pilut__array* src,
                               struct bHYPRE_Pilut__array* dest);

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_ensure(struct bHYPRE_Pilut__array* src,
                           int32_t dimen,
                           int     ordering);

#ifdef __cplusplus
}
#endif
#endif
