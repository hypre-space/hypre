/*
 * File:          bHYPRE_HGMRES.h
 * Symbol:        bHYPRE.HGMRES-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.HGMRES
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_HGMRES_h
#define included_bHYPRE_HGMRES_h

/**
 * Symbol "bHYPRE.HGMRES" (version 1.0.0)
 */
struct bHYPRE_HGMRES__object;
struct bHYPRE_HGMRES__array;
typedef struct bHYPRE_HGMRES__object* bHYPRE_HGMRES;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_bHYPRE_PreconditionedSolver_h
#include "bHYPRE_PreconditionedSolver.h"
#endif
#ifndef included_bHYPRE_Solver_h
#include "bHYPRE_Solver.h"
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

#ifndef included_sidl_io_Serializer_h
#include "sidl_io_Serializer.h"
#endif
#ifndef included_sidl_io_Deserializer_h
#include "sidl_io_Deserializer.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
struct bHYPRE_HGMRES__object*
bHYPRE_HGMRES__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_HGMRES
bHYPRE_HGMRES__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_HGMRES
bHYPRE_HGMRES__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_HGMRES_addRef(
  /* in */ bHYPRE_HGMRES self);

void
bHYPRE_HGMRES_deleteRef(
  /* in */ bHYPRE_HGMRES self);

sidl_bool
bHYPRE_HGMRES_isSame(
  /* in */ bHYPRE_HGMRES self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_HGMRES_queryInt(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name);

sidl_bool
bHYPRE_HGMRES_isType(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_HGMRES_getClassInfo(
  /* in */ bHYPRE_HGMRES self);

/**
 * Method:  Create[]
 */
bHYPRE_HGMRES
bHYPRE_HGMRES_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */
int32_t
bHYPRE_HGMRES_SetCommunicator(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HGMRES_SetIntParameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HGMRES_SetDoubleParameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HGMRES_SetStringParameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HGMRES_SetIntArray1Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HGMRES_SetIntArray2Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HGMRES_SetDoubleArray1Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HGMRES_SetDoubleArray2Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HGMRES_GetIntValue(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* out */ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HGMRES_GetDoubleValue(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* out */ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_HGMRES_Setup(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_HGMRES_Apply(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_HGMRES_ApplyAdjoint(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 * 
 */
int32_t
bHYPRE_HGMRES_SetOperator(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Operator A);

/**
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */
int32_t
bHYPRE_HGMRES_SetTolerance(
  /* in */ bHYPRE_HGMRES self,
  /* in */ double tolerance);

/**
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */
int32_t
bHYPRE_HGMRES_SetMaxIterations(
  /* in */ bHYPRE_HGMRES self,
  /* in */ int32_t max_iterations);

/**
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 * 
 */
int32_t
bHYPRE_HGMRES_SetLogging(
  /* in */ bHYPRE_HGMRES self,
  /* in */ int32_t level);

/**
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 * 
 */
int32_t
bHYPRE_HGMRES_SetPrintLevel(
  /* in */ bHYPRE_HGMRES self,
  /* in */ int32_t level);

/**
 * (Optional) Return the number of iterations taken.
 * 
 */
int32_t
bHYPRE_HGMRES_GetNumIterations(
  /* in */ bHYPRE_HGMRES self,
  /* out */ int32_t* num_iterations);

/**
 * (Optional) Return the norm of the relative residual.
 * 
 */
int32_t
bHYPRE_HGMRES_GetRelResidualNorm(
  /* in */ bHYPRE_HGMRES self,
  /* out */ double* norm);

/**
 * Set the preconditioner.
 * 
 */
int32_t
bHYPRE_HGMRES_SetPreconditioner(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Solver s);

/**
 * Method:  GetPreconditioner[]
 */
int32_t
bHYPRE_HGMRES_GetPreconditioner(
  /* in */ bHYPRE_HGMRES self,
  /* out */ bHYPRE_Solver* s);

/**
 * Method:  Clone[]
 */
int32_t
bHYPRE_HGMRES_Clone(
  /* in */ bHYPRE_HGMRES self,
  /* out */ bHYPRE_PreconditionedSolver* x);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_HGMRES__object*
bHYPRE_HGMRES__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_HGMRES__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_HGMRES__exec(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_HGMRES__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_HGMRES__getURL(
  /* in */ bHYPRE_HGMRES self);
struct bHYPRE_HGMRES__array*
bHYPRE_HGMRES__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_HGMRES__array*
bHYPRE_HGMRES__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_HGMRES__array*
bHYPRE_HGMRES__array_create1d(int32_t len);

struct bHYPRE_HGMRES__array*
bHYPRE_HGMRES__array_create1dInit(
  int32_t len, 
  bHYPRE_HGMRES* data);

struct bHYPRE_HGMRES__array*
bHYPRE_HGMRES__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_HGMRES__array*
bHYPRE_HGMRES__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_HGMRES__array*
bHYPRE_HGMRES__array_borrow(
  bHYPRE_HGMRES* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_HGMRES__array*
bHYPRE_HGMRES__array_smartCopy(
  struct bHYPRE_HGMRES__array *array);

void
bHYPRE_HGMRES__array_addRef(
  struct bHYPRE_HGMRES__array* array);

void
bHYPRE_HGMRES__array_deleteRef(
  struct bHYPRE_HGMRES__array* array);

bHYPRE_HGMRES
bHYPRE_HGMRES__array_get1(
  const struct bHYPRE_HGMRES__array* array,
  const int32_t i1);

bHYPRE_HGMRES
bHYPRE_HGMRES__array_get2(
  const struct bHYPRE_HGMRES__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_HGMRES
bHYPRE_HGMRES__array_get3(
  const struct bHYPRE_HGMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_HGMRES
bHYPRE_HGMRES__array_get4(
  const struct bHYPRE_HGMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_HGMRES
bHYPRE_HGMRES__array_get5(
  const struct bHYPRE_HGMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_HGMRES
bHYPRE_HGMRES__array_get6(
  const struct bHYPRE_HGMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_HGMRES
bHYPRE_HGMRES__array_get7(
  const struct bHYPRE_HGMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_HGMRES
bHYPRE_HGMRES__array_get(
  const struct bHYPRE_HGMRES__array* array,
  const int32_t indices[]);

void
bHYPRE_HGMRES__array_set1(
  struct bHYPRE_HGMRES__array* array,
  const int32_t i1,
  bHYPRE_HGMRES const value);

void
bHYPRE_HGMRES__array_set2(
  struct bHYPRE_HGMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_HGMRES const value);

void
bHYPRE_HGMRES__array_set3(
  struct bHYPRE_HGMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_HGMRES const value);

void
bHYPRE_HGMRES__array_set4(
  struct bHYPRE_HGMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_HGMRES const value);

void
bHYPRE_HGMRES__array_set5(
  struct bHYPRE_HGMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_HGMRES const value);

void
bHYPRE_HGMRES__array_set6(
  struct bHYPRE_HGMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_HGMRES const value);

void
bHYPRE_HGMRES__array_set7(
  struct bHYPRE_HGMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_HGMRES const value);

void
bHYPRE_HGMRES__array_set(
  struct bHYPRE_HGMRES__array* array,
  const int32_t indices[],
  bHYPRE_HGMRES const value);

int32_t
bHYPRE_HGMRES__array_dimen(
  const struct bHYPRE_HGMRES__array* array);

int32_t
bHYPRE_HGMRES__array_lower(
  const struct bHYPRE_HGMRES__array* array,
  const int32_t ind);

int32_t
bHYPRE_HGMRES__array_upper(
  const struct bHYPRE_HGMRES__array* array,
  const int32_t ind);

int32_t
bHYPRE_HGMRES__array_length(
  const struct bHYPRE_HGMRES__array* array,
  const int32_t ind);

int32_t
bHYPRE_HGMRES__array_stride(
  const struct bHYPRE_HGMRES__array* array,
  const int32_t ind);

int
bHYPRE_HGMRES__array_isColumnOrder(
  const struct bHYPRE_HGMRES__array* array);

int
bHYPRE_HGMRES__array_isRowOrder(
  const struct bHYPRE_HGMRES__array* array);

struct bHYPRE_HGMRES__array*
bHYPRE_HGMRES__array_slice(
  struct bHYPRE_HGMRES__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_HGMRES__array_copy(
  const struct bHYPRE_HGMRES__array* src,
  struct bHYPRE_HGMRES__array* dest);

struct bHYPRE_HGMRES__array*
bHYPRE_HGMRES__array_ensure(
  struct bHYPRE_HGMRES__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
