/*
 * File:          bHYPRE_CGNR.h
 * Symbol:        bHYPRE.CGNR-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Client-side glue code for bHYPRE.CGNR
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_CGNR_h
#define included_bHYPRE_CGNR_h

/**
 * Symbol "bHYPRE.CGNR" (version 1.0.0)
 * 
 * Objects of this type can be cast to PreconditionedSolver objects
 * using the {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 * CGNR solver calls Babel-interface functions
 * 
 * 
 */
struct bHYPRE_CGNR__object;
struct bHYPRE_CGNR__array;
typedef struct bHYPRE_CGNR__object* bHYPRE_CGNR;

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
struct bHYPRE_CGNR__object*
bHYPRE_CGNR__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_CGNR
bHYPRE_CGNR__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_CGNR
bHYPRE_CGNR__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_CGNR_addRef(
  /* in */ bHYPRE_CGNR self);

void
bHYPRE_CGNR_deleteRef(
  /* in */ bHYPRE_CGNR self);

sidl_bool
bHYPRE_CGNR_isSame(
  /* in */ bHYPRE_CGNR self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_CGNR_queryInt(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name);

sidl_bool
bHYPRE_CGNR_isType(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_CGNR_getClassInfo(
  /* in */ bHYPRE_CGNR self);

/**
 * Method:  Create[]
 */
bHYPRE_CGNR
bHYPRE_CGNR_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A);

/**
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */
int32_t
bHYPRE_CGNR_SetCommunicator(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_CGNR_SetIntParameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in */ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_CGNR_SetDoubleParameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in */ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_CGNR_SetStringParameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in */ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_CGNR_SetIntArray1Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_CGNR_SetIntArray2Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_CGNR_SetDoubleArray1Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_CGNR_SetDoubleArray2Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_CGNR_GetIntValue(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* out */ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_CGNR_GetDoubleValue(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* out */ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_CGNR_Setup(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_CGNR_Apply(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_CGNR_ApplyAdjoint(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 * 
 */
int32_t
bHYPRE_CGNR_SetOperator(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Operator A);

/**
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */
int32_t
bHYPRE_CGNR_SetTolerance(
  /* in */ bHYPRE_CGNR self,
  /* in */ double tolerance);

/**
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */
int32_t
bHYPRE_CGNR_SetMaxIterations(
  /* in */ bHYPRE_CGNR self,
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
bHYPRE_CGNR_SetLogging(
  /* in */ bHYPRE_CGNR self,
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
bHYPRE_CGNR_SetPrintLevel(
  /* in */ bHYPRE_CGNR self,
  /* in */ int32_t level);

/**
 * (Optional) Return the number of iterations taken.
 * 
 */
int32_t
bHYPRE_CGNR_GetNumIterations(
  /* in */ bHYPRE_CGNR self,
  /* out */ int32_t* num_iterations);

/**
 * (Optional) Return the norm of the relative residual.
 * 
 */
int32_t
bHYPRE_CGNR_GetRelResidualNorm(
  /* in */ bHYPRE_CGNR self,
  /* out */ double* norm);

/**
 * Set the preconditioner.
 * 
 */
int32_t
bHYPRE_CGNR_SetPreconditioner(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Solver s);

/**
 * Method:  GetPreconditioner[]
 */
int32_t
bHYPRE_CGNR_GetPreconditioner(
  /* in */ bHYPRE_CGNR self,
  /* out */ bHYPRE_Solver* s);

/**
 * Method:  Clone[]
 */
int32_t
bHYPRE_CGNR_Clone(
  /* in */ bHYPRE_CGNR self,
  /* out */ bHYPRE_PreconditionedSolver* x);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_CGNR__object*
bHYPRE_CGNR__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_CGNR__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_CGNR__exec(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_CGNR__getURL(
  /* in */ bHYPRE_CGNR self);
struct bHYPRE_CGNR__array*
bHYPRE_CGNR__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_CGNR__array*
bHYPRE_CGNR__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_CGNR__array*
bHYPRE_CGNR__array_create1d(int32_t len);

struct bHYPRE_CGNR__array*
bHYPRE_CGNR__array_create1dInit(
  int32_t len, 
  bHYPRE_CGNR* data);

struct bHYPRE_CGNR__array*
bHYPRE_CGNR__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_CGNR__array*
bHYPRE_CGNR__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_CGNR__array*
bHYPRE_CGNR__array_borrow(
  bHYPRE_CGNR* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_CGNR__array*
bHYPRE_CGNR__array_smartCopy(
  struct bHYPRE_CGNR__array *array);

void
bHYPRE_CGNR__array_addRef(
  struct bHYPRE_CGNR__array* array);

void
bHYPRE_CGNR__array_deleteRef(
  struct bHYPRE_CGNR__array* array);

bHYPRE_CGNR
bHYPRE_CGNR__array_get1(
  const struct bHYPRE_CGNR__array* array,
  const int32_t i1);

bHYPRE_CGNR
bHYPRE_CGNR__array_get2(
  const struct bHYPRE_CGNR__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_CGNR
bHYPRE_CGNR__array_get3(
  const struct bHYPRE_CGNR__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_CGNR
bHYPRE_CGNR__array_get4(
  const struct bHYPRE_CGNR__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_CGNR
bHYPRE_CGNR__array_get5(
  const struct bHYPRE_CGNR__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_CGNR
bHYPRE_CGNR__array_get6(
  const struct bHYPRE_CGNR__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_CGNR
bHYPRE_CGNR__array_get7(
  const struct bHYPRE_CGNR__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_CGNR
bHYPRE_CGNR__array_get(
  const struct bHYPRE_CGNR__array* array,
  const int32_t indices[]);

void
bHYPRE_CGNR__array_set1(
  struct bHYPRE_CGNR__array* array,
  const int32_t i1,
  bHYPRE_CGNR const value);

void
bHYPRE_CGNR__array_set2(
  struct bHYPRE_CGNR__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_CGNR const value);

void
bHYPRE_CGNR__array_set3(
  struct bHYPRE_CGNR__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_CGNR const value);

void
bHYPRE_CGNR__array_set4(
  struct bHYPRE_CGNR__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_CGNR const value);

void
bHYPRE_CGNR__array_set5(
  struct bHYPRE_CGNR__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_CGNR const value);

void
bHYPRE_CGNR__array_set6(
  struct bHYPRE_CGNR__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_CGNR const value);

void
bHYPRE_CGNR__array_set7(
  struct bHYPRE_CGNR__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_CGNR const value);

void
bHYPRE_CGNR__array_set(
  struct bHYPRE_CGNR__array* array,
  const int32_t indices[],
  bHYPRE_CGNR const value);

int32_t
bHYPRE_CGNR__array_dimen(
  const struct bHYPRE_CGNR__array* array);

int32_t
bHYPRE_CGNR__array_lower(
  const struct bHYPRE_CGNR__array* array,
  const int32_t ind);

int32_t
bHYPRE_CGNR__array_upper(
  const struct bHYPRE_CGNR__array* array,
  const int32_t ind);

int32_t
bHYPRE_CGNR__array_length(
  const struct bHYPRE_CGNR__array* array,
  const int32_t ind);

int32_t
bHYPRE_CGNR__array_stride(
  const struct bHYPRE_CGNR__array* array,
  const int32_t ind);

int
bHYPRE_CGNR__array_isColumnOrder(
  const struct bHYPRE_CGNR__array* array);

int
bHYPRE_CGNR__array_isRowOrder(
  const struct bHYPRE_CGNR__array* array);

struct bHYPRE_CGNR__array*
bHYPRE_CGNR__array_slice(
  struct bHYPRE_CGNR__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_CGNR__array_copy(
  const struct bHYPRE_CGNR__array* src,
  struct bHYPRE_CGNR__array* dest);

struct bHYPRE_CGNR__array*
bHYPRE_CGNR__array_ensure(
  struct bHYPRE_CGNR__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
