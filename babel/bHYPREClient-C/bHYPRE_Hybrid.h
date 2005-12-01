/*
 * File:          bHYPRE_Hybrid.h
 * Symbol:        bHYPRE.Hybrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.Hybrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_Hybrid_h
#define included_bHYPRE_Hybrid_h

/**
 * Symbol "bHYPRE.Hybrid" (version 1.0.0)
 * 
 * Hybrid solver
 * first tries to solve with the specified Krylov solver, preconditioned by
 * diagonal scaling (this combination is the "first solver")
 * If that fails to converge, it will try again with the user-specified
 * preconditioner (this combination is the "second solver").
 * 
 * Specify the preconditioner  by calling SecondSolver's SetPreconditioner
 * method.  If no preconditioner is specified (equivalently, if the
 * preconditioner for SecondSolver is IdentitySolver), the preconditioner for
 * the second try will be one of the following defaults.
 * StructMatrix: SMG.  other matrix types: not implemented
 * 
 * The Hybrid solver's Setup method will call Setup on KrylovSolver, so the
 * user should not call Setup on KrylovSolver.
 * 
 * 
 */
struct bHYPRE_Hybrid__object;
struct bHYPRE_Hybrid__array;
typedef struct bHYPRE_Hybrid__object* bHYPRE_Hybrid;

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
struct bHYPRE_Hybrid__object*
bHYPRE_Hybrid__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_Hybrid
bHYPRE_Hybrid__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_Hybrid
bHYPRE_Hybrid__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_Hybrid_addRef(
  /* in */ bHYPRE_Hybrid self);

void
bHYPRE_Hybrid_deleteRef(
  /* in */ bHYPRE_Hybrid self);

sidl_bool
bHYPRE_Hybrid_isSame(
  /* in */ bHYPRE_Hybrid self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_Hybrid_queryInt(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name);

sidl_bool
bHYPRE_Hybrid_isType(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_Hybrid_getClassInfo(
  /* in */ bHYPRE_Hybrid self);

/**
 * Method:  Create[]
 */
bHYPRE_Hybrid
bHYPRE_Hybrid_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_PreconditionedSolver SecondSolver,
  /* in */ bHYPRE_Operator A);

/**
 * Method:  GetFirstSolver[]
 */
int32_t
bHYPRE_Hybrid_GetFirstSolver(
  /* in */ bHYPRE_Hybrid self,
  /* out */ bHYPRE_PreconditionedSolver* FirstSolver);

/**
 * Method:  GetSecondSolver[]
 */
int32_t
bHYPRE_Hybrid_GetSecondSolver(
  /* in */ bHYPRE_Hybrid self,
  /* out */ bHYPRE_PreconditionedSolver* SecondSolver);

/**
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */
int32_t
bHYPRE_Hybrid_SetCommunicator(
  /* in */ bHYPRE_Hybrid self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Hybrid_SetIntParameter(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* in */ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Hybrid_SetDoubleParameter(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* in */ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Hybrid_SetStringParameter(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* in */ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Hybrid_SetIntArray1Parameter(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Hybrid_SetIntArray2Parameter(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Hybrid_SetDoubleArray1Parameter(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Hybrid_SetDoubleArray2Parameter(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Hybrid_GetIntValue(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* out */ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Hybrid_GetDoubleValue(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* out */ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_Hybrid_Setup(
  /* in */ bHYPRE_Hybrid self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_Hybrid_Apply(
  /* in */ bHYPRE_Hybrid self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_Hybrid_ApplyAdjoint(
  /* in */ bHYPRE_Hybrid self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 * 
 */
int32_t
bHYPRE_Hybrid_SetOperator(
  /* in */ bHYPRE_Hybrid self,
  /* in */ bHYPRE_Operator A);

/**
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */
int32_t
bHYPRE_Hybrid_SetTolerance(
  /* in */ bHYPRE_Hybrid self,
  /* in */ double tolerance);

/**
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */
int32_t
bHYPRE_Hybrid_SetMaxIterations(
  /* in */ bHYPRE_Hybrid self,
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
bHYPRE_Hybrid_SetLogging(
  /* in */ bHYPRE_Hybrid self,
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
bHYPRE_Hybrid_SetPrintLevel(
  /* in */ bHYPRE_Hybrid self,
  /* in */ int32_t level);

/**
 * (Optional) Return the number of iterations taken.
 * 
 */
int32_t
bHYPRE_Hybrid_GetNumIterations(
  /* in */ bHYPRE_Hybrid self,
  /* out */ int32_t* num_iterations);

/**
 * (Optional) Return the norm of the relative residual.
 * 
 */
int32_t
bHYPRE_Hybrid_GetRelResidualNorm(
  /* in */ bHYPRE_Hybrid self,
  /* out */ double* norm);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_Hybrid__object*
bHYPRE_Hybrid__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_Hybrid__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_Hybrid__exec(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_Hybrid__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_Hybrid__getURL(
  /* in */ bHYPRE_Hybrid self);
struct bHYPRE_Hybrid__array*
bHYPRE_Hybrid__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_Hybrid__array*
bHYPRE_Hybrid__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_Hybrid__array*
bHYPRE_Hybrid__array_create1d(int32_t len);

struct bHYPRE_Hybrid__array*
bHYPRE_Hybrid__array_create1dInit(
  int32_t len, 
  bHYPRE_Hybrid* data);

struct bHYPRE_Hybrid__array*
bHYPRE_Hybrid__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_Hybrid__array*
bHYPRE_Hybrid__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_Hybrid__array*
bHYPRE_Hybrid__array_borrow(
  bHYPRE_Hybrid* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_Hybrid__array*
bHYPRE_Hybrid__array_smartCopy(
  struct bHYPRE_Hybrid__array *array);

void
bHYPRE_Hybrid__array_addRef(
  struct bHYPRE_Hybrid__array* array);

void
bHYPRE_Hybrid__array_deleteRef(
  struct bHYPRE_Hybrid__array* array);

bHYPRE_Hybrid
bHYPRE_Hybrid__array_get1(
  const struct bHYPRE_Hybrid__array* array,
  const int32_t i1);

bHYPRE_Hybrid
bHYPRE_Hybrid__array_get2(
  const struct bHYPRE_Hybrid__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_Hybrid
bHYPRE_Hybrid__array_get3(
  const struct bHYPRE_Hybrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_Hybrid
bHYPRE_Hybrid__array_get4(
  const struct bHYPRE_Hybrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_Hybrid
bHYPRE_Hybrid__array_get5(
  const struct bHYPRE_Hybrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_Hybrid
bHYPRE_Hybrid__array_get6(
  const struct bHYPRE_Hybrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_Hybrid
bHYPRE_Hybrid__array_get7(
  const struct bHYPRE_Hybrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_Hybrid
bHYPRE_Hybrid__array_get(
  const struct bHYPRE_Hybrid__array* array,
  const int32_t indices[]);

void
bHYPRE_Hybrid__array_set1(
  struct bHYPRE_Hybrid__array* array,
  const int32_t i1,
  bHYPRE_Hybrid const value);

void
bHYPRE_Hybrid__array_set2(
  struct bHYPRE_Hybrid__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_Hybrid const value);

void
bHYPRE_Hybrid__array_set3(
  struct bHYPRE_Hybrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_Hybrid const value);

void
bHYPRE_Hybrid__array_set4(
  struct bHYPRE_Hybrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_Hybrid const value);

void
bHYPRE_Hybrid__array_set5(
  struct bHYPRE_Hybrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_Hybrid const value);

void
bHYPRE_Hybrid__array_set6(
  struct bHYPRE_Hybrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_Hybrid const value);

void
bHYPRE_Hybrid__array_set7(
  struct bHYPRE_Hybrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_Hybrid const value);

void
bHYPRE_Hybrid__array_set(
  struct bHYPRE_Hybrid__array* array,
  const int32_t indices[],
  bHYPRE_Hybrid const value);

int32_t
bHYPRE_Hybrid__array_dimen(
  const struct bHYPRE_Hybrid__array* array);

int32_t
bHYPRE_Hybrid__array_lower(
  const struct bHYPRE_Hybrid__array* array,
  const int32_t ind);

int32_t
bHYPRE_Hybrid__array_upper(
  const struct bHYPRE_Hybrid__array* array,
  const int32_t ind);

int32_t
bHYPRE_Hybrid__array_length(
  const struct bHYPRE_Hybrid__array* array,
  const int32_t ind);

int32_t
bHYPRE_Hybrid__array_stride(
  const struct bHYPRE_Hybrid__array* array,
  const int32_t ind);

int
bHYPRE_Hybrid__array_isColumnOrder(
  const struct bHYPRE_Hybrid__array* array);

int
bHYPRE_Hybrid__array_isRowOrder(
  const struct bHYPRE_Hybrid__array* array);

struct bHYPRE_Hybrid__array*
bHYPRE_Hybrid__array_slice(
  struct bHYPRE_Hybrid__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_Hybrid__array_copy(
  const struct bHYPRE_Hybrid__array* src,
  struct bHYPRE_Hybrid__array* dest);

struct bHYPRE_Hybrid__array*
bHYPRE_Hybrid__array_ensure(
  struct bHYPRE_Hybrid__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
