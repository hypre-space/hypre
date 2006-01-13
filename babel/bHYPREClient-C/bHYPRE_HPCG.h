/*
 * File:          bHYPRE_HPCG.h
 * Symbol:        bHYPRE.HPCG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Client-side glue code for bHYPRE.HPCG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_HPCG_h
#define included_bHYPRE_HPCG_h

/**
 * Symbol "bHYPRE.HPCG" (version 1.0.0)
 * 
 * Objects of this type can be cast to PreconditionedSolver objects
 * using the {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 * The regular PCG solver calls Babel-interface matrix and vector functions.
 * The HPCG solver calls HYPRE interface functions.
 * The regular solver will work with any consistent matrix, vector, and
 * preconditioner classes.  The HPCG solver will work with the more common
 * combinations.
 * 
 * The HPCG solver checks whether the matrix, vectors, and preconditioner
 * are of known types, and will not work with any other types.
 * Presently, the recognized data types are:
 * matrix, vector: IJParCSRMatrix, IJParCSRVector
 * matrix, vector: StructMatrix, StructVector
 * preconditioner: BoomerAMG, ParaSails, ParCSRDiagScale, IdentitySolver
 * preconditioner: StructSMG, StructPFMG
 * 
 * 
 * 
 */
struct bHYPRE_HPCG__object;
struct bHYPRE_HPCG__array;
typedef struct bHYPRE_HPCG__object* bHYPRE_HPCG;

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
struct bHYPRE_HPCG__object*
bHYPRE_HPCG__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_HPCG
bHYPRE_HPCG__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_HPCG
bHYPRE_HPCG__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_HPCG_addRef(
  /* in */ bHYPRE_HPCG self);

void
bHYPRE_HPCG_deleteRef(
  /* in */ bHYPRE_HPCG self);

sidl_bool
bHYPRE_HPCG_isSame(
  /* in */ bHYPRE_HPCG self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_HPCG_queryInt(
  /* in */ bHYPRE_HPCG self,
  /* in */ const char* name);

sidl_bool
bHYPRE_HPCG_isType(
  /* in */ bHYPRE_HPCG self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_HPCG_getClassInfo(
  /* in */ bHYPRE_HPCG self);

/**
 * Method:  Create[]
 */
bHYPRE_HPCG
bHYPRE_HPCG_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */
int32_t
bHYPRE_HPCG_SetCommunicator(
  /* in */ bHYPRE_HPCG self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HPCG_SetIntParameter(
  /* in */ bHYPRE_HPCG self,
  /* in */ const char* name,
  /* in */ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HPCG_SetDoubleParameter(
  /* in */ bHYPRE_HPCG self,
  /* in */ const char* name,
  /* in */ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HPCG_SetStringParameter(
  /* in */ bHYPRE_HPCG self,
  /* in */ const char* name,
  /* in */ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HPCG_SetIntArray1Parameter(
  /* in */ bHYPRE_HPCG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HPCG_SetIntArray2Parameter(
  /* in */ bHYPRE_HPCG self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HPCG_SetDoubleArray1Parameter(
  /* in */ bHYPRE_HPCG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HPCG_SetDoubleArray2Parameter(
  /* in */ bHYPRE_HPCG self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HPCG_GetIntValue(
  /* in */ bHYPRE_HPCG self,
  /* in */ const char* name,
  /* out */ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_HPCG_GetDoubleValue(
  /* in */ bHYPRE_HPCG self,
  /* in */ const char* name,
  /* out */ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_HPCG_Setup(
  /* in */ bHYPRE_HPCG self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_HPCG_Apply(
  /* in */ bHYPRE_HPCG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_HPCG_ApplyAdjoint(
  /* in */ bHYPRE_HPCG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 * 
 */
int32_t
bHYPRE_HPCG_SetOperator(
  /* in */ bHYPRE_HPCG self,
  /* in */ bHYPRE_Operator A);

/**
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */
int32_t
bHYPRE_HPCG_SetTolerance(
  /* in */ bHYPRE_HPCG self,
  /* in */ double tolerance);

/**
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */
int32_t
bHYPRE_HPCG_SetMaxIterations(
  /* in */ bHYPRE_HPCG self,
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
bHYPRE_HPCG_SetLogging(
  /* in */ bHYPRE_HPCG self,
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
bHYPRE_HPCG_SetPrintLevel(
  /* in */ bHYPRE_HPCG self,
  /* in */ int32_t level);

/**
 * (Optional) Return the number of iterations taken.
 * 
 */
int32_t
bHYPRE_HPCG_GetNumIterations(
  /* in */ bHYPRE_HPCG self,
  /* out */ int32_t* num_iterations);

/**
 * (Optional) Return the norm of the relative residual.
 * 
 */
int32_t
bHYPRE_HPCG_GetRelResidualNorm(
  /* in */ bHYPRE_HPCG self,
  /* out */ double* norm);

/**
 * Set the preconditioner.
 * 
 */
int32_t
bHYPRE_HPCG_SetPreconditioner(
  /* in */ bHYPRE_HPCG self,
  /* in */ bHYPRE_Solver s);

/**
 * Method:  GetPreconditioner[]
 */
int32_t
bHYPRE_HPCG_GetPreconditioner(
  /* in */ bHYPRE_HPCG self,
  /* out */ bHYPRE_Solver* s);

/**
 * Method:  Clone[]
 */
int32_t
bHYPRE_HPCG_Clone(
  /* in */ bHYPRE_HPCG self,
  /* out */ bHYPRE_PreconditionedSolver* x);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_HPCG__object*
bHYPRE_HPCG__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_HPCG__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_HPCG__exec(
  /* in */ bHYPRE_HPCG self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_HPCG__getURL(
  /* in */ bHYPRE_HPCG self);
struct bHYPRE_HPCG__array*
bHYPRE_HPCG__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_HPCG__array*
bHYPRE_HPCG__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_HPCG__array*
bHYPRE_HPCG__array_create1d(int32_t len);

struct bHYPRE_HPCG__array*
bHYPRE_HPCG__array_create1dInit(
  int32_t len, 
  bHYPRE_HPCG* data);

struct bHYPRE_HPCG__array*
bHYPRE_HPCG__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_HPCG__array*
bHYPRE_HPCG__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_HPCG__array*
bHYPRE_HPCG__array_borrow(
  bHYPRE_HPCG* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_HPCG__array*
bHYPRE_HPCG__array_smartCopy(
  struct bHYPRE_HPCG__array *array);

void
bHYPRE_HPCG__array_addRef(
  struct bHYPRE_HPCG__array* array);

void
bHYPRE_HPCG__array_deleteRef(
  struct bHYPRE_HPCG__array* array);

bHYPRE_HPCG
bHYPRE_HPCG__array_get1(
  const struct bHYPRE_HPCG__array* array,
  const int32_t i1);

bHYPRE_HPCG
bHYPRE_HPCG__array_get2(
  const struct bHYPRE_HPCG__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_HPCG
bHYPRE_HPCG__array_get3(
  const struct bHYPRE_HPCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_HPCG
bHYPRE_HPCG__array_get4(
  const struct bHYPRE_HPCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_HPCG
bHYPRE_HPCG__array_get5(
  const struct bHYPRE_HPCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_HPCG
bHYPRE_HPCG__array_get6(
  const struct bHYPRE_HPCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_HPCG
bHYPRE_HPCG__array_get7(
  const struct bHYPRE_HPCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_HPCG
bHYPRE_HPCG__array_get(
  const struct bHYPRE_HPCG__array* array,
  const int32_t indices[]);

void
bHYPRE_HPCG__array_set1(
  struct bHYPRE_HPCG__array* array,
  const int32_t i1,
  bHYPRE_HPCG const value);

void
bHYPRE_HPCG__array_set2(
  struct bHYPRE_HPCG__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_HPCG const value);

void
bHYPRE_HPCG__array_set3(
  struct bHYPRE_HPCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_HPCG const value);

void
bHYPRE_HPCG__array_set4(
  struct bHYPRE_HPCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_HPCG const value);

void
bHYPRE_HPCG__array_set5(
  struct bHYPRE_HPCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_HPCG const value);

void
bHYPRE_HPCG__array_set6(
  struct bHYPRE_HPCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_HPCG const value);

void
bHYPRE_HPCG__array_set7(
  struct bHYPRE_HPCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_HPCG const value);

void
bHYPRE_HPCG__array_set(
  struct bHYPRE_HPCG__array* array,
  const int32_t indices[],
  bHYPRE_HPCG const value);

int32_t
bHYPRE_HPCG__array_dimen(
  const struct bHYPRE_HPCG__array* array);

int32_t
bHYPRE_HPCG__array_lower(
  const struct bHYPRE_HPCG__array* array,
  const int32_t ind);

int32_t
bHYPRE_HPCG__array_upper(
  const struct bHYPRE_HPCG__array* array,
  const int32_t ind);

int32_t
bHYPRE_HPCG__array_length(
  const struct bHYPRE_HPCG__array* array,
  const int32_t ind);

int32_t
bHYPRE_HPCG__array_stride(
  const struct bHYPRE_HPCG__array* array,
  const int32_t ind);

int
bHYPRE_HPCG__array_isColumnOrder(
  const struct bHYPRE_HPCG__array* array);

int
bHYPRE_HPCG__array_isRowOrder(
  const struct bHYPRE_HPCG__array* array);

struct bHYPRE_HPCG__array*
bHYPRE_HPCG__array_slice(
  struct bHYPRE_HPCG__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_HPCG__array_copy(
  const struct bHYPRE_HPCG__array* src,
  struct bHYPRE_HPCG__array* dest);

struct bHYPRE_HPCG__array*
bHYPRE_HPCG__array_ensure(
  struct bHYPRE_HPCG__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
