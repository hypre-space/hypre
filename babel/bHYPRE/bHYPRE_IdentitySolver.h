/*
 * File:          bHYPRE_IdentitySolver.h
 * Symbol:        bHYPRE.IdentitySolver-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.IdentitySolver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_IdentitySolver_h
#define included_bHYPRE_IdentitySolver_h

/**
 * Symbol "bHYPRE.IdentitySolver" (version 1.0.0)
 * 
 * Identity solver, just solves an identity matrix, for when you don't really
 * want a preconditioner
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 */
struct bHYPRE_IdentitySolver__object;
struct bHYPRE_IdentitySolver__array;
typedef struct bHYPRE_IdentitySolver__object* bHYPRE_IdentitySolver;

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
struct bHYPRE_IdentitySolver__object*
bHYPRE_IdentitySolver__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_IdentitySolver
bHYPRE_IdentitySolver__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_IdentitySolver
bHYPRE_IdentitySolver__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_IdentitySolver_addRef(
  /* in */ bHYPRE_IdentitySolver self);

void
bHYPRE_IdentitySolver_deleteRef(
  /* in */ bHYPRE_IdentitySolver self);

sidl_bool
bHYPRE_IdentitySolver_isSame(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_IdentitySolver_queryInt(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name);

sidl_bool
bHYPRE_IdentitySolver_isType(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_IdentitySolver_getClassInfo(
  /* in */ bHYPRE_IdentitySolver self);

/**
 * Method:  Create[]
 */
bHYPRE_IdentitySolver
bHYPRE_IdentitySolver_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */
int32_t
bHYPRE_IdentitySolver_SetCommunicator(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IdentitySolver_SetIntParameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in */ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IdentitySolver_SetDoubleParameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in */ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IdentitySolver_SetStringParameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in */ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IdentitySolver_SetIntArray1Parameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IdentitySolver_SetIntArray2Parameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IdentitySolver_SetDoubleArray1Parameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IdentitySolver_SetDoubleArray2Parameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IdentitySolver_GetIntValue(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* out */ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IdentitySolver_GetDoubleValue(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* out */ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_IdentitySolver_Setup(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_IdentitySolver_Apply(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_IdentitySolver_ApplyAdjoint(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 * 
 */
int32_t
bHYPRE_IdentitySolver_SetOperator(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ bHYPRE_Operator A);

/**
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */
int32_t
bHYPRE_IdentitySolver_SetTolerance(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ double tolerance);

/**
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */
int32_t
bHYPRE_IdentitySolver_SetMaxIterations(
  /* in */ bHYPRE_IdentitySolver self,
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
bHYPRE_IdentitySolver_SetLogging(
  /* in */ bHYPRE_IdentitySolver self,
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
bHYPRE_IdentitySolver_SetPrintLevel(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ int32_t level);

/**
 * (Optional) Return the number of iterations taken.
 * 
 */
int32_t
bHYPRE_IdentitySolver_GetNumIterations(
  /* in */ bHYPRE_IdentitySolver self,
  /* out */ int32_t* num_iterations);

/**
 * (Optional) Return the norm of the relative residual.
 * 
 */
int32_t
bHYPRE_IdentitySolver_GetRelResidualNorm(
  /* in */ bHYPRE_IdentitySolver self,
  /* out */ double* norm);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_IdentitySolver__object*
bHYPRE_IdentitySolver__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_IdentitySolver__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_IdentitySolver__exec(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_IdentitySolver__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_IdentitySolver__getURL(
  /* in */ bHYPRE_IdentitySolver self);
struct bHYPRE_IdentitySolver__array*
bHYPRE_IdentitySolver__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_IdentitySolver__array*
bHYPRE_IdentitySolver__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_IdentitySolver__array*
bHYPRE_IdentitySolver__array_create1d(int32_t len);

struct bHYPRE_IdentitySolver__array*
bHYPRE_IdentitySolver__array_create1dInit(
  int32_t len, 
  bHYPRE_IdentitySolver* data);

struct bHYPRE_IdentitySolver__array*
bHYPRE_IdentitySolver__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_IdentitySolver__array*
bHYPRE_IdentitySolver__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_IdentitySolver__array*
bHYPRE_IdentitySolver__array_borrow(
  bHYPRE_IdentitySolver* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_IdentitySolver__array*
bHYPRE_IdentitySolver__array_smartCopy(
  struct bHYPRE_IdentitySolver__array *array);

void
bHYPRE_IdentitySolver__array_addRef(
  struct bHYPRE_IdentitySolver__array* array);

void
bHYPRE_IdentitySolver__array_deleteRef(
  struct bHYPRE_IdentitySolver__array* array);

bHYPRE_IdentitySolver
bHYPRE_IdentitySolver__array_get1(
  const struct bHYPRE_IdentitySolver__array* array,
  const int32_t i1);

bHYPRE_IdentitySolver
bHYPRE_IdentitySolver__array_get2(
  const struct bHYPRE_IdentitySolver__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_IdentitySolver
bHYPRE_IdentitySolver__array_get3(
  const struct bHYPRE_IdentitySolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_IdentitySolver
bHYPRE_IdentitySolver__array_get4(
  const struct bHYPRE_IdentitySolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_IdentitySolver
bHYPRE_IdentitySolver__array_get5(
  const struct bHYPRE_IdentitySolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_IdentitySolver
bHYPRE_IdentitySolver__array_get6(
  const struct bHYPRE_IdentitySolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_IdentitySolver
bHYPRE_IdentitySolver__array_get7(
  const struct bHYPRE_IdentitySolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_IdentitySolver
bHYPRE_IdentitySolver__array_get(
  const struct bHYPRE_IdentitySolver__array* array,
  const int32_t indices[]);

void
bHYPRE_IdentitySolver__array_set1(
  struct bHYPRE_IdentitySolver__array* array,
  const int32_t i1,
  bHYPRE_IdentitySolver const value);

void
bHYPRE_IdentitySolver__array_set2(
  struct bHYPRE_IdentitySolver__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_IdentitySolver const value);

void
bHYPRE_IdentitySolver__array_set3(
  struct bHYPRE_IdentitySolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_IdentitySolver const value);

void
bHYPRE_IdentitySolver__array_set4(
  struct bHYPRE_IdentitySolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_IdentitySolver const value);

void
bHYPRE_IdentitySolver__array_set5(
  struct bHYPRE_IdentitySolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_IdentitySolver const value);

void
bHYPRE_IdentitySolver__array_set6(
  struct bHYPRE_IdentitySolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_IdentitySolver const value);

void
bHYPRE_IdentitySolver__array_set7(
  struct bHYPRE_IdentitySolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_IdentitySolver const value);

void
bHYPRE_IdentitySolver__array_set(
  struct bHYPRE_IdentitySolver__array* array,
  const int32_t indices[],
  bHYPRE_IdentitySolver const value);

int32_t
bHYPRE_IdentitySolver__array_dimen(
  const struct bHYPRE_IdentitySolver__array* array);

int32_t
bHYPRE_IdentitySolver__array_lower(
  const struct bHYPRE_IdentitySolver__array* array,
  const int32_t ind);

int32_t
bHYPRE_IdentitySolver__array_upper(
  const struct bHYPRE_IdentitySolver__array* array,
  const int32_t ind);

int32_t
bHYPRE_IdentitySolver__array_length(
  const struct bHYPRE_IdentitySolver__array* array,
  const int32_t ind);

int32_t
bHYPRE_IdentitySolver__array_stride(
  const struct bHYPRE_IdentitySolver__array* array,
  const int32_t ind);

int
bHYPRE_IdentitySolver__array_isColumnOrder(
  const struct bHYPRE_IdentitySolver__array* array);

int
bHYPRE_IdentitySolver__array_isRowOrder(
  const struct bHYPRE_IdentitySolver__array* array);

struct bHYPRE_IdentitySolver__array*
bHYPRE_IdentitySolver__array_slice(
  struct bHYPRE_IdentitySolver__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_IdentitySolver__array_copy(
  const struct bHYPRE_IdentitySolver__array* src,
  struct bHYPRE_IdentitySolver__array* dest);

struct bHYPRE_IdentitySolver__array*
bHYPRE_IdentitySolver__array_ensure(
  struct bHYPRE_IdentitySolver__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
