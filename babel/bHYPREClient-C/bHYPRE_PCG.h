/*
 * File:          bHYPRE_PCG.h
 * Symbol:        bHYPRE.PCG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.PCG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_PCG_h
#define included_bHYPRE_PCG_h

/**
 * Symbol "bHYPRE.PCG" (version 1.0.0)
 */
struct bHYPRE_PCG__object;
struct bHYPRE_PCG__array;
typedef struct bHYPRE_PCG__object* bHYPRE_PCG;

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
struct bHYPRE_PCG__object*
bHYPRE_PCG__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_PCG
bHYPRE_PCG__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_PCG
bHYPRE_PCG__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_PCG_addRef(
  /* in */ bHYPRE_PCG self);

void
bHYPRE_PCG_deleteRef(
  /* in */ bHYPRE_PCG self);

sidl_bool
bHYPRE_PCG_isSame(
  /* in */ bHYPRE_PCG self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_PCG_queryInt(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name);

sidl_bool
bHYPRE_PCG_isType(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_PCG_getClassInfo(
  /* in */ bHYPRE_PCG self);

/**
 * Method:  Create[]
 */
bHYPRE_PCG
bHYPRE_PCG_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A);

/**
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */
int32_t
bHYPRE_PCG_SetCommunicator(
  /* in */ bHYPRE_PCG self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_PCG_SetIntParameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in */ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_PCG_SetDoubleParameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in */ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_PCG_SetStringParameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in */ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_PCG_SetIntArray1Parameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_PCG_SetIntArray2Parameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_PCG_SetDoubleArray1Parameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_PCG_SetDoubleArray2Parameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_PCG_GetIntValue(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* out */ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_PCG_GetDoubleValue(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* out */ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_PCG_Setup(
  /* in */ bHYPRE_PCG self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_PCG_Apply(
  /* in */ bHYPRE_PCG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Set the operator for the linear system being solved.
 * 
 */
int32_t
bHYPRE_PCG_SetOperator(
  /* in */ bHYPRE_PCG self,
  /* in */ bHYPRE_Operator A);

/**
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */
int32_t
bHYPRE_PCG_SetTolerance(
  /* in */ bHYPRE_PCG self,
  /* in */ double tolerance);

/**
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */
int32_t
bHYPRE_PCG_SetMaxIterations(
  /* in */ bHYPRE_PCG self,
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
bHYPRE_PCG_SetLogging(
  /* in */ bHYPRE_PCG self,
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
bHYPRE_PCG_SetPrintLevel(
  /* in */ bHYPRE_PCG self,
  /* in */ int32_t level);

/**
 * (Optional) Return the number of iterations taken.
 * 
 */
int32_t
bHYPRE_PCG_GetNumIterations(
  /* in */ bHYPRE_PCG self,
  /* out */ int32_t* num_iterations);

/**
 * (Optional) Return the norm of the relative residual.
 * 
 */
int32_t
bHYPRE_PCG_GetRelResidualNorm(
  /* in */ bHYPRE_PCG self,
  /* out */ double* norm);

/**
 * Set the preconditioner.
 * 
 */
int32_t
bHYPRE_PCG_SetPreconditioner(
  /* in */ bHYPRE_PCG self,
  /* in */ bHYPRE_Solver s);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_PCG__object*
bHYPRE_PCG__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_PCG__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_PCG__exec(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_PCG__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_PCG__getURL(
  /* in */ bHYPRE_PCG self);
struct bHYPRE_PCG__array*
bHYPRE_PCG__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_PCG__array*
bHYPRE_PCG__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_PCG__array*
bHYPRE_PCG__array_create1d(int32_t len);

struct bHYPRE_PCG__array*
bHYPRE_PCG__array_create1dInit(
  int32_t len, 
  bHYPRE_PCG* data);

struct bHYPRE_PCG__array*
bHYPRE_PCG__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_PCG__array*
bHYPRE_PCG__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_PCG__array*
bHYPRE_PCG__array_borrow(
  bHYPRE_PCG* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_PCG__array*
bHYPRE_PCG__array_smartCopy(
  struct bHYPRE_PCG__array *array);

void
bHYPRE_PCG__array_addRef(
  struct bHYPRE_PCG__array* array);

void
bHYPRE_PCG__array_deleteRef(
  struct bHYPRE_PCG__array* array);

bHYPRE_PCG
bHYPRE_PCG__array_get1(
  const struct bHYPRE_PCG__array* array,
  const int32_t i1);

bHYPRE_PCG
bHYPRE_PCG__array_get2(
  const struct bHYPRE_PCG__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_PCG
bHYPRE_PCG__array_get3(
  const struct bHYPRE_PCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_PCG
bHYPRE_PCG__array_get4(
  const struct bHYPRE_PCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_PCG
bHYPRE_PCG__array_get5(
  const struct bHYPRE_PCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_PCG
bHYPRE_PCG__array_get6(
  const struct bHYPRE_PCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_PCG
bHYPRE_PCG__array_get7(
  const struct bHYPRE_PCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_PCG
bHYPRE_PCG__array_get(
  const struct bHYPRE_PCG__array* array,
  const int32_t indices[]);

void
bHYPRE_PCG__array_set1(
  struct bHYPRE_PCG__array* array,
  const int32_t i1,
  bHYPRE_PCG const value);

void
bHYPRE_PCG__array_set2(
  struct bHYPRE_PCG__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_PCG const value);

void
bHYPRE_PCG__array_set3(
  struct bHYPRE_PCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_PCG const value);

void
bHYPRE_PCG__array_set4(
  struct bHYPRE_PCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_PCG const value);

void
bHYPRE_PCG__array_set5(
  struct bHYPRE_PCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_PCG const value);

void
bHYPRE_PCG__array_set6(
  struct bHYPRE_PCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_PCG const value);

void
bHYPRE_PCG__array_set7(
  struct bHYPRE_PCG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_PCG const value);

void
bHYPRE_PCG__array_set(
  struct bHYPRE_PCG__array* array,
  const int32_t indices[],
  bHYPRE_PCG const value);

int32_t
bHYPRE_PCG__array_dimen(
  const struct bHYPRE_PCG__array* array);

int32_t
bHYPRE_PCG__array_lower(
  const struct bHYPRE_PCG__array* array,
  const int32_t ind);

int32_t
bHYPRE_PCG__array_upper(
  const struct bHYPRE_PCG__array* array,
  const int32_t ind);

int32_t
bHYPRE_PCG__array_length(
  const struct bHYPRE_PCG__array* array,
  const int32_t ind);

int32_t
bHYPRE_PCG__array_stride(
  const struct bHYPRE_PCG__array* array,
  const int32_t ind);

int
bHYPRE_PCG__array_isColumnOrder(
  const struct bHYPRE_PCG__array* array);

int
bHYPRE_PCG__array_isRowOrder(
  const struct bHYPRE_PCG__array* array);

struct bHYPRE_PCG__array*
bHYPRE_PCG__array_slice(
  struct bHYPRE_PCG__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_PCG__array_copy(
  const struct bHYPRE_PCG__array* src,
  struct bHYPRE_PCG__array* dest);

struct bHYPRE_PCG__array*
bHYPRE_PCG__array_ensure(
  struct bHYPRE_PCG__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
