/*
 * File:          bHYPRE_SStructSplit.h
 * Symbol:        bHYPRE.SStructSplit-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.SStructSplit
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_SStructSplit_h
#define included_bHYPRE_SStructSplit_h

/**
 * Symbol "bHYPRE.SStructSplit" (version 1.0.0)
 * 
 * 
 * Documentation goes here.
 * 
 * The SStructSplit solver requires a SStruct matrix.
 * 
 * 
 */
struct bHYPRE_SStructSplit__object;
struct bHYPRE_SStructSplit__array;
typedef struct bHYPRE_SStructSplit__object* bHYPRE_SStructSplit;

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
struct bHYPRE_SStructSplit__object*
bHYPRE_SStructSplit__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_SStructSplit
bHYPRE_SStructSplit__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_SStructSplit
bHYPRE_SStructSplit__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_SStructSplit_addRef(
  /* in */ bHYPRE_SStructSplit self);

void
bHYPRE_SStructSplit_deleteRef(
  /* in */ bHYPRE_SStructSplit self);

sidl_bool
bHYPRE_SStructSplit_isSame(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_SStructSplit_queryInt(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name);

sidl_bool
bHYPRE_SStructSplit_isType(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_SStructSplit_getClassInfo(
  /* in */ bHYPRE_SStructSplit self);

/**
 * Method:  Create[]
 */
bHYPRE_SStructSplit
bHYPRE_SStructSplit_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A);

/**
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */
int32_t
bHYPRE_SStructSplit_SetCommunicator(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructSplit_SetIntParameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructSplit_SetDoubleParameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructSplit_SetStringParameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructSplit_SetIntArray1Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructSplit_SetIntArray2Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructSplit_SetDoubleArray1Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructSplit_SetDoubleArray2Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructSplit_GetIntValue(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* out */ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructSplit_GetDoubleValue(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* out */ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_SStructSplit_Setup(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_SStructSplit_Apply(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_SStructSplit_ApplyAdjoint(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 * 
 */
int32_t
bHYPRE_SStructSplit_SetOperator(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_Operator A);

/**
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */
int32_t
bHYPRE_SStructSplit_SetTolerance(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ double tolerance);

/**
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */
int32_t
bHYPRE_SStructSplit_SetMaxIterations(
  /* in */ bHYPRE_SStructSplit self,
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
bHYPRE_SStructSplit_SetLogging(
  /* in */ bHYPRE_SStructSplit self,
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
bHYPRE_SStructSplit_SetPrintLevel(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ int32_t level);

/**
 * (Optional) Return the number of iterations taken.
 * 
 */
int32_t
bHYPRE_SStructSplit_GetNumIterations(
  /* in */ bHYPRE_SStructSplit self,
  /* out */ int32_t* num_iterations);

/**
 * (Optional) Return the norm of the relative residual.
 * 
 */
int32_t
bHYPRE_SStructSplit_GetRelResidualNorm(
  /* in */ bHYPRE_SStructSplit self,
  /* out */ double* norm);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructSplit__object*
bHYPRE_SStructSplit__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructSplit__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_SStructSplit__exec(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_SStructSplit__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_SStructSplit__getURL(
  /* in */ bHYPRE_SStructSplit self);
struct bHYPRE_SStructSplit__array*
bHYPRE_SStructSplit__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructSplit__array*
bHYPRE_SStructSplit__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructSplit__array*
bHYPRE_SStructSplit__array_create1d(int32_t len);

struct bHYPRE_SStructSplit__array*
bHYPRE_SStructSplit__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructSplit* data);

struct bHYPRE_SStructSplit__array*
bHYPRE_SStructSplit__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructSplit__array*
bHYPRE_SStructSplit__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructSplit__array*
bHYPRE_SStructSplit__array_borrow(
  bHYPRE_SStructSplit* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_SStructSplit__array*
bHYPRE_SStructSplit__array_smartCopy(
  struct bHYPRE_SStructSplit__array *array);

void
bHYPRE_SStructSplit__array_addRef(
  struct bHYPRE_SStructSplit__array* array);

void
bHYPRE_SStructSplit__array_deleteRef(
  struct bHYPRE_SStructSplit__array* array);

bHYPRE_SStructSplit
bHYPRE_SStructSplit__array_get1(
  const struct bHYPRE_SStructSplit__array* array,
  const int32_t i1);

bHYPRE_SStructSplit
bHYPRE_SStructSplit__array_get2(
  const struct bHYPRE_SStructSplit__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_SStructSplit
bHYPRE_SStructSplit__array_get3(
  const struct bHYPRE_SStructSplit__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_SStructSplit
bHYPRE_SStructSplit__array_get4(
  const struct bHYPRE_SStructSplit__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_SStructSplit
bHYPRE_SStructSplit__array_get5(
  const struct bHYPRE_SStructSplit__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_SStructSplit
bHYPRE_SStructSplit__array_get6(
  const struct bHYPRE_SStructSplit__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_SStructSplit
bHYPRE_SStructSplit__array_get7(
  const struct bHYPRE_SStructSplit__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_SStructSplit
bHYPRE_SStructSplit__array_get(
  const struct bHYPRE_SStructSplit__array* array,
  const int32_t indices[]);

void
bHYPRE_SStructSplit__array_set1(
  struct bHYPRE_SStructSplit__array* array,
  const int32_t i1,
  bHYPRE_SStructSplit const value);

void
bHYPRE_SStructSplit__array_set2(
  struct bHYPRE_SStructSplit__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructSplit const value);

void
bHYPRE_SStructSplit__array_set3(
  struct bHYPRE_SStructSplit__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructSplit const value);

void
bHYPRE_SStructSplit__array_set4(
  struct bHYPRE_SStructSplit__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructSplit const value);

void
bHYPRE_SStructSplit__array_set5(
  struct bHYPRE_SStructSplit__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructSplit const value);

void
bHYPRE_SStructSplit__array_set6(
  struct bHYPRE_SStructSplit__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructSplit const value);

void
bHYPRE_SStructSplit__array_set7(
  struct bHYPRE_SStructSplit__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructSplit const value);

void
bHYPRE_SStructSplit__array_set(
  struct bHYPRE_SStructSplit__array* array,
  const int32_t indices[],
  bHYPRE_SStructSplit const value);

int32_t
bHYPRE_SStructSplit__array_dimen(
  const struct bHYPRE_SStructSplit__array* array);

int32_t
bHYPRE_SStructSplit__array_lower(
  const struct bHYPRE_SStructSplit__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructSplit__array_upper(
  const struct bHYPRE_SStructSplit__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructSplit__array_length(
  const struct bHYPRE_SStructSplit__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructSplit__array_stride(
  const struct bHYPRE_SStructSplit__array* array,
  const int32_t ind);

int
bHYPRE_SStructSplit__array_isColumnOrder(
  const struct bHYPRE_SStructSplit__array* array);

int
bHYPRE_SStructSplit__array_isRowOrder(
  const struct bHYPRE_SStructSplit__array* array);

struct bHYPRE_SStructSplit__array*
bHYPRE_SStructSplit__array_slice(
  struct bHYPRE_SStructSplit__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_SStructSplit__array_copy(
  const struct bHYPRE_SStructSplit__array* src,
  struct bHYPRE_SStructSplit__array* dest);

struct bHYPRE_SStructSplit__array*
bHYPRE_SStructSplit__array_ensure(
  struct bHYPRE_SStructSplit__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
