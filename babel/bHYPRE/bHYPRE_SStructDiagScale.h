/*
 * File:          bHYPRE_SStructDiagScale.h
 * Symbol:        bHYPRE.SStructDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Client-side glue code for bHYPRE.SStructDiagScale
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_SStructDiagScale_h
#define included_bHYPRE_SStructDiagScale_h

/**
 * Symbol "bHYPRE.SStructDiagScale" (version 1.0.0)
 */
struct bHYPRE_SStructDiagScale__object;
struct bHYPRE_SStructDiagScale__array;
typedef struct bHYPRE_SStructDiagScale__object* bHYPRE_SStructDiagScale;

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
struct bHYPRE_SStructDiagScale__object*
bHYPRE_SStructDiagScale__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_SStructDiagScale
bHYPRE_SStructDiagScale__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_SStructDiagScale
bHYPRE_SStructDiagScale__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_SStructDiagScale_addRef(
  /* in */ bHYPRE_SStructDiagScale self);

void
bHYPRE_SStructDiagScale_deleteRef(
  /* in */ bHYPRE_SStructDiagScale self);

sidl_bool
bHYPRE_SStructDiagScale_isSame(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_SStructDiagScale_queryInt(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name);

sidl_bool
bHYPRE_SStructDiagScale_isType(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_SStructDiagScale_getClassInfo(
  /* in */ bHYPRE_SStructDiagScale self);

/**
 * Method:  Create[]
 */
bHYPRE_SStructDiagScale
bHYPRE_SStructDiagScale_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A);

/**
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */
int32_t
bHYPRE_SStructDiagScale_SetCommunicator(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructDiagScale_SetIntParameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in */ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructDiagScale_SetDoubleParameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in */ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructDiagScale_SetStringParameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in */ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructDiagScale_SetIntArray1Parameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructDiagScale_SetIntArray2Parameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructDiagScale_SetDoubleArray1Parameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructDiagScale_SetDoubleArray2Parameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructDiagScale_GetIntValue(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* out */ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructDiagScale_GetDoubleValue(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* out */ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_SStructDiagScale_Setup(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_SStructDiagScale_Apply(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_SStructDiagScale_ApplyAdjoint(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 * 
 */
int32_t
bHYPRE_SStructDiagScale_SetOperator(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ bHYPRE_Operator A);

/**
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */
int32_t
bHYPRE_SStructDiagScale_SetTolerance(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ double tolerance);

/**
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */
int32_t
bHYPRE_SStructDiagScale_SetMaxIterations(
  /* in */ bHYPRE_SStructDiagScale self,
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
bHYPRE_SStructDiagScale_SetLogging(
  /* in */ bHYPRE_SStructDiagScale self,
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
bHYPRE_SStructDiagScale_SetPrintLevel(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ int32_t level);

/**
 * (Optional) Return the number of iterations taken.
 * 
 */
int32_t
bHYPRE_SStructDiagScale_GetNumIterations(
  /* in */ bHYPRE_SStructDiagScale self,
  /* out */ int32_t* num_iterations);

/**
 * (Optional) Return the norm of the relative residual.
 * 
 */
int32_t
bHYPRE_SStructDiagScale_GetRelResidualNorm(
  /* in */ bHYPRE_SStructDiagScale self,
  /* out */ double* norm);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructDiagScale__object*
bHYPRE_SStructDiagScale__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructDiagScale__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_SStructDiagScale__exec(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_SStructDiagScale__getURL(
  /* in */ bHYPRE_SStructDiagScale self);
struct bHYPRE_SStructDiagScale__array*
bHYPRE_SStructDiagScale__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructDiagScale__array*
bHYPRE_SStructDiagScale__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructDiagScale__array*
bHYPRE_SStructDiagScale__array_create1d(int32_t len);

struct bHYPRE_SStructDiagScale__array*
bHYPRE_SStructDiagScale__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructDiagScale* data);

struct bHYPRE_SStructDiagScale__array*
bHYPRE_SStructDiagScale__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructDiagScale__array*
bHYPRE_SStructDiagScale__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructDiagScale__array*
bHYPRE_SStructDiagScale__array_borrow(
  bHYPRE_SStructDiagScale* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_SStructDiagScale__array*
bHYPRE_SStructDiagScale__array_smartCopy(
  struct bHYPRE_SStructDiagScale__array *array);

void
bHYPRE_SStructDiagScale__array_addRef(
  struct bHYPRE_SStructDiagScale__array* array);

void
bHYPRE_SStructDiagScale__array_deleteRef(
  struct bHYPRE_SStructDiagScale__array* array);

bHYPRE_SStructDiagScale
bHYPRE_SStructDiagScale__array_get1(
  const struct bHYPRE_SStructDiagScale__array* array,
  const int32_t i1);

bHYPRE_SStructDiagScale
bHYPRE_SStructDiagScale__array_get2(
  const struct bHYPRE_SStructDiagScale__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_SStructDiagScale
bHYPRE_SStructDiagScale__array_get3(
  const struct bHYPRE_SStructDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_SStructDiagScale
bHYPRE_SStructDiagScale__array_get4(
  const struct bHYPRE_SStructDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_SStructDiagScale
bHYPRE_SStructDiagScale__array_get5(
  const struct bHYPRE_SStructDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_SStructDiagScale
bHYPRE_SStructDiagScale__array_get6(
  const struct bHYPRE_SStructDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_SStructDiagScale
bHYPRE_SStructDiagScale__array_get7(
  const struct bHYPRE_SStructDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_SStructDiagScale
bHYPRE_SStructDiagScale__array_get(
  const struct bHYPRE_SStructDiagScale__array* array,
  const int32_t indices[]);

void
bHYPRE_SStructDiagScale__array_set1(
  struct bHYPRE_SStructDiagScale__array* array,
  const int32_t i1,
  bHYPRE_SStructDiagScale const value);

void
bHYPRE_SStructDiagScale__array_set2(
  struct bHYPRE_SStructDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructDiagScale const value);

void
bHYPRE_SStructDiagScale__array_set3(
  struct bHYPRE_SStructDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructDiagScale const value);

void
bHYPRE_SStructDiagScale__array_set4(
  struct bHYPRE_SStructDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructDiagScale const value);

void
bHYPRE_SStructDiagScale__array_set5(
  struct bHYPRE_SStructDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructDiagScale const value);

void
bHYPRE_SStructDiagScale__array_set6(
  struct bHYPRE_SStructDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructDiagScale const value);

void
bHYPRE_SStructDiagScale__array_set7(
  struct bHYPRE_SStructDiagScale__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructDiagScale const value);

void
bHYPRE_SStructDiagScale__array_set(
  struct bHYPRE_SStructDiagScale__array* array,
  const int32_t indices[],
  bHYPRE_SStructDiagScale const value);

int32_t
bHYPRE_SStructDiagScale__array_dimen(
  const struct bHYPRE_SStructDiagScale__array* array);

int32_t
bHYPRE_SStructDiagScale__array_lower(
  const struct bHYPRE_SStructDiagScale__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructDiagScale__array_upper(
  const struct bHYPRE_SStructDiagScale__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructDiagScale__array_length(
  const struct bHYPRE_SStructDiagScale__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructDiagScale__array_stride(
  const struct bHYPRE_SStructDiagScale__array* array,
  const int32_t ind);

int
bHYPRE_SStructDiagScale__array_isColumnOrder(
  const struct bHYPRE_SStructDiagScale__array* array);

int
bHYPRE_SStructDiagScale__array_isRowOrder(
  const struct bHYPRE_SStructDiagScale__array* array);

struct bHYPRE_SStructDiagScale__array*
bHYPRE_SStructDiagScale__array_slice(
  struct bHYPRE_SStructDiagScale__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_SStructDiagScale__array_copy(
  const struct bHYPRE_SStructDiagScale__array* src,
  struct bHYPRE_SStructDiagScale__array* dest);

struct bHYPRE_SStructDiagScale__array*
bHYPRE_SStructDiagScale__array_ensure(
  struct bHYPRE_SStructDiagScale__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
