/*
 * File:          bHYPRE_StructJacobi.h
 * Symbol:        bHYPRE.StructJacobi-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Client-side glue code for bHYPRE.StructJacobi
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_StructJacobi_h
#define included_bHYPRE_StructJacobi_h

/**
 * Symbol "bHYPRE.StructJacobi" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects
 * using the {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 * The StructJacobi solver requires a Struct matrix.
 * 
 * 
 */
struct bHYPRE_StructJacobi__object;
struct bHYPRE_StructJacobi__array;
typedef struct bHYPRE_StructJacobi__object* bHYPRE_StructJacobi;

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
#ifndef included_bHYPRE_StructMatrix_h
#include "bHYPRE_StructMatrix.h"
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
struct bHYPRE_StructJacobi__object*
bHYPRE_StructJacobi__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_StructJacobi
bHYPRE_StructJacobi__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_StructJacobi
bHYPRE_StructJacobi__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_StructJacobi_addRef(
  /* in */ bHYPRE_StructJacobi self);

void
bHYPRE_StructJacobi_deleteRef(
  /* in */ bHYPRE_StructJacobi self);

sidl_bool
bHYPRE_StructJacobi_isSame(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_StructJacobi_queryInt(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name);

sidl_bool
bHYPRE_StructJacobi_isType(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_StructJacobi_getClassInfo(
  /* in */ bHYPRE_StructJacobi self);

/**
 * Method:  Create[]
 */
bHYPRE_StructJacobi
bHYPRE_StructJacobi_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_StructMatrix A);

/**
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */
int32_t
bHYPRE_StructJacobi_SetCommunicator(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructJacobi_SetIntParameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in */ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructJacobi_SetDoubleParameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in */ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructJacobi_SetStringParameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in */ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructJacobi_SetIntArray1Parameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructJacobi_SetIntArray2Parameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructJacobi_SetDoubleArray1Parameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructJacobi_SetDoubleArray2Parameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructJacobi_GetIntValue(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* out */ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructJacobi_GetDoubleValue(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* out */ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_StructJacobi_Setup(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_StructJacobi_Apply(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_StructJacobi_ApplyAdjoint(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 * 
 */
int32_t
bHYPRE_StructJacobi_SetOperator(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ bHYPRE_Operator A);

/**
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */
int32_t
bHYPRE_StructJacobi_SetTolerance(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ double tolerance);

/**
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */
int32_t
bHYPRE_StructJacobi_SetMaxIterations(
  /* in */ bHYPRE_StructJacobi self,
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
bHYPRE_StructJacobi_SetLogging(
  /* in */ bHYPRE_StructJacobi self,
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
bHYPRE_StructJacobi_SetPrintLevel(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ int32_t level);

/**
 * (Optional) Return the number of iterations taken.
 * 
 */
int32_t
bHYPRE_StructJacobi_GetNumIterations(
  /* in */ bHYPRE_StructJacobi self,
  /* out */ int32_t* num_iterations);

/**
 * (Optional) Return the norm of the relative residual.
 * 
 */
int32_t
bHYPRE_StructJacobi_GetRelResidualNorm(
  /* in */ bHYPRE_StructJacobi self,
  /* out */ double* norm);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructJacobi__object*
bHYPRE_StructJacobi__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructJacobi__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_StructJacobi__exec(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_StructJacobi__getURL(
  /* in */ bHYPRE_StructJacobi self);
struct bHYPRE_StructJacobi__array*
bHYPRE_StructJacobi__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructJacobi__array*
bHYPRE_StructJacobi__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructJacobi__array*
bHYPRE_StructJacobi__array_create1d(int32_t len);

struct bHYPRE_StructJacobi__array*
bHYPRE_StructJacobi__array_create1dInit(
  int32_t len, 
  bHYPRE_StructJacobi* data);

struct bHYPRE_StructJacobi__array*
bHYPRE_StructJacobi__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_StructJacobi__array*
bHYPRE_StructJacobi__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_StructJacobi__array*
bHYPRE_StructJacobi__array_borrow(
  bHYPRE_StructJacobi* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_StructJacobi__array*
bHYPRE_StructJacobi__array_smartCopy(
  struct bHYPRE_StructJacobi__array *array);

void
bHYPRE_StructJacobi__array_addRef(
  struct bHYPRE_StructJacobi__array* array);

void
bHYPRE_StructJacobi__array_deleteRef(
  struct bHYPRE_StructJacobi__array* array);

bHYPRE_StructJacobi
bHYPRE_StructJacobi__array_get1(
  const struct bHYPRE_StructJacobi__array* array,
  const int32_t i1);

bHYPRE_StructJacobi
bHYPRE_StructJacobi__array_get2(
  const struct bHYPRE_StructJacobi__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_StructJacobi
bHYPRE_StructJacobi__array_get3(
  const struct bHYPRE_StructJacobi__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_StructJacobi
bHYPRE_StructJacobi__array_get4(
  const struct bHYPRE_StructJacobi__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_StructJacobi
bHYPRE_StructJacobi__array_get5(
  const struct bHYPRE_StructJacobi__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_StructJacobi
bHYPRE_StructJacobi__array_get6(
  const struct bHYPRE_StructJacobi__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_StructJacobi
bHYPRE_StructJacobi__array_get7(
  const struct bHYPRE_StructJacobi__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_StructJacobi
bHYPRE_StructJacobi__array_get(
  const struct bHYPRE_StructJacobi__array* array,
  const int32_t indices[]);

void
bHYPRE_StructJacobi__array_set1(
  struct bHYPRE_StructJacobi__array* array,
  const int32_t i1,
  bHYPRE_StructJacobi const value);

void
bHYPRE_StructJacobi__array_set2(
  struct bHYPRE_StructJacobi__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_StructJacobi const value);

void
bHYPRE_StructJacobi__array_set3(
  struct bHYPRE_StructJacobi__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_StructJacobi const value);

void
bHYPRE_StructJacobi__array_set4(
  struct bHYPRE_StructJacobi__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_StructJacobi const value);

void
bHYPRE_StructJacobi__array_set5(
  struct bHYPRE_StructJacobi__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_StructJacobi const value);

void
bHYPRE_StructJacobi__array_set6(
  struct bHYPRE_StructJacobi__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_StructJacobi const value);

void
bHYPRE_StructJacobi__array_set7(
  struct bHYPRE_StructJacobi__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_StructJacobi const value);

void
bHYPRE_StructJacobi__array_set(
  struct bHYPRE_StructJacobi__array* array,
  const int32_t indices[],
  bHYPRE_StructJacobi const value);

int32_t
bHYPRE_StructJacobi__array_dimen(
  const struct bHYPRE_StructJacobi__array* array);

int32_t
bHYPRE_StructJacobi__array_lower(
  const struct bHYPRE_StructJacobi__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructJacobi__array_upper(
  const struct bHYPRE_StructJacobi__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructJacobi__array_length(
  const struct bHYPRE_StructJacobi__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructJacobi__array_stride(
  const struct bHYPRE_StructJacobi__array* array,
  const int32_t ind);

int
bHYPRE_StructJacobi__array_isColumnOrder(
  const struct bHYPRE_StructJacobi__array* array);

int
bHYPRE_StructJacobi__array_isRowOrder(
  const struct bHYPRE_StructJacobi__array* array);

struct bHYPRE_StructJacobi__array*
bHYPRE_StructJacobi__array_slice(
  struct bHYPRE_StructJacobi__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_StructJacobi__array_copy(
  const struct bHYPRE_StructJacobi__array* src,
  struct bHYPRE_StructJacobi__array* dest);

struct bHYPRE_StructJacobi__array*
bHYPRE_StructJacobi__array_ensure(
  struct bHYPRE_StructJacobi__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
