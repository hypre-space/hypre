/*
 * File:          bHYPRE_StructPFMG.h
 * Symbol:        bHYPRE.StructPFMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.StructPFMG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_StructPFMG_h
#define included_bHYPRE_StructPFMG_h

/**
 * Symbol "bHYPRE.StructPFMG" (version 1.0.0)
 * 
 * Objects of this type can be cast to PreconditionedSolver objects
 * using the {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 * The StructPFMG solver requires a Struct matrix.
 * 
 * 
 */
struct bHYPRE_StructPFMG__object;
struct bHYPRE_StructPFMG__array;
typedef struct bHYPRE_StructPFMG__object* bHYPRE_StructPFMG;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
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
struct bHYPRE_StructPFMG__object*
bHYPRE_StructPFMG__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_StructPFMG
bHYPRE_StructPFMG__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_StructPFMG
bHYPRE_StructPFMG__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_StructPFMG_addRef(
  /* in */ bHYPRE_StructPFMG self);

void
bHYPRE_StructPFMG_deleteRef(
  /* in */ bHYPRE_StructPFMG self);

sidl_bool
bHYPRE_StructPFMG_isSame(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_StructPFMG_queryInt(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name);

sidl_bool
bHYPRE_StructPFMG_isType(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_StructPFMG_getClassInfo(
  /* in */ bHYPRE_StructPFMG self);

/**
 * Method:  Create[]
 */
bHYPRE_StructPFMG
bHYPRE_StructPFMG_Create(
  /* in */ void* mpi_comm);

/**
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */
int32_t
bHYPRE_StructPFMG_SetCommunicator(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ void* mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructPFMG_SetIntParameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in */ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructPFMG_SetDoubleParameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in */ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructPFMG_SetStringParameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in */ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructPFMG_SetIntArray1Parameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructPFMG_SetIntArray2Parameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructPFMG_SetDoubleArray1Parameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructPFMG_SetDoubleArray2Parameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructPFMG_GetIntValue(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* out */ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructPFMG_GetDoubleValue(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* out */ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_StructPFMG_Setup(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_StructPFMG_Apply(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Set the operator for the linear system being solved.
 * 
 */
int32_t
bHYPRE_StructPFMG_SetOperator(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ bHYPRE_Operator A);

/**
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */
int32_t
bHYPRE_StructPFMG_SetTolerance(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ double tolerance);

/**
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */
int32_t
bHYPRE_StructPFMG_SetMaxIterations(
  /* in */ bHYPRE_StructPFMG self,
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
bHYPRE_StructPFMG_SetLogging(
  /* in */ bHYPRE_StructPFMG self,
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
bHYPRE_StructPFMG_SetPrintLevel(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ int32_t level);

/**
 * (Optional) Return the number of iterations taken.
 * 
 */
int32_t
bHYPRE_StructPFMG_GetNumIterations(
  /* in */ bHYPRE_StructPFMG self,
  /* out */ int32_t* num_iterations);

/**
 * (Optional) Return the norm of the relative residual.
 * 
 */
int32_t
bHYPRE_StructPFMG_GetRelResidualNorm(
  /* in */ bHYPRE_StructPFMG self,
  /* out */ double* norm);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructPFMG__object*
bHYPRE_StructPFMG__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructPFMG__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_StructPFMG__exec(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_StructPFMG__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_StructPFMG__getURL(
  /* in */ bHYPRE_StructPFMG self);
struct bHYPRE_StructPFMG__array*
bHYPRE_StructPFMG__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructPFMG__array*
bHYPRE_StructPFMG__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructPFMG__array*
bHYPRE_StructPFMG__array_create1d(int32_t len);

struct bHYPRE_StructPFMG__array*
bHYPRE_StructPFMG__array_create1dInit(
  int32_t len, 
  bHYPRE_StructPFMG* data);

struct bHYPRE_StructPFMG__array*
bHYPRE_StructPFMG__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_StructPFMG__array*
bHYPRE_StructPFMG__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_StructPFMG__array*
bHYPRE_StructPFMG__array_borrow(
  bHYPRE_StructPFMG* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_StructPFMG__array*
bHYPRE_StructPFMG__array_smartCopy(
  struct bHYPRE_StructPFMG__array *array);

void
bHYPRE_StructPFMG__array_addRef(
  struct bHYPRE_StructPFMG__array* array);

void
bHYPRE_StructPFMG__array_deleteRef(
  struct bHYPRE_StructPFMG__array* array);

bHYPRE_StructPFMG
bHYPRE_StructPFMG__array_get1(
  const struct bHYPRE_StructPFMG__array* array,
  const int32_t i1);

bHYPRE_StructPFMG
bHYPRE_StructPFMG__array_get2(
  const struct bHYPRE_StructPFMG__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_StructPFMG
bHYPRE_StructPFMG__array_get3(
  const struct bHYPRE_StructPFMG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_StructPFMG
bHYPRE_StructPFMG__array_get4(
  const struct bHYPRE_StructPFMG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_StructPFMG
bHYPRE_StructPFMG__array_get5(
  const struct bHYPRE_StructPFMG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_StructPFMG
bHYPRE_StructPFMG__array_get6(
  const struct bHYPRE_StructPFMG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_StructPFMG
bHYPRE_StructPFMG__array_get7(
  const struct bHYPRE_StructPFMG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_StructPFMG
bHYPRE_StructPFMG__array_get(
  const struct bHYPRE_StructPFMG__array* array,
  const int32_t indices[]);

void
bHYPRE_StructPFMG__array_set1(
  struct bHYPRE_StructPFMG__array* array,
  const int32_t i1,
  bHYPRE_StructPFMG const value);

void
bHYPRE_StructPFMG__array_set2(
  struct bHYPRE_StructPFMG__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_StructPFMG const value);

void
bHYPRE_StructPFMG__array_set3(
  struct bHYPRE_StructPFMG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_StructPFMG const value);

void
bHYPRE_StructPFMG__array_set4(
  struct bHYPRE_StructPFMG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_StructPFMG const value);

void
bHYPRE_StructPFMG__array_set5(
  struct bHYPRE_StructPFMG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_StructPFMG const value);

void
bHYPRE_StructPFMG__array_set6(
  struct bHYPRE_StructPFMG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_StructPFMG const value);

void
bHYPRE_StructPFMG__array_set7(
  struct bHYPRE_StructPFMG__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_StructPFMG const value);

void
bHYPRE_StructPFMG__array_set(
  struct bHYPRE_StructPFMG__array* array,
  const int32_t indices[],
  bHYPRE_StructPFMG const value);

int32_t
bHYPRE_StructPFMG__array_dimen(
  const struct bHYPRE_StructPFMG__array* array);

int32_t
bHYPRE_StructPFMG__array_lower(
  const struct bHYPRE_StructPFMG__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructPFMG__array_upper(
  const struct bHYPRE_StructPFMG__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructPFMG__array_length(
  const struct bHYPRE_StructPFMG__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructPFMG__array_stride(
  const struct bHYPRE_StructPFMG__array* array,
  const int32_t ind);

int
bHYPRE_StructPFMG__array_isColumnOrder(
  const struct bHYPRE_StructPFMG__array* array);

int
bHYPRE_StructPFMG__array_isRowOrder(
  const struct bHYPRE_StructPFMG__array* array);

struct bHYPRE_StructPFMG__array*
bHYPRE_StructPFMG__array_slice(
  struct bHYPRE_StructPFMG__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_StructPFMG__array_copy(
  const struct bHYPRE_StructPFMG__array* src,
  struct bHYPRE_StructPFMG__array* dest);

struct bHYPRE_StructPFMG__array*
bHYPRE_StructPFMG__array_ensure(
  struct bHYPRE_StructPFMG__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
