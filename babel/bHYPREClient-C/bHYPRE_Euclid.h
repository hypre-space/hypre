/*
 * File:          bHYPRE_Euclid.h
 * Symbol:        bHYPRE.Euclid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.Euclid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_Euclid_h
#define included_bHYPRE_Euclid_h

/**
 * Symbol "bHYPRE.Euclid" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * Although the usual Solver Set*Parameter functions are available,
 * a Euclid-stype parameter-setting function is also available, SetParameters.
 * 
 * 
 * 
 */
struct bHYPRE_Euclid__object;
struct bHYPRE_Euclid__array;
typedef struct bHYPRE_Euclid__object* bHYPRE_Euclid;

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
struct bHYPRE_Euclid__object*
bHYPRE_Euclid__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_Euclid
bHYPRE_Euclid__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_Euclid
bHYPRE_Euclid__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_Euclid_addRef(
  /* in */ bHYPRE_Euclid self);

void
bHYPRE_Euclid_deleteRef(
  /* in */ bHYPRE_Euclid self);

sidl_bool
bHYPRE_Euclid_isSame(
  /* in */ bHYPRE_Euclid self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_Euclid_queryInt(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name);

sidl_bool
bHYPRE_Euclid_isType(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_Euclid_getClassInfo(
  /* in */ bHYPRE_Euclid self);

/**
 * Method:  Create[]
 */
bHYPRE_Euclid
bHYPRE_Euclid_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A);

/**
 * Method:  SetParameters[]
 */
int32_t
bHYPRE_Euclid_SetParameters(
  /* in */ bHYPRE_Euclid self,
  /* in */ int32_t argc,
  /* inout */ char** argv);

/**
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */
int32_t
bHYPRE_Euclid_SetCommunicator(
  /* in */ bHYPRE_Euclid self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Euclid_SetIntParameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in */ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Euclid_SetDoubleParameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in */ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Euclid_SetStringParameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in */ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Euclid_SetIntArray1Parameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Euclid_SetIntArray2Parameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Euclid_SetDoubleArray1Parameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Euclid_SetDoubleArray2Parameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Euclid_GetIntValue(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* out */ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Euclid_GetDoubleValue(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* out */ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_Euclid_Setup(
  /* in */ bHYPRE_Euclid self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_Euclid_Apply(
  /* in */ bHYPRE_Euclid self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_Euclid_ApplyAdjoint(
  /* in */ bHYPRE_Euclid self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 * 
 */
int32_t
bHYPRE_Euclid_SetOperator(
  /* in */ bHYPRE_Euclid self,
  /* in */ bHYPRE_Operator A);

/**
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */
int32_t
bHYPRE_Euclid_SetTolerance(
  /* in */ bHYPRE_Euclid self,
  /* in */ double tolerance);

/**
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */
int32_t
bHYPRE_Euclid_SetMaxIterations(
  /* in */ bHYPRE_Euclid self,
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
bHYPRE_Euclid_SetLogging(
  /* in */ bHYPRE_Euclid self,
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
bHYPRE_Euclid_SetPrintLevel(
  /* in */ bHYPRE_Euclid self,
  /* in */ int32_t level);

/**
 * (Optional) Return the number of iterations taken.
 * 
 */
int32_t
bHYPRE_Euclid_GetNumIterations(
  /* in */ bHYPRE_Euclid self,
  /* out */ int32_t* num_iterations);

/**
 * (Optional) Return the norm of the relative residual.
 * 
 */
int32_t
bHYPRE_Euclid_GetRelResidualNorm(
  /* in */ bHYPRE_Euclid self,
  /* out */ double* norm);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_Euclid__object*
bHYPRE_Euclid__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_Euclid__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_Euclid__exec(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_Euclid__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_Euclid__getURL(
  /* in */ bHYPRE_Euclid self);
struct bHYPRE_Euclid__array*
bHYPRE_Euclid__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_Euclid__array*
bHYPRE_Euclid__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_Euclid__array*
bHYPRE_Euclid__array_create1d(int32_t len);

struct bHYPRE_Euclid__array*
bHYPRE_Euclid__array_create1dInit(
  int32_t len, 
  bHYPRE_Euclid* data);

struct bHYPRE_Euclid__array*
bHYPRE_Euclid__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_Euclid__array*
bHYPRE_Euclid__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_Euclid__array*
bHYPRE_Euclid__array_borrow(
  bHYPRE_Euclid* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_Euclid__array*
bHYPRE_Euclid__array_smartCopy(
  struct bHYPRE_Euclid__array *array);

void
bHYPRE_Euclid__array_addRef(
  struct bHYPRE_Euclid__array* array);

void
bHYPRE_Euclid__array_deleteRef(
  struct bHYPRE_Euclid__array* array);

bHYPRE_Euclid
bHYPRE_Euclid__array_get1(
  const struct bHYPRE_Euclid__array* array,
  const int32_t i1);

bHYPRE_Euclid
bHYPRE_Euclid__array_get2(
  const struct bHYPRE_Euclid__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_Euclid
bHYPRE_Euclid__array_get3(
  const struct bHYPRE_Euclid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_Euclid
bHYPRE_Euclid__array_get4(
  const struct bHYPRE_Euclid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_Euclid
bHYPRE_Euclid__array_get5(
  const struct bHYPRE_Euclid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_Euclid
bHYPRE_Euclid__array_get6(
  const struct bHYPRE_Euclid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_Euclid
bHYPRE_Euclid__array_get7(
  const struct bHYPRE_Euclid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_Euclid
bHYPRE_Euclid__array_get(
  const struct bHYPRE_Euclid__array* array,
  const int32_t indices[]);

void
bHYPRE_Euclid__array_set1(
  struct bHYPRE_Euclid__array* array,
  const int32_t i1,
  bHYPRE_Euclid const value);

void
bHYPRE_Euclid__array_set2(
  struct bHYPRE_Euclid__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_Euclid const value);

void
bHYPRE_Euclid__array_set3(
  struct bHYPRE_Euclid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_Euclid const value);

void
bHYPRE_Euclid__array_set4(
  struct bHYPRE_Euclid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_Euclid const value);

void
bHYPRE_Euclid__array_set5(
  struct bHYPRE_Euclid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_Euclid const value);

void
bHYPRE_Euclid__array_set6(
  struct bHYPRE_Euclid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_Euclid const value);

void
bHYPRE_Euclid__array_set7(
  struct bHYPRE_Euclid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_Euclid const value);

void
bHYPRE_Euclid__array_set(
  struct bHYPRE_Euclid__array* array,
  const int32_t indices[],
  bHYPRE_Euclid const value);

int32_t
bHYPRE_Euclid__array_dimen(
  const struct bHYPRE_Euclid__array* array);

int32_t
bHYPRE_Euclid__array_lower(
  const struct bHYPRE_Euclid__array* array,
  const int32_t ind);

int32_t
bHYPRE_Euclid__array_upper(
  const struct bHYPRE_Euclid__array* array,
  const int32_t ind);

int32_t
bHYPRE_Euclid__array_length(
  const struct bHYPRE_Euclid__array* array,
  const int32_t ind);

int32_t
bHYPRE_Euclid__array_stride(
  const struct bHYPRE_Euclid__array* array,
  const int32_t ind);

int
bHYPRE_Euclid__array_isColumnOrder(
  const struct bHYPRE_Euclid__array* array);

int
bHYPRE_Euclid__array_isRowOrder(
  const struct bHYPRE_Euclid__array* array);

struct bHYPRE_Euclid__array*
bHYPRE_Euclid__array_slice(
  struct bHYPRE_Euclid__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_Euclid__array_copy(
  const struct bHYPRE_Euclid__array* src,
  struct bHYPRE_Euclid__array* dest);

struct bHYPRE_Euclid__array*
bHYPRE_Euclid__array_ensure(
  struct bHYPRE_Euclid__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
