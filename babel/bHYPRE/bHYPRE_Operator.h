/*
 * File:          bHYPRE_Operator.h
 * Symbol:        bHYPRE.Operator-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.Operator
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_Operator_h
#define included_bHYPRE_Operator_h

/**
 * Symbol "bHYPRE.Operator" (version 1.0.0)
 * 
 * An Operator is anything that maps one Vector to another.  The
 * terms {\tt Setup} and {\tt Apply} are reserved for Operators.
 * The implementation is allowed to assume that supplied parameter
 * arrays will not be destroyed.
 * 
 */
struct bHYPRE_Operator__object;
struct bHYPRE_Operator__array;
typedef struct bHYPRE_Operator__object* bHYPRE_Operator;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
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
 * RMI connector function for the class.
 */
bHYPRE_Operator
bHYPRE_Operator__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_Operator_addRef(
  /* in */ bHYPRE_Operator self);

void
bHYPRE_Operator_deleteRef(
  /* in */ bHYPRE_Operator self);

sidl_bool
bHYPRE_Operator_isSame(
  /* in */ bHYPRE_Operator self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_Operator_queryInt(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name);

sidl_bool
bHYPRE_Operator_isType(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_Operator_getClassInfo(
  /* in */ bHYPRE_Operator self);

/**
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */
int32_t
bHYPRE_Operator_SetCommunicator(
  /* in */ bHYPRE_Operator self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Operator_SetIntParameter(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* in */ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Operator_SetDoubleParameter(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* in */ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Operator_SetStringParameter(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* in */ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Operator_SetIntArray1Parameter(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Operator_SetIntArray2Parameter(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Operator_SetDoubleArray1Parameter(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Operator_SetDoubleArray2Parameter(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Operator_GetIntValue(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* out */ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_Operator_GetDoubleValue(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* out */ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_Operator_Setup(
  /* in */ bHYPRE_Operator self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_Operator_Apply(
  /* in */ bHYPRE_Operator self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_Operator_ApplyAdjoint(
  /* in */ bHYPRE_Operator self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_Operator__object*
bHYPRE_Operator__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_Operator__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_Operator__exec(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_Operator__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_Operator__getURL(
  /* in */ bHYPRE_Operator self);
struct bHYPRE_Operator__array*
bHYPRE_Operator__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_Operator__array*
bHYPRE_Operator__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_Operator__array*
bHYPRE_Operator__array_create1d(int32_t len);

struct bHYPRE_Operator__array*
bHYPRE_Operator__array_create1dInit(
  int32_t len, 
  bHYPRE_Operator* data);

struct bHYPRE_Operator__array*
bHYPRE_Operator__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_Operator__array*
bHYPRE_Operator__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_Operator__array*
bHYPRE_Operator__array_borrow(
  bHYPRE_Operator* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_Operator__array*
bHYPRE_Operator__array_smartCopy(
  struct bHYPRE_Operator__array *array);

void
bHYPRE_Operator__array_addRef(
  struct bHYPRE_Operator__array* array);

void
bHYPRE_Operator__array_deleteRef(
  struct bHYPRE_Operator__array* array);

bHYPRE_Operator
bHYPRE_Operator__array_get1(
  const struct bHYPRE_Operator__array* array,
  const int32_t i1);

bHYPRE_Operator
bHYPRE_Operator__array_get2(
  const struct bHYPRE_Operator__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_Operator
bHYPRE_Operator__array_get3(
  const struct bHYPRE_Operator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_Operator
bHYPRE_Operator__array_get4(
  const struct bHYPRE_Operator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_Operator
bHYPRE_Operator__array_get5(
  const struct bHYPRE_Operator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_Operator
bHYPRE_Operator__array_get6(
  const struct bHYPRE_Operator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_Operator
bHYPRE_Operator__array_get7(
  const struct bHYPRE_Operator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_Operator
bHYPRE_Operator__array_get(
  const struct bHYPRE_Operator__array* array,
  const int32_t indices[]);

void
bHYPRE_Operator__array_set1(
  struct bHYPRE_Operator__array* array,
  const int32_t i1,
  bHYPRE_Operator const value);

void
bHYPRE_Operator__array_set2(
  struct bHYPRE_Operator__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_Operator const value);

void
bHYPRE_Operator__array_set3(
  struct bHYPRE_Operator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_Operator const value);

void
bHYPRE_Operator__array_set4(
  struct bHYPRE_Operator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_Operator const value);

void
bHYPRE_Operator__array_set5(
  struct bHYPRE_Operator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_Operator const value);

void
bHYPRE_Operator__array_set6(
  struct bHYPRE_Operator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_Operator const value);

void
bHYPRE_Operator__array_set7(
  struct bHYPRE_Operator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_Operator const value);

void
bHYPRE_Operator__array_set(
  struct bHYPRE_Operator__array* array,
  const int32_t indices[],
  bHYPRE_Operator const value);

int32_t
bHYPRE_Operator__array_dimen(
  const struct bHYPRE_Operator__array* array);

int32_t
bHYPRE_Operator__array_lower(
  const struct bHYPRE_Operator__array* array,
  const int32_t ind);

int32_t
bHYPRE_Operator__array_upper(
  const struct bHYPRE_Operator__array* array,
  const int32_t ind);

int32_t
bHYPRE_Operator__array_length(
  const struct bHYPRE_Operator__array* array,
  const int32_t ind);

int32_t
bHYPRE_Operator__array_stride(
  const struct bHYPRE_Operator__array* array,
  const int32_t ind);

int
bHYPRE_Operator__array_isColumnOrder(
  const struct bHYPRE_Operator__array* array);

int
bHYPRE_Operator__array_isRowOrder(
  const struct bHYPRE_Operator__array* array);

struct bHYPRE_Operator__array*
bHYPRE_Operator__array_slice(
  struct bHYPRE_Operator__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_Operator__array_copy(
  const struct bHYPRE_Operator__array* src,
  struct bHYPRE_Operator__array* dest);

struct bHYPRE_Operator__array*
bHYPRE_Operator__array_ensure(
  struct bHYPRE_Operator__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
