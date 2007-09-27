/*
 * File:          bHYPRE_Operator.h
 * Symbol:        bHYPRE.Operator-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.4
 * Description:   Client-side glue code for bHYPRE.Operator
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
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
#ifndef included_sidl_BaseException_h
#include "sidl_BaseException.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif

#ifndef included_sidl_rmi_Call_h
#include "sidl_rmi_Call.h"
#endif
#ifndef included_sidl_rmi_Return_h
#include "sidl_rmi_Return.h"
#endif
#ifdef SIDL_C_HAS_INLINE
#ifndef included_bHYPRE_Operator_IOR_h
#include "bHYPRE_Operator_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_Operator
bHYPRE_Operator__connect(const char *, sidl_BaseInterface *_ex);

/**
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Operator_SetCommunicator(
  /* in */ bHYPRE_Operator self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_SetCommunicator)(
    self->d_object,
    mpi_comm,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */
SIDL_C_INLINE_DECL
void
bHYPRE_Operator_Destroy(
  /* in */ bHYPRE_Operator self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_Destroy)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the int parameter associated with {\tt name}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Operator_SetIntParameter(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_SetIntParameter)(
    self->d_object,
    name,
    value,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the double parameter associated with {\tt name}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Operator_SetDoubleParameter(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_SetDoubleParameter)(
    self->d_object,
    name,
    value,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the string parameter associated with {\tt name}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Operator_SetStringParameter(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_SetStringParameter)(
    self->d_object,
    name,
    value,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the int 1-D array parameter associated with {\tt name}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Operator_SetIntArray1Parameter(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  int32_t value_lower[1], value_upper[1], value_stride[1]; 
  struct sidl_int__array value_real;
  struct sidl_int__array*value_tmp = &value_real;
  value_upper[0] = nvalues-1;
  sidl_int__array_init(value, value_tmp, 1, value_lower, value_upper, 
    value_stride);
  _result = (*self->d_epv->f_SetIntArray1Parameter)(
    self->d_object,
    name,
    value_tmp,
    _ex);
#ifdef SIDL_DEBUG_REFCOUNT
  sidl__array_deleteRef((struct sidl__array*)value_tmp);
#endif /* SIDL_DEBUG_REFCOUNT */
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the int 2-D array parameter associated with {\tt name}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Operator_SetIntArray2Parameter(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_SetIntArray2Parameter)(
    self->d_object,
    name,
    value,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the double 1-D array parameter associated with {\tt name}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Operator_SetDoubleArray1Parameter(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  int32_t value_lower[1], value_upper[1], value_stride[1]; 
  struct sidl_double__array value_real;
  struct sidl_double__array*value_tmp = &value_real;
  value_upper[0] = nvalues-1;
  sidl_double__array_init(value, value_tmp, 1, value_lower, value_upper, 
    value_stride);
  _result = (*self->d_epv->f_SetDoubleArray1Parameter)(
    self->d_object,
    name,
    value_tmp,
    _ex);
#ifdef SIDL_DEBUG_REFCOUNT
  sidl__array_deleteRef((struct sidl__array*)value_tmp);
#endif /* SIDL_DEBUG_REFCOUNT */
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the double 2-D array parameter associated with {\tt name}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Operator_SetDoubleArray2Parameter(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_SetDoubleArray2Parameter)(
    self->d_object,
    name,
    value,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the int parameter associated with {\tt name}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Operator_GetIntValue(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_GetIntValue)(
    self->d_object,
    name,
    value,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Get the double parameter associated with {\tt name}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Operator_GetDoubleValue(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_GetDoubleValue)(
    self->d_object,
    name,
    value,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Operator_Setup(
  /* in */ bHYPRE_Operator self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_Setup)(
    self->d_object,
    b,
    x,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Operator_Apply(
  /* in */ bHYPRE_Operator self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_Apply)(
    self->d_object,
    b,
    x,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Operator_ApplyAdjoint(
  /* in */ bHYPRE_Operator self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_ApplyAdjoint)(
    self->d_object,
    b,
    x,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
void
bHYPRE_Operator_addRef(
  /* in */ bHYPRE_Operator self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_addRef)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
void
bHYPRE_Operator_deleteRef(
  /* in */ bHYPRE_Operator self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_deleteRef)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_bool
bHYPRE_Operator_isSame(
  /* in */ bHYPRE_Operator self,
  /* in */ sidl_BaseInterface iobj,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  sidl_bool _result;
  _result = (*self->d_epv->f_isSame)(
    self->d_object,
    iobj,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_bool
bHYPRE_Operator_isType(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* name,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  sidl_bool _result;
  _result = (*self->d_epv->f_isType)(
    self->d_object,
    name,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_ClassInfo
bHYPRE_Operator_getClassInfo(
  /* in */ bHYPRE_Operator self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  sidl_ClassInfo _result;
  _result = (*self->d_epv->f_getClassInfo)(
    self->d_object,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_Operator__object*
bHYPRE_Operator__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_Operator__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_Operator__exec(
  /* in */ bHYPRE_Operator self,
  /* in */ const char* methodName,
  /* in */ sidl_rmi_Call inArgs,
  /* in */ sidl_rmi_Return outArgs,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f__exec)(
    self->d_object,
    methodName,
    inArgs,
    outArgs,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * Get the URL of the Implementation of this object (for RMI)
 */
SIDL_C_INLINE_DECL
char*
bHYPRE_Operator__getURL(
  /* in */ bHYPRE_Operator self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  char* _result;
  _result = (*self->d_epv->f__getURL)(
    self->d_object,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * On a remote object, addrefs the remote instance.
 */
SIDL_C_INLINE_DECL
void
bHYPRE_Operator__raddRef(
  /* in */ bHYPRE_Operator self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f__raddRef)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * TRUE if this object is remote, false if local
 */
SIDL_C_INLINE_DECL
sidl_bool
bHYPRE_Operator__isRemote(
  /* in */ bHYPRE_Operator self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  sidl_bool _result;
  _result = (*self->d_epv->f__isRemote)(
    self->d_object,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * TRUE if this object is remote, false if local
 */
sidl_bool
bHYPRE_Operator__isLocal(
  /* in */ bHYPRE_Operator self,
  /* out */ sidl_BaseInterface *_ex);
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


#pragma weak bHYPRE_Operator__connectI

#pragma weak bHYPRE_Operator__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_Operator__object*
bHYPRE_Operator__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_Operator__object*
bHYPRE_Operator__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
