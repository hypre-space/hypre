/*
 * File:          bHYPRE_Pilut.h
 * Symbol:        bHYPRE.Pilut-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.Pilut
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_Pilut_h
#define included_bHYPRE_Pilut_h

/**
 * Symbol "bHYPRE.Pilut" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * Pilut has not been implemented yet.
 */
struct bHYPRE_Pilut__object;
struct bHYPRE_Pilut__array;
typedef struct bHYPRE_Pilut__object* bHYPRE_Pilut;

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
#ifndef included_bHYPRE_Pilut_IOR_h
#include "bHYPRE_Pilut_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
struct bHYPRE_Pilut__object*
bHYPRE_Pilut__create(sidl_BaseInterface* _ex);

/**
 * RMI constructor function for the class.
 */
bHYPRE_Pilut
bHYPRE_Pilut__createRemote(const char * url, sidl_BaseInterface *_ex);

/**
 * Wraps up the private data struct pointer (struct bHYPRE_Pilut__data) passed in rather than running the constructor.
 */
bHYPRE_Pilut
bHYPRE_Pilut__wrapObj(void * data, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_Pilut
bHYPRE_Pilut__connect(const char *, sidl_BaseInterface *_ex);

/**
 *  This function is the preferred way to create a Pilut solver. 
 */
bHYPRE_Pilut
bHYPRE_Pilut_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex);

SIDL_C_INLINE_DECL
void
bHYPRE_Pilut_addRef(
  /* in */ bHYPRE_Pilut self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_addRef)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
void
bHYPRE_Pilut_deleteRef(
  /* in */ bHYPRE_Pilut self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_deleteRef)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_bool
bHYPRE_Pilut_isSame(
  /* in */ bHYPRE_Pilut self,
  /* in */ sidl_BaseInterface iobj,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_isSame)(
    self,
    iobj,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_bool
bHYPRE_Pilut_isType(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_isType)(
    self,
    name,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_ClassInfo
bHYPRE_Pilut_getClassInfo(
  /* in */ bHYPRE_Pilut self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_getClassInfo)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Pilut_SetOperator(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetOperator)(
    self,
    A,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Pilut_SetTolerance(
  /* in */ bHYPRE_Pilut self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetTolerance)(
    self,
    tolerance,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Pilut_SetMaxIterations(
  /* in */ bHYPRE_Pilut self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetMaxIterations)(
    self,
    max_iterations,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Pilut_SetLogging(
  /* in */ bHYPRE_Pilut self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetLogging)(
    self,
    level,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Pilut_SetPrintLevel(
  /* in */ bHYPRE_Pilut self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetPrintLevel)(
    self,
    level,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * (Optional) Return the number of iterations taken.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Pilut_GetNumIterations(
  /* in */ bHYPRE_Pilut self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_GetNumIterations)(
    self,
    num_iterations,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * (Optional) Return the norm of the relative residual.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Pilut_GetRelResidualNorm(
  /* in */ bHYPRE_Pilut self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_GetRelResidualNorm)(
    self,
    norm,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Pilut_SetCommunicator(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetCommunicator)(
    self,
    mpi_comm,
    _ex);
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
bHYPRE_Pilut_Destroy(
  /* in */ bHYPRE_Pilut self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_Destroy)(
    self,
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
bHYPRE_Pilut_SetIntParameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetIntParameter)(
    self,
    name,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the double parameter associated with {\tt name}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Pilut_SetDoubleParameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetDoubleParameter)(
    self,
    name,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the string parameter associated with {\tt name}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Pilut_SetStringParameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetStringParameter)(
    self,
    name,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the int 1-D array parameter associated with {\tt name}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Pilut_SetIntArray1Parameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t value_lower[1], value_upper[1], value_stride[1]; 
  struct sidl_int__array value_real;
  struct sidl_int__array*value_tmp = &value_real;
  value_upper[0] = nvalues-1;
  sidl_int__array_init(value, value_tmp, 1, value_lower, value_upper,
    value_stride);
  return (*self->d_epv->f_SetIntArray1Parameter)(
    self,
    name,
    value_tmp,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the int 2-D array parameter associated with {\tt name}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Pilut_SetIntArray2Parameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetIntArray2Parameter)(
    self,
    name,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the double 1-D array parameter associated with {\tt name}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Pilut_SetDoubleArray1Parameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t value_lower[1], value_upper[1], value_stride[1]; 
  struct sidl_double__array value_real;
  struct sidl_double__array*value_tmp = &value_real;
  value_upper[0] = nvalues-1;
  sidl_double__array_init(value, value_tmp, 1, value_lower, value_upper,
    value_stride);
  return (*self->d_epv->f_SetDoubleArray1Parameter)(
    self,
    name,
    value_tmp,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the double 2-D array parameter associated with {\tt name}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Pilut_SetDoubleArray2Parameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetDoubleArray2Parameter)(
    self,
    name,
    value,
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
bHYPRE_Pilut_GetIntValue(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_GetIntValue)(
    self,
    name,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Get the double parameter associated with {\tt name}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Pilut_GetDoubleValue(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_GetDoubleValue)(
    self,
    name,
    value,
    _ex);
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
bHYPRE_Pilut_Setup(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Setup)(
    self,
    b,
    x,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Pilut_Apply(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Apply)(
    self,
    b,
    x,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Pilut_ApplyAdjoint(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_ApplyAdjoint)(
    self,
    b,
    x,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_Pilut__object*
bHYPRE_Pilut__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_Pilut__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_Pilut__exec(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* methodName,
  /* in */ sidl_rmi_Call inArgs,
  /* in */ sidl_rmi_Return outArgs,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f__exec)(
    self,
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
bHYPRE_Pilut__getURL(
  /* in */ bHYPRE_Pilut self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f__getURL)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * On a remote object, addrefs the remote instance.
 */
SIDL_C_INLINE_DECL
void
bHYPRE_Pilut__raddRef(
  /* in */ bHYPRE_Pilut self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f__raddRef)(
    self,
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
bHYPRE_Pilut__isRemote(
  /* in */ bHYPRE_Pilut self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f__isRemote)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * TRUE if this object is remote, false if local
 */
sidl_bool
bHYPRE_Pilut__isLocal(
  /* in */ bHYPRE_Pilut self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_create1d(int32_t len);

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_create1dInit(
  int32_t len, 
  bHYPRE_Pilut* data);

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_borrow(
  bHYPRE_Pilut* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_smartCopy(
  struct bHYPRE_Pilut__array *array);

void
bHYPRE_Pilut__array_addRef(
  struct bHYPRE_Pilut__array* array);

void
bHYPRE_Pilut__array_deleteRef(
  struct bHYPRE_Pilut__array* array);

bHYPRE_Pilut
bHYPRE_Pilut__array_get1(
  const struct bHYPRE_Pilut__array* array,
  const int32_t i1);

bHYPRE_Pilut
bHYPRE_Pilut__array_get2(
  const struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_Pilut
bHYPRE_Pilut__array_get3(
  const struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_Pilut
bHYPRE_Pilut__array_get4(
  const struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_Pilut
bHYPRE_Pilut__array_get5(
  const struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_Pilut
bHYPRE_Pilut__array_get6(
  const struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_Pilut
bHYPRE_Pilut__array_get7(
  const struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_Pilut
bHYPRE_Pilut__array_get(
  const struct bHYPRE_Pilut__array* array,
  const int32_t indices[]);

void
bHYPRE_Pilut__array_set1(
  struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  bHYPRE_Pilut const value);

void
bHYPRE_Pilut__array_set2(
  struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_Pilut const value);

void
bHYPRE_Pilut__array_set3(
  struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_Pilut const value);

void
bHYPRE_Pilut__array_set4(
  struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_Pilut const value);

void
bHYPRE_Pilut__array_set5(
  struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_Pilut const value);

void
bHYPRE_Pilut__array_set6(
  struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_Pilut const value);

void
bHYPRE_Pilut__array_set7(
  struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_Pilut const value);

void
bHYPRE_Pilut__array_set(
  struct bHYPRE_Pilut__array* array,
  const int32_t indices[],
  bHYPRE_Pilut const value);

int32_t
bHYPRE_Pilut__array_dimen(
  const struct bHYPRE_Pilut__array* array);

int32_t
bHYPRE_Pilut__array_lower(
  const struct bHYPRE_Pilut__array* array,
  const int32_t ind);

int32_t
bHYPRE_Pilut__array_upper(
  const struct bHYPRE_Pilut__array* array,
  const int32_t ind);

int32_t
bHYPRE_Pilut__array_length(
  const struct bHYPRE_Pilut__array* array,
  const int32_t ind);

int32_t
bHYPRE_Pilut__array_stride(
  const struct bHYPRE_Pilut__array* array,
  const int32_t ind);

int
bHYPRE_Pilut__array_isColumnOrder(
  const struct bHYPRE_Pilut__array* array);

int
bHYPRE_Pilut__array_isRowOrder(
  const struct bHYPRE_Pilut__array* array);

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_slice(
  struct bHYPRE_Pilut__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_Pilut__array_copy(
  const struct bHYPRE_Pilut__array* src,
  struct bHYPRE_Pilut__array* dest);

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_ensure(
  struct bHYPRE_Pilut__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_Pilut__connectI

#pragma weak bHYPRE_Pilut__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_Pilut__object*
bHYPRE_Pilut__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_Pilut__object*
bHYPRE_Pilut__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
