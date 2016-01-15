/*
 * File:          bHYPRE_PreconditionedSolver.h
 * Symbol:        bHYPRE.PreconditionedSolver-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.4
 * Description:   Client-side glue code for bHYPRE.PreconditionedSolver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_PreconditionedSolver_h
#define included_bHYPRE_PreconditionedSolver_h

/**
 * Symbol "bHYPRE.PreconditionedSolver" (version 1.0.0)
 */
struct bHYPRE_PreconditionedSolver__object;
struct bHYPRE_PreconditionedSolver__array;
typedef struct bHYPRE_PreconditionedSolver__object* bHYPRE_PreconditionedSolver;

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
#ifndef included_bHYPRE_PreconditionedSolver_IOR_h
#include "bHYPRE_PreconditionedSolver_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__connect(const char *, sidl_BaseInterface *_ex);

/**
 * Set the preconditioner.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_SetPreconditioner(
  /* in */ bHYPRE_PreconditionedSolver self,
  /* in */ bHYPRE_Solver s,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_SetPreconditioner)(
    self->d_object,
    s,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  GetPreconditioner[]
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_GetPreconditioner(
  /* in */ bHYPRE_PreconditionedSolver self,
  /* out */ bHYPRE_Solver* s,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_GetPreconditioner)(
    self->d_object,
    s,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  Clone[]
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_Clone(
  /* in */ bHYPRE_PreconditionedSolver self,
  /* out */ bHYPRE_PreconditionedSolver* x,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_Clone)(
    self->d_object,
    x,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_SetOperator(
  /* in */ bHYPRE_PreconditionedSolver self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_SetOperator)(
    self->d_object,
    A,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_SetTolerance(
  /* in */ bHYPRE_PreconditionedSolver self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_SetTolerance)(
    self->d_object,
    tolerance,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_SetMaxIterations(
  /* in */ bHYPRE_PreconditionedSolver self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_SetMaxIterations)(
    self->d_object,
    max_iterations,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_SetLogging(
  /* in */ bHYPRE_PreconditionedSolver self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_SetLogging)(
    self->d_object,
    level,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_SetPrintLevel(
  /* in */ bHYPRE_PreconditionedSolver self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_SetPrintLevel)(
    self->d_object,
    level,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_GetNumIterations(
  /* in */ bHYPRE_PreconditionedSolver self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_GetNumIterations)(
    self->d_object,
    num_iterations,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_GetRelResidualNorm(
  /* in */ bHYPRE_PreconditionedSolver self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_GetRelResidualNorm)(
    self->d_object,
    norm,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_SetCommunicator(
  /* in */ bHYPRE_PreconditionedSolver self,
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


SIDL_C_INLINE_DECL
void
bHYPRE_PreconditionedSolver_Destroy(
  /* in */ bHYPRE_PreconditionedSolver self,
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


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_SetIntParameter(
  /* in */ bHYPRE_PreconditionedSolver self,
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


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_SetDoubleParameter(
  /* in */ bHYPRE_PreconditionedSolver self,
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


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_SetStringParameter(
  /* in */ bHYPRE_PreconditionedSolver self,
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


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_SetIntArray1Parameter(
  /* in */ bHYPRE_PreconditionedSolver self,
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


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_SetIntArray2Parameter(
  /* in */ bHYPRE_PreconditionedSolver self,
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


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_SetDoubleArray1Parameter(
  /* in */ bHYPRE_PreconditionedSolver self,
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


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_SetDoubleArray2Parameter(
  /* in */ bHYPRE_PreconditionedSolver self,
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


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_GetIntValue(
  /* in */ bHYPRE_PreconditionedSolver self,
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


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_GetDoubleValue(
  /* in */ bHYPRE_PreconditionedSolver self,
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


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_Setup(
  /* in */ bHYPRE_PreconditionedSolver self,
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


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_Apply(
  /* in */ bHYPRE_PreconditionedSolver self,
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


SIDL_C_INLINE_DECL
int32_t
bHYPRE_PreconditionedSolver_ApplyAdjoint(
  /* in */ bHYPRE_PreconditionedSolver self,
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
bHYPRE_PreconditionedSolver_addRef(
  /* in */ bHYPRE_PreconditionedSolver self,
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
bHYPRE_PreconditionedSolver_deleteRef(
  /* in */ bHYPRE_PreconditionedSolver self,
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
bHYPRE_PreconditionedSolver_isSame(
  /* in */ bHYPRE_PreconditionedSolver self,
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
bHYPRE_PreconditionedSolver_isType(
  /* in */ bHYPRE_PreconditionedSolver self,
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
bHYPRE_PreconditionedSolver_getClassInfo(
  /* in */ bHYPRE_PreconditionedSolver self,
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
struct bHYPRE_PreconditionedSolver__object*
bHYPRE_PreconditionedSolver__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_PreconditionedSolver__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_PreconditionedSolver__exec(
  /* in */ bHYPRE_PreconditionedSolver self,
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
bHYPRE_PreconditionedSolver__getURL(
  /* in */ bHYPRE_PreconditionedSolver self,
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
bHYPRE_PreconditionedSolver__raddRef(
  /* in */ bHYPRE_PreconditionedSolver self,
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
bHYPRE_PreconditionedSolver__isRemote(
  /* in */ bHYPRE_PreconditionedSolver self,
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
bHYPRE_PreconditionedSolver__isLocal(
  /* in */ bHYPRE_PreconditionedSolver self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_create1d(int32_t len);

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_create1dInit(
  int32_t len, 
  bHYPRE_PreconditionedSolver* data);

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_borrow(
  bHYPRE_PreconditionedSolver* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_smartCopy(
  struct bHYPRE_PreconditionedSolver__array *array);

void
bHYPRE_PreconditionedSolver__array_addRef(
  struct bHYPRE_PreconditionedSolver__array* array);

void
bHYPRE_PreconditionedSolver__array_deleteRef(
  struct bHYPRE_PreconditionedSolver__array* array);

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get1(
  const struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t i1);

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get2(
  const struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get3(
  const struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get4(
  const struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get5(
  const struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get6(
  const struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get7(
  const struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get(
  const struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t indices[]);

void
bHYPRE_PreconditionedSolver__array_set1(
  struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t i1,
  bHYPRE_PreconditionedSolver const value);

void
bHYPRE_PreconditionedSolver__array_set2(
  struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_PreconditionedSolver const value);

void
bHYPRE_PreconditionedSolver__array_set3(
  struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_PreconditionedSolver const value);

void
bHYPRE_PreconditionedSolver__array_set4(
  struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_PreconditionedSolver const value);

void
bHYPRE_PreconditionedSolver__array_set5(
  struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_PreconditionedSolver const value);

void
bHYPRE_PreconditionedSolver__array_set6(
  struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_PreconditionedSolver const value);

void
bHYPRE_PreconditionedSolver__array_set7(
  struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_PreconditionedSolver const value);

void
bHYPRE_PreconditionedSolver__array_set(
  struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t indices[],
  bHYPRE_PreconditionedSolver const value);

int32_t
bHYPRE_PreconditionedSolver__array_dimen(
  const struct bHYPRE_PreconditionedSolver__array* array);

int32_t
bHYPRE_PreconditionedSolver__array_lower(
  const struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t ind);

int32_t
bHYPRE_PreconditionedSolver__array_upper(
  const struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t ind);

int32_t
bHYPRE_PreconditionedSolver__array_length(
  const struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t ind);

int32_t
bHYPRE_PreconditionedSolver__array_stride(
  const struct bHYPRE_PreconditionedSolver__array* array,
  const int32_t ind);

int
bHYPRE_PreconditionedSolver__array_isColumnOrder(
  const struct bHYPRE_PreconditionedSolver__array* array);

int
bHYPRE_PreconditionedSolver__array_isRowOrder(
  const struct bHYPRE_PreconditionedSolver__array* array);

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_slice(
  struct bHYPRE_PreconditionedSolver__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_PreconditionedSolver__array_copy(
  const struct bHYPRE_PreconditionedSolver__array* src,
  struct bHYPRE_PreconditionedSolver__array* dest);

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_ensure(
  struct bHYPRE_PreconditionedSolver__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_PreconditionedSolver__connectI

#pragma weak bHYPRE_PreconditionedSolver__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_PreconditionedSolver__object*
bHYPRE_PreconditionedSolver__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_PreconditionedSolver__object*
bHYPRE_PreconditionedSolver__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
