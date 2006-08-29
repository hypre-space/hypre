/*
 * File:          bHYPRE_ProblemDefinition.h
 * Symbol:        bHYPRE.ProblemDefinition-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.ProblemDefinition
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_ProblemDefinition_h
#define included_bHYPRE_ProblemDefinition_h

/**
 * Symbol "bHYPRE.ProblemDefinition" (version 1.0.0)
 * 
 * The purpose of a ProblemDefinition is to:
 * 
 * \begin{itemize}
 * \item provide a particular view of how to define a problem
 * \item construct and return a {\it problem object}
 * \end{itemize}
 * 
 * A {\it problem object} is an intentionally vague term that
 * corresponds to any useful object used to define a problem.
 * Prime examples are:
 * 
 * \begin{itemize}
 * \item a LinearOperator object, i.e., something with a matvec
 * \item a MatrixAccess object, i.e., something with a getrow
 * \item a Vector, i.e., something with a dot, axpy, ...
 * \end{itemize}
 * 
 * Note that {\tt Initialize} and {\tt Assemble} are reserved here
 * for defining problem objects through a particular interface.
 */
struct bHYPRE_ProblemDefinition__object;
struct bHYPRE_ProblemDefinition__array;
typedef struct bHYPRE_ProblemDefinition__object* bHYPRE_ProblemDefinition;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
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
#ifndef included_bHYPRE_ProblemDefinition_IOR_h
#include "bHYPRE_ProblemDefinition_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_ProblemDefinition
bHYPRE_ProblemDefinition__connect(const char *, sidl_BaseInterface *_ex);

/**
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_ProblemDefinition_SetCommunicator(
  /* in */ bHYPRE_ProblemDefinition self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetCommunicator)(
    self->d_object,
    mpi_comm,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_ProblemDefinition_Initialize(
  /* in */ bHYPRE_ProblemDefinition self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Initialize)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_ProblemDefinition_Assemble(
  /* in */ bHYPRE_ProblemDefinition self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Assemble)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
void
bHYPRE_ProblemDefinition_addRef(
  /* in */ bHYPRE_ProblemDefinition self,
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
bHYPRE_ProblemDefinition_deleteRef(
  /* in */ bHYPRE_ProblemDefinition self,
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
bHYPRE_ProblemDefinition_isSame(
  /* in */ bHYPRE_ProblemDefinition self,
  /* in */ sidl_BaseInterface iobj,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_isSame)(
    self->d_object,
    iobj,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_bool
bHYPRE_ProblemDefinition_isType(
  /* in */ bHYPRE_ProblemDefinition self,
  /* in */ const char* name,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_isType)(
    self->d_object,
    name,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_ClassInfo
bHYPRE_ProblemDefinition_getClassInfo(
  /* in */ bHYPRE_ProblemDefinition self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_getClassInfo)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_ProblemDefinition__object*
bHYPRE_ProblemDefinition__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_ProblemDefinition__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_ProblemDefinition__exec(
  /* in */ bHYPRE_ProblemDefinition self,
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
bHYPRE_ProblemDefinition__getURL(
  /* in */ bHYPRE_ProblemDefinition self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f__getURL)(
    self->d_object,
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
bHYPRE_ProblemDefinition__raddRef(
  /* in */ bHYPRE_ProblemDefinition self,
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
bHYPRE_ProblemDefinition__isRemote(
  /* in */ bHYPRE_ProblemDefinition self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f__isRemote)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * TRUE if this object is remote, false if local
 */
sidl_bool
bHYPRE_ProblemDefinition__isLocal(
  /* in */ bHYPRE_ProblemDefinition self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_create1d(int32_t len);

struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_create1dInit(
  int32_t len, 
  bHYPRE_ProblemDefinition* data);

struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_borrow(
  bHYPRE_ProblemDefinition* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_smartCopy(
  struct bHYPRE_ProblemDefinition__array *array);

void
bHYPRE_ProblemDefinition__array_addRef(
  struct bHYPRE_ProblemDefinition__array* array);

void
bHYPRE_ProblemDefinition__array_deleteRef(
  struct bHYPRE_ProblemDefinition__array* array);

bHYPRE_ProblemDefinition
bHYPRE_ProblemDefinition__array_get1(
  const struct bHYPRE_ProblemDefinition__array* array,
  const int32_t i1);

bHYPRE_ProblemDefinition
bHYPRE_ProblemDefinition__array_get2(
  const struct bHYPRE_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_ProblemDefinition
bHYPRE_ProblemDefinition__array_get3(
  const struct bHYPRE_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_ProblemDefinition
bHYPRE_ProblemDefinition__array_get4(
  const struct bHYPRE_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_ProblemDefinition
bHYPRE_ProblemDefinition__array_get5(
  const struct bHYPRE_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_ProblemDefinition
bHYPRE_ProblemDefinition__array_get6(
  const struct bHYPRE_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_ProblemDefinition
bHYPRE_ProblemDefinition__array_get7(
  const struct bHYPRE_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_ProblemDefinition
bHYPRE_ProblemDefinition__array_get(
  const struct bHYPRE_ProblemDefinition__array* array,
  const int32_t indices[]);

void
bHYPRE_ProblemDefinition__array_set1(
  struct bHYPRE_ProblemDefinition__array* array,
  const int32_t i1,
  bHYPRE_ProblemDefinition const value);

void
bHYPRE_ProblemDefinition__array_set2(
  struct bHYPRE_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_ProblemDefinition const value);

void
bHYPRE_ProblemDefinition__array_set3(
  struct bHYPRE_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_ProblemDefinition const value);

void
bHYPRE_ProblemDefinition__array_set4(
  struct bHYPRE_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_ProblemDefinition const value);

void
bHYPRE_ProblemDefinition__array_set5(
  struct bHYPRE_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_ProblemDefinition const value);

void
bHYPRE_ProblemDefinition__array_set6(
  struct bHYPRE_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_ProblemDefinition const value);

void
bHYPRE_ProblemDefinition__array_set7(
  struct bHYPRE_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_ProblemDefinition const value);

void
bHYPRE_ProblemDefinition__array_set(
  struct bHYPRE_ProblemDefinition__array* array,
  const int32_t indices[],
  bHYPRE_ProblemDefinition const value);

int32_t
bHYPRE_ProblemDefinition__array_dimen(
  const struct bHYPRE_ProblemDefinition__array* array);

int32_t
bHYPRE_ProblemDefinition__array_lower(
  const struct bHYPRE_ProblemDefinition__array* array,
  const int32_t ind);

int32_t
bHYPRE_ProblemDefinition__array_upper(
  const struct bHYPRE_ProblemDefinition__array* array,
  const int32_t ind);

int32_t
bHYPRE_ProblemDefinition__array_length(
  const struct bHYPRE_ProblemDefinition__array* array,
  const int32_t ind);

int32_t
bHYPRE_ProblemDefinition__array_stride(
  const struct bHYPRE_ProblemDefinition__array* array,
  const int32_t ind);

int
bHYPRE_ProblemDefinition__array_isColumnOrder(
  const struct bHYPRE_ProblemDefinition__array* array);

int
bHYPRE_ProblemDefinition__array_isRowOrder(
  const struct bHYPRE_ProblemDefinition__array* array);

struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_slice(
  struct bHYPRE_ProblemDefinition__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_ProblemDefinition__array_copy(
  const struct bHYPRE_ProblemDefinition__array* src,
  struct bHYPRE_ProblemDefinition__array* dest);

struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_ensure(
  struct bHYPRE_ProblemDefinition__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_ProblemDefinition__connectI

#pragma weak bHYPRE_ProblemDefinition__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_ProblemDefinition__object*
bHYPRE_ProblemDefinition__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_ProblemDefinition__object*
bHYPRE_ProblemDefinition__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
