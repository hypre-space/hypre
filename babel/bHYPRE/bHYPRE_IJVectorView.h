/*
 * File:          bHYPRE_IJVectorView.h
 * Symbol:        bHYPRE.IJVectorView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.4
 * Description:   Client-side glue code for bHYPRE.IJVectorView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_IJVectorView_h
#define included_bHYPRE_IJVectorView_h

/**
 * Symbol "bHYPRE.IJVectorView" (version 1.0.0)
 */
struct bHYPRE_IJVectorView__object;
struct bHYPRE_IJVectorView__array;
typedef struct bHYPRE_IJVectorView__object* bHYPRE_IJVectorView;

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
#ifndef included_bHYPRE_IJVectorView_IOR_h
#include "bHYPRE_IJVectorView_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_IJVectorView
bHYPRE_IJVectorView__connect(const char *, sidl_BaseInterface *_ex);

/**
 * Set the local range for a vector object.  Each process owns
 * some unique consecutive range of vector unknowns, indicated
 * by the global indices {\tt jlower} and {\tt jupper}.  The
 * data is required to be such that the value of {\tt jlower} on
 * any process $p$ be exactly one more than the value of {\tt
 * jupper} on process $p-1$.  Note that the first index of the
 * global vector may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 * 
 * Collective.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_IJVectorView_SetLocalRange(
  /* in */ bHYPRE_IJVectorView self,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_SetLocalRange)(
    self->d_object,
    jlower,
    jupper,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Sets values in vector.  The arrays {\tt values} and {\tt
 * indices} are of dimension {\tt nvalues} and contain the
 * vector values to be set and the corresponding global vector
 * indices, respectively.  Erases any previous values at the
 * specified locations and replaces them with new ones.
 * 
 * Not collective.
 */
int32_t
bHYPRE_IJVectorView_SetValues(
  /* in */ bHYPRE_IJVectorView self,
  /* in */ int32_t nvalues,
  /* in rarray[nvalues] */ int32_t* indices,
  /* in rarray[nvalues] */ double* values,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Adds to values in vector.  Usage details are analogous to
 * {\tt SetValues}.
 * 
 * Not collective.
 */
int32_t
bHYPRE_IJVectorView_AddToValues(
  /* in */ bHYPRE_IJVectorView self,
  /* in */ int32_t nvalues,
  /* in rarray[nvalues] */ int32_t* indices,
  /* in rarray[nvalues] */ double* values,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Returns range of the part of the vector owned by this
 * processor.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_IJVectorView_GetLocalRange(
  /* in */ bHYPRE_IJVectorView self,
  /* out */ int32_t* jlower,
  /* out */ int32_t* jupper,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_GetLocalRange)(
    self->d_object,
    jlower,
    jupper,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Gets values in vector.  Usage details are analogous to {\tt
 * SetValues}.
 * 
 * Not collective.
 */
int32_t
bHYPRE_IJVectorView_GetValues(
  /* in */ bHYPRE_IJVectorView self,
  /* in */ int32_t nvalues,
  /* in rarray[nvalues] */ int32_t* indices,
  /* inout rarray[nvalues] */ double* values,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_IJVectorView_Print(
  /* in */ bHYPRE_IJVectorView self,
  /* in */ const char* filename,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_Print)(
    self->d_object,
    filename,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Read the vector from file.  This is mainly for debugging
 * purposes.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_IJVectorView_Read(
  /* in */ bHYPRE_IJVectorView self,
  /* in */ const char* filename,
  /* in */ bHYPRE_MPICommunicator comm,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_Read)(
    self->d_object,
    filename,
    comm,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
int32_t
bHYPRE_IJVectorView_SetCommunicator(
  /* in */ bHYPRE_IJVectorView self,
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
bHYPRE_IJVectorView_Destroy(
  /* in */ bHYPRE_IJVectorView self,
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
bHYPRE_IJVectorView_Initialize(
  /* in */ bHYPRE_IJVectorView self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_Initialize)(
    self->d_object,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
int32_t
bHYPRE_IJVectorView_Assemble(
  /* in */ bHYPRE_IJVectorView self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_Assemble)(
    self->d_object,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
void
bHYPRE_IJVectorView_addRef(
  /* in */ bHYPRE_IJVectorView self,
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
bHYPRE_IJVectorView_deleteRef(
  /* in */ bHYPRE_IJVectorView self,
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
bHYPRE_IJVectorView_isSame(
  /* in */ bHYPRE_IJVectorView self,
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
bHYPRE_IJVectorView_isType(
  /* in */ bHYPRE_IJVectorView self,
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
bHYPRE_IJVectorView_getClassInfo(
  /* in */ bHYPRE_IJVectorView self,
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
struct bHYPRE_IJVectorView__object*
bHYPRE_IJVectorView__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_IJVectorView__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_IJVectorView__exec(
  /* in */ bHYPRE_IJVectorView self,
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
bHYPRE_IJVectorView__getURL(
  /* in */ bHYPRE_IJVectorView self,
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
bHYPRE_IJVectorView__raddRef(
  /* in */ bHYPRE_IJVectorView self,
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
bHYPRE_IJVectorView__isRemote(
  /* in */ bHYPRE_IJVectorView self,
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
bHYPRE_IJVectorView__isLocal(
  /* in */ bHYPRE_IJVectorView self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_create1d(int32_t len);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_create1dInit(
  int32_t len, 
  bHYPRE_IJVectorView* data);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_borrow(
  bHYPRE_IJVectorView* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_smartCopy(
  struct bHYPRE_IJVectorView__array *array);

void
bHYPRE_IJVectorView__array_addRef(
  struct bHYPRE_IJVectorView__array* array);

void
bHYPRE_IJVectorView__array_deleteRef(
  struct bHYPRE_IJVectorView__array* array);

bHYPRE_IJVectorView
bHYPRE_IJVectorView__array_get1(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t i1);

bHYPRE_IJVectorView
bHYPRE_IJVectorView__array_get2(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_IJVectorView
bHYPRE_IJVectorView__array_get3(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_IJVectorView
bHYPRE_IJVectorView__array_get4(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_IJVectorView
bHYPRE_IJVectorView__array_get5(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_IJVectorView
bHYPRE_IJVectorView__array_get6(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_IJVectorView
bHYPRE_IJVectorView__array_get7(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_IJVectorView
bHYPRE_IJVectorView__array_get(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t indices[]);

void
bHYPRE_IJVectorView__array_set1(
  struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  bHYPRE_IJVectorView const value);

void
bHYPRE_IJVectorView__array_set2(
  struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_IJVectorView const value);

void
bHYPRE_IJVectorView__array_set3(
  struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_IJVectorView const value);

void
bHYPRE_IJVectorView__array_set4(
  struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_IJVectorView const value);

void
bHYPRE_IJVectorView__array_set5(
  struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_IJVectorView const value);

void
bHYPRE_IJVectorView__array_set6(
  struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_IJVectorView const value);

void
bHYPRE_IJVectorView__array_set7(
  struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_IJVectorView const value);

void
bHYPRE_IJVectorView__array_set(
  struct bHYPRE_IJVectorView__array* array,
  const int32_t indices[],
  bHYPRE_IJVectorView const value);

int32_t
bHYPRE_IJVectorView__array_dimen(
  const struct bHYPRE_IJVectorView__array* array);

int32_t
bHYPRE_IJVectorView__array_lower(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_IJVectorView__array_upper(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_IJVectorView__array_length(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_IJVectorView__array_stride(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t ind);

int
bHYPRE_IJVectorView__array_isColumnOrder(
  const struct bHYPRE_IJVectorView__array* array);

int
bHYPRE_IJVectorView__array_isRowOrder(
  const struct bHYPRE_IJVectorView__array* array);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_slice(
  struct bHYPRE_IJVectorView__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_IJVectorView__array_copy(
  const struct bHYPRE_IJVectorView__array* src,
  struct bHYPRE_IJVectorView__array* dest);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_ensure(
  struct bHYPRE_IJVectorView__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_IJVectorView__connectI

#pragma weak bHYPRE_IJVectorView__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_IJVectorView__object*
bHYPRE_IJVectorView__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_IJVectorView__object*
bHYPRE_IJVectorView__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
