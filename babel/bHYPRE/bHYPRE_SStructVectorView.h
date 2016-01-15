/*
 * File:          bHYPRE_SStructVectorView.h
 * Symbol:        bHYPRE.SStructVectorView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.4
 * Description:   Client-side glue code for bHYPRE.SStructVectorView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructVectorView_h
#define included_bHYPRE_SStructVectorView_h

/**
 * Symbol "bHYPRE.SStructVectorView" (version 1.0.0)
 */
struct bHYPRE_SStructVectorView__object;
struct bHYPRE_SStructVectorView__array;
typedef struct bHYPRE_SStructVectorView__object* bHYPRE_SStructVectorView;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_SStructGrid_h
#include "bHYPRE_SStructGrid.h"
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
#ifndef included_bHYPRE_SStructVectorView_IOR_h
#include "bHYPRE_SStructVectorView_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__connect(const char *, sidl_BaseInterface *_ex);

/**
 * Set the vector grid.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVectorView_SetGrid(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ bHYPRE_SStructGrid grid,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_SetGrid)(
    self->d_object,
    grid,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set vector coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVectorView_SetValues(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  int32_t index_lower[1], index_upper[1], index_stride[1]; 
  struct sidl_int__array index_real;
  struct sidl_int__array*index_tmp = &index_real;
  index_upper[0] = dim-1;
  sidl_int__array_init(index, index_tmp, 1, index_lower, index_upper, 
    index_stride);
  _result = (*self->d_epv->f_SetValues)(
    self->d_object,
    part,
    index_tmp,
    var,
    value,
    _ex);
#ifdef SIDL_DEBUG_REFCOUNT
  sidl__array_deleteRef((struct sidl__array*)index_tmp);
#endif /* SIDL_DEBUG_REFCOUNT */
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set vector coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */
int32_t
bHYPRE_SStructVectorView_SetBoxValues(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Set vector coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVectorView_AddToValues(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  int32_t index_lower[1], index_upper[1], index_stride[1]; 
  struct sidl_int__array index_real;
  struct sidl_int__array*index_tmp = &index_real;
  index_upper[0] = dim-1;
  sidl_int__array_init(index, index_tmp, 1, index_lower, index_upper, 
    index_stride);
  _result = (*self->d_epv->f_AddToValues)(
    self->d_object,
    part,
    index_tmp,
    var,
    value,
    _ex);
#ifdef SIDL_DEBUG_REFCOUNT
  sidl__array_deleteRef((struct sidl__array*)index_tmp);
#endif /* SIDL_DEBUG_REFCOUNT */
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set vector coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */
int32_t
bHYPRE_SStructVectorView_AddToBoxValues(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Gather vector data before calling {\tt GetValues}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVectorView_Gather(
  /* in */ bHYPRE_SStructVectorView self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_Gather)(
    self->d_object,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Get vector coefficients index by index.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVectorView_GetValues(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  int32_t index_lower[1], index_upper[1], index_stride[1]; 
  struct sidl_int__array index_real;
  struct sidl_int__array*index_tmp = &index_real;
  index_upper[0] = dim-1;
  sidl_int__array_init(index, index_tmp, 1, index_lower, index_upper, 
    index_stride);
  _result = (*self->d_epv->f_GetValues)(
    self->d_object,
    part,
    index_tmp,
    var,
    value,
    _ex);
#ifdef SIDL_DEBUG_REFCOUNT
  sidl__array_deleteRef((struct sidl__array*)index_tmp);
#endif /* SIDL_DEBUG_REFCOUNT */
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Get vector coefficients a box at a time.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */
int32_t
bHYPRE_SStructVectorView_GetBoxValues(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* inout rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Set the vector to be complex.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVectorView_SetComplex(
  /* in */ bHYPRE_SStructVectorView self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_SetComplex)(
    self->d_object,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVectorView_Print(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ const char* filename,
  /* in */ int32_t all,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_Print)(
    self->d_object,
    filename,
    all,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVectorView_GetObject(
  /* in */ bHYPRE_SStructVectorView self,
  /* out */ sidl_BaseInterface* A,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_GetObject)(
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
bHYPRE_SStructVectorView_SetCommunicator(
  /* in */ bHYPRE_SStructVectorView self,
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
bHYPRE_SStructVectorView_Destroy(
  /* in */ bHYPRE_SStructVectorView self,
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
bHYPRE_SStructVectorView_Initialize(
  /* in */ bHYPRE_SStructVectorView self,
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
bHYPRE_SStructVectorView_Assemble(
  /* in */ bHYPRE_SStructVectorView self,
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
bHYPRE_SStructVectorView_addRef(
  /* in */ bHYPRE_SStructVectorView self,
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
bHYPRE_SStructVectorView_deleteRef(
  /* in */ bHYPRE_SStructVectorView self,
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
bHYPRE_SStructVectorView_isSame(
  /* in */ bHYPRE_SStructVectorView self,
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
bHYPRE_SStructVectorView_isType(
  /* in */ bHYPRE_SStructVectorView self,
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
bHYPRE_SStructVectorView_getClassInfo(
  /* in */ bHYPRE_SStructVectorView self,
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
struct bHYPRE_SStructVectorView__object*
bHYPRE_SStructVectorView__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructVectorView__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_SStructVectorView__exec(
  /* in */ bHYPRE_SStructVectorView self,
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
bHYPRE_SStructVectorView__getURL(
  /* in */ bHYPRE_SStructVectorView self,
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
bHYPRE_SStructVectorView__raddRef(
  /* in */ bHYPRE_SStructVectorView self,
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
bHYPRE_SStructVectorView__isRemote(
  /* in */ bHYPRE_SStructVectorView self,
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
bHYPRE_SStructVectorView__isLocal(
  /* in */ bHYPRE_SStructVectorView self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_create1d(int32_t len);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructVectorView* data);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_borrow(
  bHYPRE_SStructVectorView* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_smartCopy(
  struct bHYPRE_SStructVectorView__array *array);

void
bHYPRE_SStructVectorView__array_addRef(
  struct bHYPRE_SStructVectorView__array* array);

void
bHYPRE_SStructVectorView__array_deleteRef(
  struct bHYPRE_SStructVectorView__array* array);

bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__array_get1(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1);

bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__array_get2(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__array_get3(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__array_get4(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__array_get5(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__array_get6(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__array_get7(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__array_get(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t indices[]);

void
bHYPRE_SStructVectorView__array_set1(
  struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  bHYPRE_SStructVectorView const value);

void
bHYPRE_SStructVectorView__array_set2(
  struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructVectorView const value);

void
bHYPRE_SStructVectorView__array_set3(
  struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructVectorView const value);

void
bHYPRE_SStructVectorView__array_set4(
  struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructVectorView const value);

void
bHYPRE_SStructVectorView__array_set5(
  struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructVectorView const value);

void
bHYPRE_SStructVectorView__array_set6(
  struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructVectorView const value);

void
bHYPRE_SStructVectorView__array_set7(
  struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructVectorView const value);

void
bHYPRE_SStructVectorView__array_set(
  struct bHYPRE_SStructVectorView__array* array,
  const int32_t indices[],
  bHYPRE_SStructVectorView const value);

int32_t
bHYPRE_SStructVectorView__array_dimen(
  const struct bHYPRE_SStructVectorView__array* array);

int32_t
bHYPRE_SStructVectorView__array_lower(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructVectorView__array_upper(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructVectorView__array_length(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructVectorView__array_stride(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t ind);

int
bHYPRE_SStructVectorView__array_isColumnOrder(
  const struct bHYPRE_SStructVectorView__array* array);

int
bHYPRE_SStructVectorView__array_isRowOrder(
  const struct bHYPRE_SStructVectorView__array* array);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_slice(
  struct bHYPRE_SStructVectorView__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_SStructVectorView__array_copy(
  const struct bHYPRE_SStructVectorView__array* src,
  struct bHYPRE_SStructVectorView__array* dest);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_ensure(
  struct bHYPRE_SStructVectorView__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_SStructVectorView__connectI

#pragma weak bHYPRE_SStructVectorView__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructVectorView__object*
bHYPRE_SStructVectorView__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructVectorView__object*
bHYPRE_SStructVectorView__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
