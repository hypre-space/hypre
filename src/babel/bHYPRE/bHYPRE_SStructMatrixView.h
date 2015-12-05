/*
 * File:          bHYPRE_SStructMatrixView.h
 * Symbol:        bHYPRE.SStructMatrixView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.SStructMatrixView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructMatrixView_h
#define included_bHYPRE_SStructMatrixView_h

/**
 * Symbol "bHYPRE.SStructMatrixView" (version 1.0.0)
 */
struct bHYPRE_SStructMatrixView__object;
struct bHYPRE_SStructMatrixView__array;
typedef struct bHYPRE_SStructMatrixView__object* bHYPRE_SStructMatrixView;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_SStructGraph_h
#include "bHYPRE_SStructGraph.h"
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
#ifndef included_bHYPRE_SStructMatrixView_IOR_h
#include "bHYPRE_SStructMatrixView_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_SStructMatrixView
bHYPRE_SStructMatrixView__connect(const char *, sidl_BaseInterface *_ex);

/**
 * Set the matrix graph.
 * DEPRECATED     Use Create
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructMatrixView_SetGraph(
  /* in */ bHYPRE_SStructMatrixView self,
  /* in */ bHYPRE_SStructGraph graph,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetGraph)(
    self->d_object,
    graph,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */
int32_t
bHYPRE_SStructMatrixView_SetValues(
  /* in */ bHYPRE_SStructMatrixView self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nentries] */ double* values,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Set matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */
int32_t
bHYPRE_SStructMatrixView_SetBoxValues(
  /* in */ bHYPRE_SStructMatrixView self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Add to matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */
int32_t
bHYPRE_SStructMatrixView_AddToValues(
  /* in */ bHYPRE_SStructMatrixView self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nentries] */ double* values,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Add to matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of stencil
 * type.  Also, they must all represent couplings to the same
 * variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */
int32_t
bHYPRE_SStructMatrixView_AddToBoxValues(
  /* in */ bHYPRE_SStructMatrixView self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Define symmetry properties for the stencil entries in the
 * matrix.  The boolean argument {\tt symmetric} is applied to
 * stencil entries on part {\tt part} that couple variable {\tt
 * var} to variable {\tt to\_var}.  A value of -1 may be used
 * for {\tt part}, {\tt var}, or {\tt to\_var} to specify
 * ``all''.  For example, if {\tt part} and {\tt to\_var} are
 * set to -1, then the boolean is applied to stencil entries on
 * all parts that couple variable {\tt var} to all other
 * variables.
 * 
 * By default, matrices are assumed to be nonsymmetric.
 * Significant storage savings can be made if the matrix is
 * symmetric.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructMatrixView_SetSymmetric(
  /* in */ bHYPRE_SStructMatrixView self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ int32_t to_var,
  /* in */ int32_t symmetric,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetSymmetric)(
    self->d_object,
    part,
    var,
    to_var,
    symmetric,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Define symmetry properties for all non-stencil matrix
 * entries.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructMatrixView_SetNSSymmetric(
  /* in */ bHYPRE_SStructMatrixView self,
  /* in */ int32_t symmetric,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetNSSymmetric)(
    self->d_object,
    symmetric,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the matrix to be complex.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructMatrixView_SetComplex(
  /* in */ bHYPRE_SStructMatrixView self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetComplex)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructMatrixView_Print(
  /* in */ bHYPRE_SStructMatrixView self,
  /* in */ const char* filename,
  /* in */ int32_t all,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Print)(
    self->d_object,
    filename,
    all,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructMatrixView_GetObject(
  /* in */ bHYPRE_SStructMatrixView self,
  /* out */ sidl_BaseInterface* A,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_GetObject)(
    self->d_object,
    A,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructMatrixView_SetCommunicator(
  /* in */ bHYPRE_SStructMatrixView self,
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


SIDL_C_INLINE_DECL
void
bHYPRE_SStructMatrixView_Destroy(
  /* in */ bHYPRE_SStructMatrixView self,
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
bHYPRE_SStructMatrixView_Initialize(
  /* in */ bHYPRE_SStructMatrixView self,
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


SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructMatrixView_Assemble(
  /* in */ bHYPRE_SStructMatrixView self,
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
bHYPRE_SStructMatrixView_addRef(
  /* in */ bHYPRE_SStructMatrixView self,
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
bHYPRE_SStructMatrixView_deleteRef(
  /* in */ bHYPRE_SStructMatrixView self,
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
bHYPRE_SStructMatrixView_isSame(
  /* in */ bHYPRE_SStructMatrixView self,
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
bHYPRE_SStructMatrixView_isType(
  /* in */ bHYPRE_SStructMatrixView self,
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
bHYPRE_SStructMatrixView_getClassInfo(
  /* in */ bHYPRE_SStructMatrixView self,
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
struct bHYPRE_SStructMatrixView__object*
bHYPRE_SStructMatrixView__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructMatrixView__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_SStructMatrixView__exec(
  /* in */ bHYPRE_SStructMatrixView self,
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
bHYPRE_SStructMatrixView__getURL(
  /* in */ bHYPRE_SStructMatrixView self,
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
bHYPRE_SStructMatrixView__raddRef(
  /* in */ bHYPRE_SStructMatrixView self,
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
bHYPRE_SStructMatrixView__isRemote(
  /* in */ bHYPRE_SStructMatrixView self,
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
bHYPRE_SStructMatrixView__isLocal(
  /* in */ bHYPRE_SStructMatrixView self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_SStructMatrixView__array*
bHYPRE_SStructMatrixView__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructMatrixView__array*
bHYPRE_SStructMatrixView__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructMatrixView__array*
bHYPRE_SStructMatrixView__array_create1d(int32_t len);

struct bHYPRE_SStructMatrixView__array*
bHYPRE_SStructMatrixView__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructMatrixView* data);

struct bHYPRE_SStructMatrixView__array*
bHYPRE_SStructMatrixView__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructMatrixView__array*
bHYPRE_SStructMatrixView__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructMatrixView__array*
bHYPRE_SStructMatrixView__array_borrow(
  bHYPRE_SStructMatrixView* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_SStructMatrixView__array*
bHYPRE_SStructMatrixView__array_smartCopy(
  struct bHYPRE_SStructMatrixView__array *array);

void
bHYPRE_SStructMatrixView__array_addRef(
  struct bHYPRE_SStructMatrixView__array* array);

void
bHYPRE_SStructMatrixView__array_deleteRef(
  struct bHYPRE_SStructMatrixView__array* array);

bHYPRE_SStructMatrixView
bHYPRE_SStructMatrixView__array_get1(
  const struct bHYPRE_SStructMatrixView__array* array,
  const int32_t i1);

bHYPRE_SStructMatrixView
bHYPRE_SStructMatrixView__array_get2(
  const struct bHYPRE_SStructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_SStructMatrixView
bHYPRE_SStructMatrixView__array_get3(
  const struct bHYPRE_SStructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_SStructMatrixView
bHYPRE_SStructMatrixView__array_get4(
  const struct bHYPRE_SStructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_SStructMatrixView
bHYPRE_SStructMatrixView__array_get5(
  const struct bHYPRE_SStructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_SStructMatrixView
bHYPRE_SStructMatrixView__array_get6(
  const struct bHYPRE_SStructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_SStructMatrixView
bHYPRE_SStructMatrixView__array_get7(
  const struct bHYPRE_SStructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_SStructMatrixView
bHYPRE_SStructMatrixView__array_get(
  const struct bHYPRE_SStructMatrixView__array* array,
  const int32_t indices[]);

void
bHYPRE_SStructMatrixView__array_set1(
  struct bHYPRE_SStructMatrixView__array* array,
  const int32_t i1,
  bHYPRE_SStructMatrixView const value);

void
bHYPRE_SStructMatrixView__array_set2(
  struct bHYPRE_SStructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructMatrixView const value);

void
bHYPRE_SStructMatrixView__array_set3(
  struct bHYPRE_SStructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructMatrixView const value);

void
bHYPRE_SStructMatrixView__array_set4(
  struct bHYPRE_SStructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructMatrixView const value);

void
bHYPRE_SStructMatrixView__array_set5(
  struct bHYPRE_SStructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructMatrixView const value);

void
bHYPRE_SStructMatrixView__array_set6(
  struct bHYPRE_SStructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructMatrixView const value);

void
bHYPRE_SStructMatrixView__array_set7(
  struct bHYPRE_SStructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructMatrixView const value);

void
bHYPRE_SStructMatrixView__array_set(
  struct bHYPRE_SStructMatrixView__array* array,
  const int32_t indices[],
  bHYPRE_SStructMatrixView const value);

int32_t
bHYPRE_SStructMatrixView__array_dimen(
  const struct bHYPRE_SStructMatrixView__array* array);

int32_t
bHYPRE_SStructMatrixView__array_lower(
  const struct bHYPRE_SStructMatrixView__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructMatrixView__array_upper(
  const struct bHYPRE_SStructMatrixView__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructMatrixView__array_length(
  const struct bHYPRE_SStructMatrixView__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructMatrixView__array_stride(
  const struct bHYPRE_SStructMatrixView__array* array,
  const int32_t ind);

int
bHYPRE_SStructMatrixView__array_isColumnOrder(
  const struct bHYPRE_SStructMatrixView__array* array);

int
bHYPRE_SStructMatrixView__array_isRowOrder(
  const struct bHYPRE_SStructMatrixView__array* array);

struct bHYPRE_SStructMatrixView__array*
bHYPRE_SStructMatrixView__array_slice(
  struct bHYPRE_SStructMatrixView__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_SStructMatrixView__array_copy(
  const struct bHYPRE_SStructMatrixView__array* src,
  struct bHYPRE_SStructMatrixView__array* dest);

struct bHYPRE_SStructMatrixView__array*
bHYPRE_SStructMatrixView__array_ensure(
  struct bHYPRE_SStructMatrixView__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_SStructMatrixView__connectI

#pragma weak bHYPRE_SStructMatrixView__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructMatrixView__object*
bHYPRE_SStructMatrixView__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructMatrixView__object*
bHYPRE_SStructMatrixView__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
