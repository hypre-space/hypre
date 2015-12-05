/*
 * File:          bHYPRE_SStructGrid.h
 * Symbol:        bHYPRE.SStructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.SStructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructGrid_h
#define included_bHYPRE_SStructGrid_h

/**
 * Symbol "bHYPRE.SStructGrid" (version 1.0.0)
 * 
 * The semi-structured grid class.
 */
struct bHYPRE_SStructGrid__object;
struct bHYPRE_SStructGrid__array;
typedef struct bHYPRE_SStructGrid__object* bHYPRE_SStructGrid;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_SStructVariable_h
#include "bHYPRE_SStructVariable.h"
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
#ifndef included_bHYPRE_SStructGrid_IOR_h
#include "bHYPRE_SStructGrid_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
struct bHYPRE_SStructGrid__object*
bHYPRE_SStructGrid__create(sidl_BaseInterface* _ex);

/**
 * RMI constructor function for the class.
 */
bHYPRE_SStructGrid
bHYPRE_SStructGrid__createRemote(const char * url, sidl_BaseInterface *_ex);

/**
 * Wraps up the private data struct pointer (struct bHYPRE\_SStructGrid\_\_data) passed in rather than running the constructor.
 */
bHYPRE_SStructGrid
bHYPRE_SStructGrid__wrapObj(void * data, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_SStructGrid
bHYPRE_SStructGrid__connect(const char *, sidl_BaseInterface *_ex);

/**
 *  This function is the preferred way to create a SStruct Grid. 
 */
bHYPRE_SStructGrid
bHYPRE_SStructGrid_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t ndim,
  /* in */ int32_t nparts,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Method:  SetNumDimParts[]
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructGrid_SetNumDimParts(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t ndim,
  /* in */ int32_t nparts,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetNumDimParts)(
    self,
    ndim,
    nparts,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  SetCommunicator[]
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructGrid_SetCommunicator(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid_Destroy(
  /* in */ bHYPRE_SStructGrid self,
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
 * Set the extents for a box on a structured part of the grid.
 */
int32_t
bHYPRE_SStructGrid_SetExtents(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Describe the variables that live on a structured part of the
 * grid.  Input: part number, variable number, total number of
 * variables on that part (needed for memory allocation),
 * variable type.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructGrid_SetVariable(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ int32_t nvars,
  /* in */ enum bHYPRE_SStructVariable__enum vartype,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetVariable)(
    self,
    part,
    var,
    nvars,
    vartype,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Describe additional variables that live at a particular
 * index.  These variables are appended to the array of
 * variables set in {\tt SetVariables}, and are referenced as
 * such.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructGrid_AddVariable(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ enum bHYPRE_SStructVariable__enum vartype,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t index_lower[1], index_upper[1], index_stride[1]; 
  struct sidl_int__array index_real;
  struct sidl_int__array*index_tmp = &index_real;
  index_upper[0] = dim-1;
  sidl_int__array_init(index, index_tmp, 1, index_lower, index_upper,
    index_stride);
  return (*self->d_epv->f_AddVariable)(
    self,
    part,
    index_tmp,
    var,
    vartype,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Describe how regions just outside of a part relate to other
 * parts.  This is done a box at a time.
 * 
 * The indexes {\tt ilower} and {\tt iupper} map directly to the
 * indexes {\tt nbor\_ilower} and {\tt nbor\_iupper}.  Although,
 * it is required that indexes increase from {\tt ilower} to
 * {\tt iupper}, indexes may increase and/or decrease from {\tt
 * nbor\_ilower} to {\tt nbor\_iupper}.
 * 
 * The {\tt index\_map} describes the mapping of indexes 0, 1,
 * and 2 on part {\tt part} to the corresponding indexes on part
 * {\tt nbor\_part}.  For example, triple (1, 2, 0) means that
 * indexes 0, 1, and 2 on part {\tt part} map to indexes 1, 2,
 * and 0 on part {\tt nbor\_part}, respectively.
 * 
 * NOTE: All parts related to each other via this routine must
 * have an identical list of variables and variable types.  For
 * example, if part 0 has only two variables on it, a cell
 * centered variable and a node centered variable, and we
 * declare part 1 to be a neighbor of part 0, then part 1 must
 * also have only two variables on it, and they must be of type
 * cell and node.
 */
int32_t
bHYPRE_SStructGrid_SetNeighborBox(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t nbor_part,
  /* in rarray[dim] */ int32_t* nbor_ilower,
  /* in rarray[dim] */ int32_t* nbor_iupper,
  /* in rarray[dim] */ int32_t* index_map,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Add an unstructured part to the grid.  The variables in the
 * unstructured part of the grid are referenced by a global rank
 * between 0 and the total number of unstructured variables
 * minus one.  Each process owns some unique consecutive range
 * of variables, defined by {\tt ilower} and {\tt iupper}.
 * 
 * NOTE: This is just a placeholder.  This part of the interface
 * is not finished.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructGrid_AddUnstructuredPart(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_AddUnstructuredPart)(
    self,
    ilower,
    iupper,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * (Optional) Set periodic for a particular part.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructGrid_SetPeriodic(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* periodic,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t periodic_lower[1], periodic_upper[1], periodic_stride[1]; 
  struct sidl_int__array periodic_real;
  struct sidl_int__array*periodic_tmp = &periodic_real;
  periodic_upper[0] = dim-1;
  sidl_int__array_init(periodic, periodic_tmp, 1, periodic_lower,
    periodic_upper, periodic_stride);
  return (*self->d_epv->f_SetPeriodic)(
    self,
    part,
    periodic_tmp,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Setting ghost in the sgrids.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructGrid_SetNumGhost(
  /* in */ bHYPRE_SStructGrid self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t num_ghost_lower[1], num_ghost_upper[1], num_ghost_stride[1]; 
  struct sidl_int__array num_ghost_real;
  struct sidl_int__array*num_ghost_tmp = &num_ghost_real;
  num_ghost_upper[0] = dim2-1;
  sidl_int__array_init(num_ghost, num_ghost_tmp, 1, num_ghost_lower,
    num_ghost_upper, num_ghost_stride);
  return (*self->d_epv->f_SetNumGhost)(
    self,
    num_ghost_tmp,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 *  final construction of the object before its use 
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructGrid_Assemble(
  /* in */ bHYPRE_SStructGrid self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Assemble)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
void
bHYPRE_SStructGrid_addRef(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid_deleteRef(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid_isSame(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid_isType(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid_getClassInfo(
  /* in */ bHYPRE_SStructGrid self,
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
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructGrid__object*
bHYPRE_SStructGrid__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructGrid__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_SStructGrid__exec(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid__getURL(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid__raddRef(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid__isRemote(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid__isLocal(
  /* in */ bHYPRE_SStructGrid self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_create1d(int32_t len);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructGrid* data);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_borrow(
  bHYPRE_SStructGrid* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_smartCopy(
  struct bHYPRE_SStructGrid__array *array);

void
bHYPRE_SStructGrid__array_addRef(
  struct bHYPRE_SStructGrid__array* array);

void
bHYPRE_SStructGrid__array_deleteRef(
  struct bHYPRE_SStructGrid__array* array);

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get1(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1);

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get2(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get3(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get4(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get5(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get6(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get7(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t indices[]);

void
bHYPRE_SStructGrid__array_set1(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  bHYPRE_SStructGrid const value);

void
bHYPRE_SStructGrid__array_set2(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructGrid const value);

void
bHYPRE_SStructGrid__array_set3(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructGrid const value);

void
bHYPRE_SStructGrid__array_set4(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructGrid const value);

void
bHYPRE_SStructGrid__array_set5(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructGrid const value);

void
bHYPRE_SStructGrid__array_set6(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructGrid const value);

void
bHYPRE_SStructGrid__array_set7(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructGrid const value);

void
bHYPRE_SStructGrid__array_set(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t indices[],
  bHYPRE_SStructGrid const value);

int32_t
bHYPRE_SStructGrid__array_dimen(
  const struct bHYPRE_SStructGrid__array* array);

int32_t
bHYPRE_SStructGrid__array_lower(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructGrid__array_upper(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructGrid__array_length(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructGrid__array_stride(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t ind);

int
bHYPRE_SStructGrid__array_isColumnOrder(
  const struct bHYPRE_SStructGrid__array* array);

int
bHYPRE_SStructGrid__array_isRowOrder(
  const struct bHYPRE_SStructGrid__array* array);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_slice(
  struct bHYPRE_SStructGrid__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_SStructGrid__array_copy(
  const struct bHYPRE_SStructGrid__array* src,
  struct bHYPRE_SStructGrid__array* dest);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_ensure(
  struct bHYPRE_SStructGrid__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_SStructGrid__connectI

#pragma weak bHYPRE_SStructGrid__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructGrid__object*
bHYPRE_SStructGrid__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructGrid__object*
bHYPRE_SStructGrid__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
