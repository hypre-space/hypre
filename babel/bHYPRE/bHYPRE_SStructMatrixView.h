/*
 * File:          bHYPRE_SStructMatrixView.h
 * Symbol:        bHYPRE.SStructMatrixView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.12
 * Description:   Client-side glue code for bHYPRE.SStructMatrixView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
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
bHYPRE_SStructMatrixView
bHYPRE_SStructMatrixView__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_SStructMatrixView_addRef(
  /* in */ bHYPRE_SStructMatrixView self);

void
bHYPRE_SStructMatrixView_deleteRef(
  /* in */ bHYPRE_SStructMatrixView self);

sidl_bool
bHYPRE_SStructMatrixView_isSame(
  /* in */ bHYPRE_SStructMatrixView self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_SStructMatrixView_queryInt(
  /* in */ bHYPRE_SStructMatrixView self,
  /* in */ const char* name);

sidl_bool
bHYPRE_SStructMatrixView_isType(
  /* in */ bHYPRE_SStructMatrixView self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_SStructMatrixView_getClassInfo(
  /* in */ bHYPRE_SStructMatrixView self);

int32_t
bHYPRE_SStructMatrixView_SetCommunicator(
  /* in */ bHYPRE_SStructMatrixView self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

int32_t
bHYPRE_SStructMatrixView_Initialize(
  /* in */ bHYPRE_SStructMatrixView self);

int32_t
bHYPRE_SStructMatrixView_Assemble(
  /* in */ bHYPRE_SStructMatrixView self);

int32_t
bHYPRE_SStructMatrixView_GetObject(
  /* in */ bHYPRE_SStructMatrixView self,
  /* out */ sidl_BaseInterface* A);

/**
 * Set the matrix graph.
 * DEPRECATED     Use Create
 * 
 */
int32_t
bHYPRE_SStructMatrixView_SetGraph(
  /* in */ bHYPRE_SStructMatrixView self,
  /* in */ bHYPRE_SStructGraph graph);

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
 * 
 * 
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
  /* in rarray[nentries] */ double* values);

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
 * 
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
  /* in */ int32_t nvalues);

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
 * 
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
  /* in rarray[nentries] */ double* values);

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
 * 
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
  /* in */ int32_t nvalues);

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
 * 
 */
int32_t
bHYPRE_SStructMatrixView_SetSymmetric(
  /* in */ bHYPRE_SStructMatrixView self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ int32_t to_var,
  /* in */ int32_t symmetric);

/**
 * Define symmetry properties for all non-stencil matrix
 * entries.
 * 
 */
int32_t
bHYPRE_SStructMatrixView_SetNSSymmetric(
  /* in */ bHYPRE_SStructMatrixView self,
  /* in */ int32_t symmetric);

/**
 * Set the matrix to be complex.
 * 
 */
int32_t
bHYPRE_SStructMatrixView_SetComplex(
  /* in */ bHYPRE_SStructMatrixView self);

/**
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_SStructMatrixView_Print(
  /* in */ bHYPRE_SStructMatrixView self,
  /* in */ const char* filename,
  /* in */ int32_t all);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructMatrixView__object*
bHYPRE_SStructMatrixView__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructMatrixView__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_SStructMatrixView__exec(
  /* in */ bHYPRE_SStructMatrixView self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_SStructMatrixView__getURL(
  /* in */ bHYPRE_SStructMatrixView self);
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

#ifdef __cplusplus
}
#endif
#endif
