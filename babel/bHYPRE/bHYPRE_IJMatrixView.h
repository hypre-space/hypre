/*
 * File:          bHYPRE_IJMatrixView.h
 * Symbol:        bHYPRE.IJMatrixView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.12
 * Description:   Client-side glue code for bHYPRE.IJMatrixView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_IJMatrixView_h
#define included_bHYPRE_IJMatrixView_h

/**
 * Symbol "bHYPRE.IJMatrixView" (version 1.0.0)
 * 
 * This interface represents a linear-algebraic conceptual view of a
 * linear system.  The 'I' and 'J' in the name are meant to be
 * mnemonic for the traditional matrix notation A(I,J).
 * 
 */
struct bHYPRE_IJMatrixView__object;
struct bHYPRE_IJMatrixView__array;
typedef struct bHYPRE_IJMatrixView__object* bHYPRE_IJMatrixView;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
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
bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_IJMatrixView_addRef(
  /* in */ bHYPRE_IJMatrixView self);

void
bHYPRE_IJMatrixView_deleteRef(
  /* in */ bHYPRE_IJMatrixView self);

sidl_bool
bHYPRE_IJMatrixView_isSame(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_IJMatrixView_queryInt(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ const char* name);

sidl_bool
bHYPRE_IJMatrixView_isType(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_IJMatrixView_getClassInfo(
  /* in */ bHYPRE_IJMatrixView self);

int32_t
bHYPRE_IJMatrixView_SetCommunicator(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

int32_t
bHYPRE_IJMatrixView_Initialize(
  /* in */ bHYPRE_IJMatrixView self);

int32_t
bHYPRE_IJMatrixView_Assemble(
  /* in */ bHYPRE_IJMatrixView self);

/**
 * Set the local range for a matrix object.  Each process owns
 * some unique consecutive range of rows, indicated by the
 * global row indices {\tt ilower} and {\tt iupper}.  The row
 * data is required to be such that the value of {\tt ilower} on
 * any process $p$ be exactly one more than the value of {\tt
 * iupper} on process $p-1$.  Note that the first row of the
 * global matrix may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 * 
 * For square matrices, {\tt jlower} and {\tt jupper} typically
 * should match {\tt ilower} and {\tt iupper}, respectively.
 * For rectangular matrices, {\tt jlower} and {\tt jupper}
 * should define a partitioning of the columns.  This
 * partitioning must be used for any vector $v$ that will be
 * used in matrix-vector products with the rectangular matrix.
 * The matrix data structure may use {\tt jlower} and {\tt
 * jupper} to store the diagonal blocks (rectangular in general)
 * of the matrix separately from the rest of the matrix.
 * 
 * Collective.
 * 
 */
int32_t
bHYPRE_IJMatrixView_SetLocalRange(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper);

/**
 * Sets values for {\tt nrows} of the matrix.  The arrays {\tt
 * ncols} and {\tt rows} are of dimension {\tt nrows} and
 * contain the number of columns in each row and the row
 * indices, respectively.  The array {\tt cols} contains the
 * column indices for each of the {\tt rows}, and is ordered by
 * rows.  The data in the {\tt values} array corresponds
 * directly to the column entries in {\tt cols}.  The last argument
 * is the size of the cols and values arrays, i.e. the total number
 * of nonzeros being provided, i.e. the sum of all values in ncols.
 * This functin erases any previous values at the specified locations and
 * replaces them with new ones, or, if there was no value there before,
 * inserts a new one.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJMatrixView_SetValues(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* ncols,
  /* in rarray[nrows] */ int32_t* rows,
  /* in rarray[nnonzeros] */ int32_t* cols,
  /* in rarray[nnonzeros] */ double* values,
  /* in */ int32_t nnonzeros);

/**
 * Adds to values for {\tt nrows} of the matrix.  Usage details
 * are analogous to {\tt SetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value
 * there before, inserts a new one.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJMatrixView_AddToValues(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* ncols,
  /* in rarray[nrows] */ int32_t* rows,
  /* in rarray[nnonzeros] */ int32_t* cols,
  /* in rarray[nnonzeros] */ double* values,
  /* in */ int32_t nnonzeros);

/**
 * Gets range of rows owned by this processor and range of
 * column partitioning for this processor.
 * 
 */
int32_t
bHYPRE_IJMatrixView_GetLocalRange(
  /* in */ bHYPRE_IJMatrixView self,
  /* out */ int32_t* ilower,
  /* out */ int32_t* iupper,
  /* out */ int32_t* jlower,
  /* out */ int32_t* jupper);

/**
 * Gets number of nonzeros elements for {\tt nrows} rows
 * specified in {\tt rows} and returns them in {\tt ncols},
 * which needs to be allocated by the user.
 * 
 */
int32_t
bHYPRE_IJMatrixView_GetRowCounts(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* rows,
  /* inout rarray[nrows] */ int32_t* ncols);

/**
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 * 
 */
int32_t
bHYPRE_IJMatrixView_GetValues(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* ncols,
  /* in rarray[nrows] */ int32_t* rows,
  /* in rarray[nnonzeros] */ int32_t* cols,
  /* inout rarray[nnonzeros] */ double* values,
  /* in */ int32_t nnonzeros);

/**
 * (Optional) Set the max number of nonzeros to expect in each
 * row.  The array {\tt sizes} contains estimated sizes for each
 * row on this process.  The integer nrows is the number of rows in
 * the local matrix.  This call can significantly improve the
 * efficiency of matrix construction, and should always be
 * utilized if possible.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJMatrixView_SetRowSizes(
  /* in */ bHYPRE_IJMatrixView self,
  /* in rarray[nrows] */ int32_t* sizes,
  /* in */ int32_t nrows);

/**
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_IJMatrixView_Print(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ const char* filename);

/**
 * Read the matrix from file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_IJMatrixView_Read(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ const char* filename,
  /* in */ bHYPRE_MPICommunicator comm);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_IJMatrixView__object*
bHYPRE_IJMatrixView__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_IJMatrixView__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_IJMatrixView__exec(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_IJMatrixView__getURL(
  /* in */ bHYPRE_IJMatrixView self);
struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_create1d(int32_t len);

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_create1dInit(
  int32_t len, 
  bHYPRE_IJMatrixView* data);

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_borrow(
  bHYPRE_IJMatrixView* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_smartCopy(
  struct bHYPRE_IJMatrixView__array *array);

void
bHYPRE_IJMatrixView__array_addRef(
  struct bHYPRE_IJMatrixView__array* array);

void
bHYPRE_IJMatrixView__array_deleteRef(
  struct bHYPRE_IJMatrixView__array* array);

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__array_get1(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1);

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__array_get2(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__array_get3(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__array_get4(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__array_get5(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__array_get6(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__array_get7(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__array_get(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t indices[]);

void
bHYPRE_IJMatrixView__array_set1(
  struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  bHYPRE_IJMatrixView const value);

void
bHYPRE_IJMatrixView__array_set2(
  struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_IJMatrixView const value);

void
bHYPRE_IJMatrixView__array_set3(
  struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_IJMatrixView const value);

void
bHYPRE_IJMatrixView__array_set4(
  struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_IJMatrixView const value);

void
bHYPRE_IJMatrixView__array_set5(
  struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_IJMatrixView const value);

void
bHYPRE_IJMatrixView__array_set6(
  struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_IJMatrixView const value);

void
bHYPRE_IJMatrixView__array_set7(
  struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_IJMatrixView const value);

void
bHYPRE_IJMatrixView__array_set(
  struct bHYPRE_IJMatrixView__array* array,
  const int32_t indices[],
  bHYPRE_IJMatrixView const value);

int32_t
bHYPRE_IJMatrixView__array_dimen(
  const struct bHYPRE_IJMatrixView__array* array);

int32_t
bHYPRE_IJMatrixView__array_lower(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t ind);

int32_t
bHYPRE_IJMatrixView__array_upper(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t ind);

int32_t
bHYPRE_IJMatrixView__array_length(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t ind);

int32_t
bHYPRE_IJMatrixView__array_stride(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t ind);

int
bHYPRE_IJMatrixView__array_isColumnOrder(
  const struct bHYPRE_IJMatrixView__array* array);

int
bHYPRE_IJMatrixView__array_isRowOrder(
  const struct bHYPRE_IJMatrixView__array* array);

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_slice(
  struct bHYPRE_IJMatrixView__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_IJMatrixView__array_copy(
  const struct bHYPRE_IJMatrixView__array* src,
  struct bHYPRE_IJMatrixView__array* dest);

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_ensure(
  struct bHYPRE_IJMatrixView__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
