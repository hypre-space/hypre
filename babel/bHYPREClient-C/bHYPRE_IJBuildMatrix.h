/*
 * File:          bHYPRE_IJBuildMatrix.h
 * Symbol:        bHYPRE.IJBuildMatrix-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:34 PST
 * Generated:     20030401 14:47:40 PST
 * Description:   Client-side glue code for bHYPRE.IJBuildMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 85
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_IJBuildMatrix_h
#define included_bHYPRE_IJBuildMatrix_h

/**
 * Symbol "bHYPRE.IJBuildMatrix" (version 1.0.0)
 * 
 * This interface represents a linear-algebraic conceptual view of a
 * linear system.  The 'I' and 'J' in the name are meant to be
 * mnemonic for the traditional matrix notation A(I,J).
 * 
 */
struct bHYPRE_IJBuildMatrix__object;
struct bHYPRE_IJBuildMatrix__array;
typedef struct bHYPRE_IJBuildMatrix__object* bHYPRE_IJBuildMatrix;

/*
 * Includes for all header dependencies.
 */

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_SIDL_ClassInfo_h
#include "SIDL_ClassInfo.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_IJBuildMatrix_addRef(
  bHYPRE_IJBuildMatrix self);

void
bHYPRE_IJBuildMatrix_deleteRef(
  bHYPRE_IJBuildMatrix self);

SIDL_bool
bHYPRE_IJBuildMatrix_isSame(
  bHYPRE_IJBuildMatrix self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_IJBuildMatrix_queryInt(
  bHYPRE_IJBuildMatrix self,
  const char* name);

SIDL_bool
bHYPRE_IJBuildMatrix_isType(
  bHYPRE_IJBuildMatrix self,
  const char* name);

SIDL_ClassInfo
bHYPRE_IJBuildMatrix_getClassInfo(
  bHYPRE_IJBuildMatrix self);

int32_t
bHYPRE_IJBuildMatrix_SetCommunicator(
  bHYPRE_IJBuildMatrix self,
  void* mpi_comm);

int32_t
bHYPRE_IJBuildMatrix_Initialize(
  bHYPRE_IJBuildMatrix self);

int32_t
bHYPRE_IJBuildMatrix_Assemble(
  bHYPRE_IJBuildMatrix self);

int32_t
bHYPRE_IJBuildMatrix_GetObject(
  bHYPRE_IJBuildMatrix self,
  SIDL_BaseInterface* A);

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
bHYPRE_IJBuildMatrix_SetLocalRange(
  bHYPRE_IJBuildMatrix self,
  int32_t ilower,
  int32_t iupper,
  int32_t jlower,
  int32_t jupper);

/**
 * Sets values for {\tt nrows} of the matrix.  The arrays {\tt
 * ncols} and {\tt rows} are of dimension {\tt nrows} and
 * contain the number of columns in each row and the row
 * indices, respectively.  The array {\tt cols} contains the
 * column indices for each of the {\tt rows}, and is ordered by
 * rows.  The data in the {\tt values} array corresponds
 * directly to the column entries in {\tt cols}.  Erases any
 * previous values at the specified locations and replaces them
 * with new ones, or, if there was no value there before,
 * inserts a new one.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJBuildMatrix_SetValues(
  bHYPRE_IJBuildMatrix self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values);

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
bHYPRE_IJBuildMatrix_AddToValues(
  bHYPRE_IJBuildMatrix self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values);

/**
 * Gets range of rows owned by this processor and range of
 * column partitioning for this processor.
 * 
 */
int32_t
bHYPRE_IJBuildMatrix_GetLocalRange(
  bHYPRE_IJBuildMatrix self,
  int32_t* ilower,
  int32_t* iupper,
  int32_t* jlower,
  int32_t* jupper);

/**
 * Gets number of nonzeros elements for {\tt nrows} rows
 * specified in {\tt rows} and returns them in {\tt ncols},
 * which needs to be allocated by the user.
 * 
 */
int32_t
bHYPRE_IJBuildMatrix_GetRowCounts(
  bHYPRE_IJBuildMatrix self,
  int32_t nrows,
  struct SIDL_int__array* rows,
  struct SIDL_int__array** ncols);

/**
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 * 
 */
int32_t
bHYPRE_IJBuildMatrix_GetValues(
  bHYPRE_IJBuildMatrix self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array** values);

/**
 * (Optional) Set the max number of nonzeros to expect in each
 * row.  The array {\tt sizes} contains estimated sizes for each
 * row on this process.  This call can significantly improve the
 * efficiency of matrix construction, and should always be
 * utilized if possible.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJBuildMatrix_SetRowSizes(
  bHYPRE_IJBuildMatrix self,
  struct SIDL_int__array* sizes);

/**
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_IJBuildMatrix_Print(
  bHYPRE_IJBuildMatrix self,
  const char* filename);

/**
 * Read the matrix from file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_IJBuildMatrix_Read(
  bHYPRE_IJBuildMatrix self,
  const char* filename,
  void* comm);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_IJBuildMatrix
bHYPRE_IJBuildMatrix__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_IJBuildMatrix__cast2(
  void* obj,
  const char* type);

struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_createCol(int32_t        dimen,
                                      const int32_t lower[],
                                      const int32_t upper[]);

struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_createRow(int32_t        dimen,
                                      const int32_t lower[],
                                      const int32_t upper[]);

struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_create1d(int32_t len);

struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_borrow(bHYPRE_IJBuildMatrix*firstElement,
                                   int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_smartCopy(struct bHYPRE_IJBuildMatrix__array 
  *array);

void
bHYPRE_IJBuildMatrix__array_addRef(struct bHYPRE_IJBuildMatrix__array* array);

void
bHYPRE_IJBuildMatrix__array_deleteRef(struct bHYPRE_IJBuildMatrix__array* 
  array);

bHYPRE_IJBuildMatrix
bHYPRE_IJBuildMatrix__array_get1(const struct bHYPRE_IJBuildMatrix__array* 
  array,
                                 const int32_t i1);

bHYPRE_IJBuildMatrix
bHYPRE_IJBuildMatrix__array_get2(const struct bHYPRE_IJBuildMatrix__array* 
  array,
                                 const int32_t i1,
                                 const int32_t i2);

bHYPRE_IJBuildMatrix
bHYPRE_IJBuildMatrix__array_get3(const struct bHYPRE_IJBuildMatrix__array* 
  array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 const int32_t i3);

bHYPRE_IJBuildMatrix
bHYPRE_IJBuildMatrix__array_get4(const struct bHYPRE_IJBuildMatrix__array* 
  array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 const int32_t i3,
                                 const int32_t i4);

bHYPRE_IJBuildMatrix
bHYPRE_IJBuildMatrix__array_get(const struct bHYPRE_IJBuildMatrix__array* array,
                                const int32_t indices[]);

void
bHYPRE_IJBuildMatrix__array_set1(struct bHYPRE_IJBuildMatrix__array* array,
                                 const int32_t i1,
                                 bHYPRE_IJBuildMatrix const value);

void
bHYPRE_IJBuildMatrix__array_set2(struct bHYPRE_IJBuildMatrix__array* array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 bHYPRE_IJBuildMatrix const value);

void
bHYPRE_IJBuildMatrix__array_set3(struct bHYPRE_IJBuildMatrix__array* array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 const int32_t i3,
                                 bHYPRE_IJBuildMatrix const value);

void
bHYPRE_IJBuildMatrix__array_set4(struct bHYPRE_IJBuildMatrix__array* array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 const int32_t i3,
                                 const int32_t i4,
                                 bHYPRE_IJBuildMatrix const value);

void
bHYPRE_IJBuildMatrix__array_set(struct bHYPRE_IJBuildMatrix__array* array,
                                const int32_t indices[],
                                bHYPRE_IJBuildMatrix const value);

int32_t
bHYPRE_IJBuildMatrix__array_dimen(const struct bHYPRE_IJBuildMatrix__array* 
  array);

int32_t
bHYPRE_IJBuildMatrix__array_lower(const struct bHYPRE_IJBuildMatrix__array* 
  array,
                                  const int32_t ind);

int32_t
bHYPRE_IJBuildMatrix__array_upper(const struct bHYPRE_IJBuildMatrix__array* 
  array,
                                  const int32_t ind);

int32_t
bHYPRE_IJBuildMatrix__array_stride(const struct bHYPRE_IJBuildMatrix__array* 
  array,
                                   const int32_t ind);

int
bHYPRE_IJBuildMatrix__array_isColumnOrder(const struct 
  bHYPRE_IJBuildMatrix__array* array);

int
bHYPRE_IJBuildMatrix__array_isRowOrder(const struct 
  bHYPRE_IJBuildMatrix__array* array);

void
bHYPRE_IJBuildMatrix__array_slice(const struct bHYPRE_IJBuildMatrix__array* src,
                                        int32_t        dimen,
                                        const int32_t  numElem[],
                                        const int32_t  *srcStart,
                                        const int32_t  *srcStride,
                                        const int32_t  *newStart);

void
bHYPRE_IJBuildMatrix__array_copy(const struct bHYPRE_IJBuildMatrix__array* src,
                                       struct bHYPRE_IJBuildMatrix__array* 
  dest);

struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_ensure(struct bHYPRE_IJBuildMatrix__array* src,
                                   int32_t dimen,
                                   int     ordering);

#ifdef __cplusplus
}
#endif
#endif
