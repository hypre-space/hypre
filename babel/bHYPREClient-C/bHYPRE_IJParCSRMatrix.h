/*
 * File:          bHYPRE_IJParCSRMatrix.h
 * Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:34 PST
 * Generated:     20030401 14:47:44 PST
 * Description:   Client-side glue code for bHYPRE.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 789
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_IJParCSRMatrix_h
#define included_bHYPRE_IJParCSRMatrix_h

/**
 * Symbol "bHYPRE.IJParCSRMatrix" (version 1.0.0)
 * 
 * The IJParCSR matrix class.
 * 
 * Objects of this type can be cast to IJBuildMatrix, Operator, or
 * CoefficientAccess objects using the {\tt \_\_cast} methods.
 * 
 */
struct bHYPRE_IJParCSRMatrix__object;
struct bHYPRE_IJParCSRMatrix__array;
typedef struct bHYPRE_IJParCSRMatrix__object* bHYPRE_IJParCSRMatrix;

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
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix__create(void);

void
bHYPRE_IJParCSRMatrix_addRef(
  bHYPRE_IJParCSRMatrix self);

void
bHYPRE_IJParCSRMatrix_deleteRef(
  bHYPRE_IJParCSRMatrix self);

SIDL_bool
bHYPRE_IJParCSRMatrix_isSame(
  bHYPRE_IJParCSRMatrix self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_IJParCSRMatrix_queryInt(
  bHYPRE_IJParCSRMatrix self,
  const char* name);

SIDL_bool
bHYPRE_IJParCSRMatrix_isType(
  bHYPRE_IJParCSRMatrix self,
  const char* name);

SIDL_ClassInfo
bHYPRE_IJParCSRMatrix_getClassInfo(
  bHYPRE_IJParCSRMatrix self);

/**
 * (Optional) Set the max number of nonzeros to expect in each
 * row of the diagonal and off-diagonal blocks.  The diagonal
 * block is the submatrix whose column numbers correspond to
 * rows owned by this process, and the off-diagonal block is
 * everything else.  The arrays {\tt diag\_sizes} and {\tt
 * offdiag\_sizes} contain estimated sizes for each row of the
 * diagonal and off-diagonal blocks, respectively.  This routine
 * can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetDiagOffdSizes(
  bHYPRE_IJParCSRMatrix self,
  struct SIDL_int__array* diag_sizes,
  struct SIDL_int__array* offdiag_sizes);

/**
 * Set the MPI Communicator.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetCommunicator(
  bHYPRE_IJParCSRMatrix self,
  void* mpi_comm);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_Initialize(
  bHYPRE_IJParCSRMatrix self);

/**
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_Assemble(
  bHYPRE_IJParCSRMatrix self);

/**
 * The problem definition interface is a {\it builder} that
 * creates an object that contains the problem definition
 * information, e.g. a matrix. To perform subsequent operations
 * with that object, it must be returned from the problem
 * definition object. {\tt GetObject} performs this function.
 * At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a SIDL.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_GetObject(
  bHYPRE_IJParCSRMatrix self,
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
bHYPRE_IJParCSRMatrix_SetLocalRange(
  bHYPRE_IJParCSRMatrix self,
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
bHYPRE_IJParCSRMatrix_SetValues(
  bHYPRE_IJParCSRMatrix self,
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
bHYPRE_IJParCSRMatrix_AddToValues(
  bHYPRE_IJParCSRMatrix self,
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
bHYPRE_IJParCSRMatrix_GetLocalRange(
  bHYPRE_IJParCSRMatrix self,
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
bHYPRE_IJParCSRMatrix_GetRowCounts(
  bHYPRE_IJParCSRMatrix self,
  int32_t nrows,
  struct SIDL_int__array* rows,
  struct SIDL_int__array** ncols);

/**
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_GetValues(
  bHYPRE_IJParCSRMatrix self,
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
bHYPRE_IJParCSRMatrix_SetRowSizes(
  bHYPRE_IJParCSRMatrix self,
  struct SIDL_int__array* sizes);

/**
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_Print(
  bHYPRE_IJParCSRMatrix self,
  const char* filename);

/**
 * Read the matrix from file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_Read(
  bHYPRE_IJParCSRMatrix self,
  const char* filename,
  void* comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetIntParameter(
  bHYPRE_IJParCSRMatrix self,
  const char* name,
  int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetDoubleParameter(
  bHYPRE_IJParCSRMatrix self,
  const char* name,
  double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetStringParameter(
  bHYPRE_IJParCSRMatrix self,
  const char* name,
  const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetIntArray1Parameter(
  bHYPRE_IJParCSRMatrix self,
  const char* name,
  struct SIDL_int__array* value);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetIntArray2Parameter(
  bHYPRE_IJParCSRMatrix self,
  const char* name,
  struct SIDL_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter(
  bHYPRE_IJParCSRMatrix self,
  const char* name,
  struct SIDL_double__array* value);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter(
  bHYPRE_IJParCSRMatrix self,
  const char* name,
  struct SIDL_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_GetIntValue(
  bHYPRE_IJParCSRMatrix self,
  const char* name,
  int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_GetDoubleValue(
  bHYPRE_IJParCSRMatrix self,
  const char* name,
  double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_Setup(
  bHYPRE_IJParCSRMatrix self,
  bHYPRE_Vector b,
  bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_Apply(
  bHYPRE_IJParCSRMatrix self,
  bHYPRE_Vector b,
  bHYPRE_Vector* x);

/**
 * The GetRow method will allocate space for its two output
 * arrays on the first call.  The space will be reused on
 * subsequent calls.  Thus the user must not delete them, yet
 * must not depend on the data from GetRow to persist beyond the
 * next GetRow call.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_GetRow(
  bHYPRE_IJParCSRMatrix self,
  int32_t row,
  int32_t* size,
  struct SIDL_int__array** col_ind,
  struct SIDL_double__array** values);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_IJParCSRMatrix__cast2(
  void* obj,
  const char* type);

struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_createCol(int32_t        dimen,
                                       const int32_t lower[],
                                       const int32_t upper[]);

struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_createRow(int32_t        dimen,
                                       const int32_t lower[],
                                       const int32_t upper[]);

struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_create1d(int32_t len);

struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_borrow(bHYPRE_IJParCSRMatrix*firstElement,
                                    int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_smartCopy(struct bHYPRE_IJParCSRMatrix__array 
  *array);

void
bHYPRE_IJParCSRMatrix__array_addRef(struct bHYPRE_IJParCSRMatrix__array* array);

void
bHYPRE_IJParCSRMatrix__array_deleteRef(struct bHYPRE_IJParCSRMatrix__array* 
  array);

bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix__array_get1(const struct bHYPRE_IJParCSRMatrix__array* 
  array,
                                  const int32_t i1);

bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix__array_get2(const struct bHYPRE_IJParCSRMatrix__array* 
  array,
                                  const int32_t i1,
                                  const int32_t i2);

bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix__array_get3(const struct bHYPRE_IJParCSRMatrix__array* 
  array,
                                  const int32_t i1,
                                  const int32_t i2,
                                  const int32_t i3);

bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix__array_get4(const struct bHYPRE_IJParCSRMatrix__array* 
  array,
                                  const int32_t i1,
                                  const int32_t i2,
                                  const int32_t i3,
                                  const int32_t i4);

bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix__array_get(const struct bHYPRE_IJParCSRMatrix__array* 
  array,
                                 const int32_t indices[]);

void
bHYPRE_IJParCSRMatrix__array_set1(struct bHYPRE_IJParCSRMatrix__array* array,
                                  const int32_t i1,
                                  bHYPRE_IJParCSRMatrix const value);

void
bHYPRE_IJParCSRMatrix__array_set2(struct bHYPRE_IJParCSRMatrix__array* array,
                                  const int32_t i1,
                                  const int32_t i2,
                                  bHYPRE_IJParCSRMatrix const value);

void
bHYPRE_IJParCSRMatrix__array_set3(struct bHYPRE_IJParCSRMatrix__array* array,
                                  const int32_t i1,
                                  const int32_t i2,
                                  const int32_t i3,
                                  bHYPRE_IJParCSRMatrix const value);

void
bHYPRE_IJParCSRMatrix__array_set4(struct bHYPRE_IJParCSRMatrix__array* array,
                                  const int32_t i1,
                                  const int32_t i2,
                                  const int32_t i3,
                                  const int32_t i4,
                                  bHYPRE_IJParCSRMatrix const value);

void
bHYPRE_IJParCSRMatrix__array_set(struct bHYPRE_IJParCSRMatrix__array* array,
                                 const int32_t indices[],
                                 bHYPRE_IJParCSRMatrix const value);

int32_t
bHYPRE_IJParCSRMatrix__array_dimen(const struct bHYPRE_IJParCSRMatrix__array* 
  array);

int32_t
bHYPRE_IJParCSRMatrix__array_lower(const struct bHYPRE_IJParCSRMatrix__array* 
  array,
                                   const int32_t ind);

int32_t
bHYPRE_IJParCSRMatrix__array_upper(const struct bHYPRE_IJParCSRMatrix__array* 
  array,
                                   const int32_t ind);

int32_t
bHYPRE_IJParCSRMatrix__array_stride(const struct bHYPRE_IJParCSRMatrix__array* 
  array,
                                    const int32_t ind);

int
bHYPRE_IJParCSRMatrix__array_isColumnOrder(const struct 
  bHYPRE_IJParCSRMatrix__array* array);

int
bHYPRE_IJParCSRMatrix__array_isRowOrder(const struct 
  bHYPRE_IJParCSRMatrix__array* array);

void
bHYPRE_IJParCSRMatrix__array_slice(const struct bHYPRE_IJParCSRMatrix__array* 
  src,
                                         int32_t        dimen,
                                         const int32_t  numElem[],
                                         const int32_t  *srcStart,
                                         const int32_t  *srcStride,
                                         const int32_t  *newStart);

void
bHYPRE_IJParCSRMatrix__array_copy(const struct bHYPRE_IJParCSRMatrix__array* 
  src,
                                        struct bHYPRE_IJParCSRMatrix__array* 
  dest);

struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_ensure(struct bHYPRE_IJParCSRMatrix__array* src,
                                    int32_t dimen,
                                    int     ordering);

#ifdef __cplusplus
}
#endif
#endif
