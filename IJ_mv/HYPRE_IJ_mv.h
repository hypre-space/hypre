/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#ifndef HYPRE_IJ_MV_HEADER
#define HYPRE_IJ_MV_HEADER

#include "HYPRE_config.h"
#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name IJ System Interface
 *
 * This interface represents a linear-algebraic conceptual view of a
 * linear system.  The 'I' and 'J' in the name are meant to be
 * mnemonic for the traditional matrix notation A(I,J).
 *
 * @memo A linear-algebraic conceptual interface
 * @author Andrew J. Cleary
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name IJ Matrices
 **/
/*@{*/

struct hypre_IJMatrix_struct;
/**
 * The matrix object.
 **/
typedef struct hypre_IJMatrix_struct *HYPRE_IJMatrix;

/**
 * Create a {\tt global\_m} by {\tt global\_n} matrix object.
 *
 * Collective.
 *
 * @param global_m [IN] global number of rows.
 *
 * @param global_n [IN] global number of columns.
 **/
int HYPRE_IJMatrixCreate(MPI_Comm        comm,
                         HYPRE_IJMatrix *matrix,
                         int             global_m,
                         int             global_n);

/**
 * Destroy a matrix object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
int HYPRE_IJMatrixDestroy(HYPRE_IJMatrix matrix);

/**
 * Prepare a matrix object for setting coefficient values.
 **/
int HYPRE_IJMatrixInitialize(HYPRE_IJMatrix matrix);

/**
 * Sets values in a particular row of the matrix.  Erases any previous
 * values at the specified locations and replaces them with new ones,
 * or, if there was no value there before, inserts a new one.
 *
 * Not collective.
 *
 * @param n      [IN] the number of values to be set in the row.
 *
 * @param row    [IN] row index for the values to be set.
 *
 * @param cols   [IN] column indices for the values to be set.
 *
 * @param values [IN] values to be set.
 **/
int HYPRE_IJMatrixSetValues(HYPRE_IJMatrix  matrix,
                            int             n,
                            int             row,
                            const int      *cols,
                            const double   *values);

/**
 * Adds to values in a particular row of the matrix.  Adds to any
 * previous values at the specified locations, or, if there was no
 * value there before, inserts a new one.
 *
 * Not collective.
 *
 * @param n      [IN] the number of values to be added in the row.
 *
 * @param row    [IN] row index for the values to be added.
 *
 * @param cols   [IN] column indices for the values to be added.
 *
 * @param values [IN] values to be added.
 **/
int HYPRE_IJMatrixAddToValues(HYPRE_IJMatrix  matrix,
                              int             n,
                              int             row,
                              const int      *cols,
                              const double   *values);

/**
 * Sets values in one or more rows of the matrix.  Erases any previous
 * values at the specified locations and replaces them with new ones,
 * or, if there was no value there before, inserts a new one.
 * Equivalent to \Ref{HYPRE_IJMatrixSetValues} except can span more
 * than one row at a time.
 *
 * Not collective.
 *
 * @param m      [IN] the number of row values to be set.
 *
 * @param n      [IN] the number of column values to be set.
 *
 * @param rows   [IN] row indices for the values to be set.
 *
 * @param cols   [IN] column indices for the values to be set.
 *
 * @param values [IN] values to be set.
 **/
int HYPRE_IJMatrixSetBlockValues(HYPRE_IJMatrix  matrix,
                                 int             m,
                                 int             n,
                                 const int      *rows,
                                 const int      *cols,
                                 const double   *values);

/**
 * Adds to values in one or more rows of the matrix.  Adds to any
 * previous values at the specified locations, or, if there was no
 * value there before, inserts a new one.  Equivalent to
 * \Ref{HYPRE_IJMatrixAddToValues} except can span more than one row
 * at a time.
 *
 * Not collective.
 *
 * @param m      [IN] the number of row values to be added.
 *
 * @param n      [IN] the number of column values to be added.
 *
 * @param rows   [IN] row indices for the values to be added.
 *
 * @param cols   [IN] column indices for the values to be added.
 *
 * @param values [IN] values to be added.
 **/
int HYPRE_IJMatrixAddToBlockValues(HYPRE_IJMatrix matrix,
                                   int             m,
                                   int             n,
                                   const int      *rows,
                                   const int      *cols,
                                   const double   *values);

/**
 * Finalize the construction of the matrix before using.
 **/
int HYPRE_IJMatrixAssemble(HYPRE_IJMatrix matrix);


/**
 * Set the storage type of the matrix object to be constructed.
 * Currently, {\tt type} can only be {\tt HYPRE\_PARCSR}.
 *
 * Not collective, but must be the same on all processes.
 *
 * @see HYPRE_IJMatrixGetLocalStorage
 **/
int HYPRE_IJMatrixSetLocalStorageType(HYPRE_IJMatrix matrix,
                                      int            type);

/**
 * Get a reference to the constructed matrix object.
 *
 * @see HYPRE_IJMatrixSetLocalStorageType
 **/
void *HYPRE_IJMatrixGetLocalStorage(HYPRE_IJMatrix matrix);

/**
 * (Optional) Set the number of rows and columns owned by this
 * process.
 *
 * Not collective.
 *
 * REQUIREMENTS: \Ref{HYPRE_IJMatrixSetLocalStorageType} must be
 * called before this routine.
 *
 * @param local_m [IN] local number of rows.
 *
 * @param local_n [IN] local number of columns.
 **/
int HYPRE_IJMatrixSetLocalSize(HYPRE_IJMatrix matrix,
                               int            local_m,
                               int            local_n);

/**
 * (Optional) Set the max number of nonzeros to expect in each row.
 * This call can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 *
 * Not collective.
 *
 * @param sizes [IN] a vector of length {\tt local\_m} with the
 * estimated sizes for each row.
 **/
int HYPRE_IJMatrixSetRowSizes(HYPRE_IJMatrix  matrix,
                              const int      *sizes);

/**
 * (Optional) Set the max number of nonzeros to expect in each row of
 * the diagonal block (the submatrix whose column numbers correspond
 * to rows owned by this process).  This routine can significantly
 * improve the efficiency of matrix construction, and should always be
 * utilized if possible.
 *
 * Not collective.
 *
 * @param sizes [IN] a vector of length {\tt local\_m} with the
 * estimated sizes for the diagonal parts of each row.
 **/
int HYPRE_IJMatrixSetDiagRowSizes(HYPRE_IJMatrix  matrix,
                                  const int      *sizes);

/**
 * (Optional) Set the max number of nonzeros to expect in each row of
 * the off-diagonal block (the submatrix whose column numbers do not
 * correspond to rows owned by this process).  This routine can
 * significantly improve the efficiency of matrix construction, and
 * should always be utilized if possible.
 *
 * Not collective.
 *
 * @param sizes [IN] a vector of length {\tt local\_m} with the
 * estimated sizes for the diagonal parts of each row.
 **/
int HYPRE_IJMatrixSetOffDiagRowSizes(HYPRE_IJMatrix  matrix,
                                     const int      *sizes);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name IJ Vectors
 **/
/*@{*/

struct hypre_IJVector_struct;
/**
 * The vector object.
 **/
typedef struct hypre_IJVector_struct *HYPRE_IJVector;

/**
 * Create a {\tt global\_n} dimensioned vector object.
 *
 * Collective.
 *
 * @param global_n [IN] global number of unknowns.
 **/
int HYPRE_IJVectorCreate(MPI_Comm        comm,
                         HYPRE_IJVector *vector,
                         int             global_n);

/**
 * Destroy a vector object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
int HYPRE_IJVectorDestroy(HYPRE_IJVector vector);

/**
 * Prepare a vector object for setting coefficient values.
 **/
int HYPRE_IJVectorInitialize(HYPRE_IJVector vector);

/**
 * Sets values in vector.  Erases any previous values at the specified
 * locations and replaces them with new ones.
 *
 * Not collective.
 *
 * @param num_values    [IN] the number of values to be set.
 *
 * @param glob_indices  [IN] global vector indices for the values to be set.
 *
 * @param value_indices [IN] corresonding indices for {\tt values} array.
 *
 * @param values        [IN] values to be set.
 **/
int HYPRE_IJVectorSetLocalComponents(HYPRE_IJVector  vector,
                                     int             num_values,
                                     const int      *glob_indices,
                                     const int      *value_indices,
                                     const double   *values);

/*
 * RE-VISIT
 **/
int HYPRE_IJVectorSetLocalComponentsInBlock(HYPRE_IJVector  vector,
                                            int             glob_index_start,
                                            int             glob_index_stop,
                                            const int      *value_indices,
                                            const double   *values);

/**
 * Adds to values in vector.
 *
 * Not collective.
 *
 * @param num_values    [IN] the number of values to be added.
 *
 * @param glob_indices  [IN] global vector indices for the values to be added.
 *
 * @param value_indices [IN] corresonding indices for {\tt values} array.
 *
 * @param values        [IN] values to be added.
 **/
int HYPRE_IJVectorAddToLocalComponents(HYPRE_IJVector  vector,
                                       int             num_values,
                                       const int      *glob_indices,
                                       const int      *value_indices,
                                       const double   *values);

/*
 * RE-VISIT
 **/
int HYPRE_IJVectorAddToLocalComponentsInBlock(HYPRE_IJVector  vector,
                                              int             glob_index_start,
                                              int             glob_index_stop,
                                              const int      *value_indices,
                                              const double   *values);

/**
 * Finalize the construction of the vector before using.
 **/
int HYPRE_IJVectorAssemble(HYPRE_IJVector vector);

/**
 * Set the storage type of the vector object to be constructed.
 * Currently, {\tt type} can only be {\tt HYPRE\_PARCSR}.
 *
 * Not collective, but must be the same on all processes.
 *
 * @see HYPRE_IJVectorGetLocalStorage
 **/
int HYPRE_IJVectorSetLocalStorageType(HYPRE_IJVector vector,
                                      int            type);

/**
 * Get a reference to the constructed vector object.
 *
 * @see HYPRE_IJVectorSetLocalStorageType
 **/
void *HYPRE_IJVectorGetLocalStorage(HYPRE_IJVector vector);

/**
 * Gets values in vector.
 *
 * Not collective.
 *
 * @param num_values    [IN] the number of values to be set.
 *
 * @param glob_indices  [IN] global vector indices for the values to be set.
 *
 * @param value_indices [IN] corresonding indices for {\tt values} array.
 *
 * @param values        [IN] values to be set.
 **/
int HYPRE_IJVectorGetLocalComponents(HYPRE_IJVector  vector,
                                     int             num_values,
                                     const int      *glob_indices,
                                     const int      *value_indices,
                                     double         *values);

/*
 * RE-VISIT
 **/
int HYPRE_IJVectorGetLocalComponentsInBlock(HYPRE_IJVector  vector,
                                            int             glob_index_start,
                                            int             glob_index_stop,
                                            const int      *value_indices,
                                            double         *values);

/**
 * (Optional) Indicate the number of elements of the vector assigned
 * to each processor.
 *
 * REQUIREMENTS: \Ref{HYPRE_IJVectorSetLocalStorageType} must be
 * called before this routine.
 **/
int HYPRE_IJVectorSetPartitioning(HYPRE_IJVector  vector,
                                  const int      *partitioning);

/**
 * (Optional) Indicate which rows are stored on this processor.
 *
 * @param vec_start_this_proc [IN] first vector index on this process.
 *
 * @param vec_start_next_proc [IN] first vector index on next process.
 **/
int HYPRE_IJVectorSetLocalPartitioning(HYPRE_IJVector vector,
                                       int            vec_start_this_proc,
                                       int            vec_start_next_proc);

/*@}*/
/*@}*/

/*--------------------------------------------------------------------------
 * Miscellaneous: These probably do not belong in the interface.
 *--------------------------------------------------------------------------*/

int HYPRE_IJVectorGetLocalStorageType(HYPRE_IJVector  vector,
                                      int            *type);

/*
 * There are three possible semantic levels in the integer parameter
 * {\tt row} used in Set and Add functions, and each implementation
 * supports one of them. This function returns a value indicating the
 * semantics supported by the instantiated matrix.
 *
 * level = -1: processors may include values for "row" for any row
 * number in the global matrix.
 *
 * level = 0: processors may only include values for row representing
 * locally stored rows.
 *
 * level = > 0: in addition to the above, processors may also include
 * values for "row" representing rows within the set of locally stored
 * rows plus the next "level" levels of nearest neighbors, also known
 * as "level sets" or "nearest neighbors".
 *
 * Since level 0 is the most restrictive, it is also the easiest to
 * implement, and the safest to use, as all IJ matrices MUST support
 * level 0.  In contrast, level 1 is the most general and most
 * difficult to implement, least safe to use, and potentially least
 * efficient.  Levels greater than 0 represent a compromise that is
 * appropriate for many engineering applications, like finite element
 * applications, where contributions to a particular matrix row may be
 * made from more than one processor, typically in some small
 * neighborhood around that row.
 *
 * Not collective.
 *
 * {\bf Note:} It is probably best to ignore this routine for the time
 * being as it is really planned as a possible future contingency and
 * is confusing as it is now. -AJC, 6/99
 *
 * @param level [OUT] level of off-processor value-setting that this
 * implementation supports.
 **/
int HYPRE_IJMatrixQueryInsertionSemantics(HYPRE_IJMatrix  matrix,
                                          int            *level);

/*
 * Inserts a block of coefficients into the matrix, overwriting any
 * coefficients in the event of a collision.
 *
 * Not collective.
 *
 * @param m [IN] the size of the block of values to be added.
 *
 * @param n [IN] the size of the block of values to be added.
 *
 * @param rows [IN] an integer vector of length m giving the indices
 * in the global matrix corresponding to the rows in "values".
 *
 * @param cols [IN] an integer vector of length n giving the indices
 * in the global matrix corresponding to the columns in "values".
 *
 * @param values [IN] The values to be inserted into the matrix,
 * stored in a dense, row-major block of size m X n.
 **/
int HYPRE_IJMatrixInsertBlock(HYPRE_IJMatrix  matrix,
                              int             m,
                              int             n,
                              const int      *rows,
                              const int      *cols,
                              const double   *values);

/*
 * Modifies the values stored in matrix by adding in a block.  If
 * there is no value already in a particular matrix position, the
 * structure is augmented with a new entry. In the event of a
 * collision, the corresponding values are summed.
 *
 * Not collective.
 *
 * @param m, n [IN] the size of the block of values to be added.
 * 
 * @param rows [IN] an integer vector of length m giving the indices
 * in the global matrix corresponding to the rows in "values".
 *
 * @param cols [IN] an integer vector of length n giving the indices
 * in the global matrix corresponding to the columns in "values".
 *
 * @param values [IN] The values to be inserted into the matrix,
 * stored in a dense, row-major block of size m X n.
 **/
int HYPRE_IJMatrixAddToBlock(HYPRE_IJMatrix  matrix,
                             int             m,
                             int             n,
                             const int      *rows,
                             const int      *cols,
                             const double   *values);

/*
 * Inserts a row into the matrix. This is generally a high-speed but
 * inflexible method to build the matrix. This call replaces any
 * previously existing row structure with the structure represented by
 * indices and coeffs.
 *
 * Not collective.
 *
 * @param n [IN] the number of values in the row to be inserted.
 *
 * @param row [IN] index of row to be inserted.
 *
 * @param cols [IN] an integer vector of length n giving the indices
 * in the global matrix corresponding to the columns in "values".
 *
 * @param values [IN] The values to be inserted into the matrix.
 **/
int HYPRE_IJMatrixInsertRow(HYPRE_IJMatrix  matrix,
                            int             n,
                            int             row,
                            const int      *cols,
                            const double   *values);

/*
 * Adds a row to the row of a matrix before assembly. 
 *
 * Not collective.
 *
 * @param n [IN] the number of values in the row to be added.
 *
 * @param row [IN] index of row to be added.
 *
 * @param cols [IN] an integer vector of length n giving the indices
 * in the global matrix corresponding to the columns in "values".
 *
 * @param values [IN] The values to be added to the matrix.
 **/
int HYPRE_IJMatrixAddToRow(HYPRE_IJMatrix  matrix,
                           int             n,
                           int             row,
                           const int      *cols,
                           const double   *values);

/*
 * Adds a row to the row of a matrix after assembly.  Note: Adds only
 * to already existing elements.
 *
 * DOES ASSEMBLE NEED TO BE RECALLED?
 *
 * IMHO, this routine should not be a standard part of the IJ
 * interface... AJC.
 *
 * Not collective.
 *
 * @param n [IN] the number of values in the row to be added.
 *
 * @param row [IN] index of row to be added.
 *
 * @param cols [IN] an integer vector of length n giving the indices
 * in the global matrix corresponding to the columns in "values".
 *
 * @param values [IN] The values to be added to the matrix.
 **/
int HYPRE_IJMatrixAddToRowAfter(HYPRE_IJMatrix  matrix,
                                int             n,
                                int             row,
                                const int      *cols,
                                const double   *values);

/* Internal routine only. */
int HYPRE_IJMatrixDistribute(HYPRE_IJMatrix  matrix,
                             const int      *row_starts,
                             const int      *col_starts);

int HYPRE_IJMatrixGetRowPartitioning(HYPRE_IJMatrix matrix,
                                     const int **row_partitioning);
int HYPRE_IJMatrixGetColPartitioning(HYPRE_IJMatrix   matrix,
                                     const int      **col_partitioning);

/* Internal routine only. */
int HYPRE_IJVectorDistribute(HYPRE_IJVector  vector,
                             const int      *vec_starts);

/* Internal routine only. */
int hypre_RefIJMatrix(HYPRE_IJMatrix  matrix,
                      HYPRE_IJMatrix *reference);

/* Internal routine only. */
int hypre_RefIJVector(HYPRE_IJVector  vector,
                      HYPRE_IJVector *reference);

/*
 * Zeros all of the local vector components, overwriting all indexed
 * coefficients.
 *
 * Not collective.
**/
int HYPRE_IJVectorZeroLocalComponents(HYPRE_IJVector vector);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif
