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
 * @name IJ Matrix-Vector Building Interfaces
 *
 * IJ represents the ``linear algebraic'' conceptual view of a matrix.
 * The 'I' and 'J' in the name are meant to be mnemonic for the
 * traditional matrix notation A(I,J).
 *
 * @memo A linear-algebraic abstract matrix building interface
 * @version 0.3
 * @author Andrew J. Cleary
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name IJ Matrix Builder
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt HYPRE\_IJMatrix} object: This object follows the classic
 * builder pattern. An IJMatrix is *not* a matrix data structure
 * itself, it *builds* one. This allows users to use the same
 * interface to build different underlying data structures.
 **/
struct hypre_IJMatrix_struct;
typedef struct hypre_IJMatrix_struct *HYPRE_IJMatrix;

/**
 * Create a matrix builder that implements the IJ interface.
 * Collective.
 *
 * {\bf Note:} Must be the first function called using "matrix" as an
 * actual argument.
 *
 * @return Error code.
 *
 * @param IJmatrix the matrix to be initialized.
 *
 * @param comm a single MPI communicator that contains exactly the
 * MPI processes that are to participate in any collective operations.
 *
 * @param global_m [IN] global number of rows.
 *
 * @param global_n [IN] global number of columns.
 *
 * @param local_m [IN] local number of rows.
 *
 * @param local_n [IN] local number of columns.
 **/
int HYPRE_IJMatrixCreate(MPI_Comm        comm,
                         HYPRE_IJMatrix *in_matrix_ptr,
                         int             global_m,
                         int             global_n);

/**
 * Destroy a matrix builder object. An object should be explicitly
 * destroyed using this destructor when the user's code no longer
 * needs direct access to the grid description.  Once destroyed, the
 * object must not be referenced again.  Note that the object may not
 * be deallocated at the completion of this call, since there may be
 * internal package references to the object.  The object will then be
 * destroyed when all internal reference counts go to zero.
 *
 * @return Error code.
 *
 * @param IJmatrix the matrix to be destroyed.
 **/
int HYPRE_IJMatrixDestroy(HYPRE_IJMatrix IJmatrix);

/**
 * Initialize an {\tt IJ\_Matrix} object.  Must be called before any
 * of the {\tt Insert} or {\tt Add} routines are called.
 *
 * @return Error code.
 *
 * @param IJmatrix the matrix to be initialized.
 **/
int HYPRE_IJMatrixInitialize(HYPRE_IJMatrix IJmatrix);

/**
 * Assemble an {\tt IJ\_Matrix} object.  Must be called after all of
 * the Insert or add routines are called.  After assemble completes,
 * the builder has constructed a matrix that can be used to multiply
 * vectors, define linear systems, build solvers and preconditioners,
 * etc.
 *
 * @return Error code.
 *
 * @param IJmatrix the matrix to be assembled.
 **/
int HYPRE_IJMatrixAssemble(HYPRE_IJMatrix IJmatrix);


/**
 * Tells the builder which storage type it should build from the
 * IJMatrix interface calls. Conceptually, IJMatrix is just an
 * interface (an abstract base class in C++-speak), with no
 * implementation. Specific concrete classes implement this
 * interface. One way for a user to choose which concrete class to use
 * is by *instantiating* the class of choice (in an OO implementation)
 * or by *linking* in the implementation of choice (a compile time
 * choice). An intermediate method is through the use of {\tt
 * HYPRE\_SetIJMatrixStorageType} that *may* be provided by specific
 * implementations. For example, it is possible to build either a
 * PETSc MPIAIJ matrix or a ParCSR matrix from the IJ matrix
 * interface, and this choice can be controlled through this function.
 *
 * Not collective, but must be same on all processors in group that
 * stores matrix.
 *
 * @return Error code.
 *
 * @param matrix [IN] the matrix to be operated on.
 *
 * @param type [IN] possible types should be documented with each
 * implementation. The first HYPRE implementation will list its
 * possible values in HYPRE.h (at least for now). Note that this
 * function is optional, as all implementations must have a
 * default. Eventually the plan is to let solvers choose a storage
 * type that they can work with automatically, but this is in the
 * future.
 **/
int HYPRE_IJMatrixSetLocalStorageType(HYPRE_IJMatrix IJmatrix,
                                      int            type);

/**
 * Tells "matrix" how many rows and columns are locally owned.  Not
 * collective.  REQUIREMENTS: {\tt HYPRE\_IJMatrixSetLocalStorageType}
 * needs to be called before this routine.
 *
 * @return Error code.
 *
 * @param matrix [IN] the matrix to be operated on.
 *
 * @param local_m [IN] local number of rows.
 *
 * @param local_n [IN] local number of columns.
 **/
int HYPRE_IJMatrixSetLocalSize(HYPRE_IJMatrix IJmatrix,
                               int            local_m,
                               int            local_n);

/**
 * Not collective.  Tells "matrix" how many nonzeros to expect in each
 * row.  Knowing this quantity apriori may have a significant impact
 * on the time needed for the Assemble phase, and this option should
 * always be utilized if the information is available.
 *
 * @return Error code.
 *
 * @param matrix [IN] the matrix to be operated on.
 *
 * @param sizes [IN] a vector of length {\tt local\_m} giving the
 * estimated sizes for the diagonal parts of all {\tt local\_m} rows,
 * in order from lowest globally numbered local row to highest.
 **/
int HYPRE_IJMatrixSetRowSizes(HYPRE_IJMatrix  IJmatrix,
                              const int      *sizes);

/**
 * Not collective.  Tells "matrix" how many nonzeros to expect in each
 * row corresponding to other rows also on this processor, also known
 * as the "diagonal block" corresponding to this processor. Knowing
 * this quantity apriori may have a significant impact on the time
 * needed for the Assemble phase, and this option should always be
 * utilized if the information is available. It is most useful in
 * conjunction with the next function.
 *
 * @return Error code.
 *
 * @param matrix [IN] the matrix to be operated on.
 *
 * @param sizes [IN] a vector of length {\tt local\_m} giving the
 * exact sizes for the diagonal parts of all {\tt local\_m} rows, in
 * order from lowest globally numbered local row to highest.
 **/
int HYPRE_IJMatrixSetDiagRowSizes(HYPRE_IJMatrix  IJmatrix,
                                  const int      *sizes);

/**
 * Tells "matrix" how many nonzeros to expect in each corresponding to
 * rows NOT on this processor, i.e. the off-diagonal block
 * corresponding to this processor. As above, this option should
 * always be utilized if the information is available, as the setting
 * of coefficients and assemble operations can be very expensive
 * without them.
 *
 * Not collective.
 *
 * @return Error code.
 *
 * @param matrix [IN] the matrix to be operated on.
 *
 * @param sizes [IN] a vector of length $\geq$ {\tt local\_m} giving
 * the exact sizes for the off-diagonal parts of all {\tt local\_m}
 * rows, in order from lowest globally numbered local row to highest.
 **/
int HYPRE_IJMatrixSetOffDiagRowSizes(HYPRE_IJMatrix  IJmatrix,
                                     const int      *sizes);

/**
 * There are three possible semantic levels in the integer parameter
 * {\tt row} used in Set and Add functions, and each implementation
 * supports one of them. This function returns a value indicating the
 * semantics supported by the instantiated IJMatrix.
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
 * @return Error code.
 *
 * @param matrix [IN] the matrix to be initialized.
 *
 * @param level [OUT] level of off-processor value-setting that this
 * implementation supports.
 **/
int HYPRE_IJMatrixQueryInsertionSemantics(HYPRE_IJMatrix  IJmatrix,
                                          int            *level);

/**
 * Inserts a block of coefficients into an IJMatrix, overwriting any
 * coefficients in the event of a collision.  See {\tt
 * HYPRE\_IJMatrixQueryInsertionSemantics} for discussion on which
 * processor this routine can be called from.
 *
 * Not collective.
 *
 * @return Error code.
 *
 * @param IJmatrix [IN] the matrix to be operated on.
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
int HYPRE_IJMatrixInsertBlock(HYPRE_IJMatrix  IJmatrix,
                              int             m,
                              int             n,
                              const int      *rows,
                              const int      *cols,
                              const double   *values);

/**
 * Modifies the values stored in matrix by adding in a block.  If
 * there is no value already in a particular matrix position, the
 * structure is augmented with a new entry. In the event of a
 * collision, the corresponding values are summed.
 *
 * Not collective.
 *
 * @return Error code.
 *
 * @param matrix [IN] the matrix to be operated on.
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
int HYPRE_IJMatrixAddToBlock(HYPRE_IJMatrix  IJmatrix,
                             int             m,
                             int             n,
                             const int      *rows,
                             const int      *cols,
                             const double   *values);

/**
 * Inserts a row into the matrix. This is generally a high-speed but
 * inflexible method to build the matrix. This call replaces any
 * previously existing row structure with the structure represented by
 * indices and coeffs.
 *
 * Not collective.
 *
 * @return Error code.
 *
 * @param matrix [IN] the matrix to be operated on.
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
int HYPRE_IJMatrixInsertRow(HYPRE_IJMatrix  IJmatrix,
                            int             n,
                            int             row,
                            const int      *cols,
                            const double   *values);

/**
 * Adds a row to the row of a matrix before assembly. 
 *
 * Not collective.
 *
 * @return Error code.
 *
 * @param matrix [IN] the matrix to be operated on.
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
int HYPRE_IJMatrixAddToRow(HYPRE_IJMatrix  IJmatrix,
                           int             n,
                           int             row,
                           const int      *cols,
                           const double   *values);

/**
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
 * @return Error code.
 *
 * @param matrix [IN] the matrix to be operated on.
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
int HYPRE_IJMatrixAddToRowAfter(HYPRE_IJMatrix  IJmatrix,
                                int             n,
                                int             row,
                                const int      *cols,
                                const double   *values);

/**
 * Inserts values in a particular row of the matrix.  Erases any
 * previous values at the specified locations and replaces them with
 * new ones, or, if there was no value there before, inserts the
 * value.
 *
 * Not collective.
 *
 * @return Error code.
 *
 * @param matrix [IN] the matrix to be operated on.
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
int HYPRE_IJMatrixSetValues(HYPRE_IJMatrix  IJmatrix,
                            int             n,
                            int             row,
                            const int      *cols,
                            const double   *values);

/**
 * Adds to values in a particular row of the matrix.  Adds to any
 * previous values at the specified locations, or, if there was no
 * value there before, inserts the value.
 *
 * Not collective.
 *
 * @return Error code.
 *
 * @param matrix [IN] the matrix to be operated on.
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
int HYPRE_IJMatrixAddToValues(HYPRE_IJMatrix  IJmatrix,
                              int             n,
                              int             row,
                              const int      *cols,
                              const double   *values);

/**
 * Inserts values in one or more rows of the matrix.  Erases any
 * previous values at the specified locations and replaces them with
 * new ones, or, if there was no value there before, inserts the
 * value.  Equivalent to SetValues except can span more than one row
 * at a time.
 *
 * Not collective.
 *
 * @return Error code.
 *
 * @param matrix [IN] the matrix to be operated on.
 *
 * @param n [IN] the number of values in the row to be added.
 *
 * @param rows [IN] m indexes of rows to be added.
 *
 * @param cols [IN] an integer vector of length n giving the indices
 * in the global matrix corresponding to the columns in "values".
 *
 * @param values [IN] The values to be added to the matrix.
 **/
int HYPRE_IJMatrixSetBlockValues(HYPRE_IJMatrix  IJmatrix,
                                 int             m,
                                 int             n,
                                 const int      *rows,
                                 const int      *cols,
                                 const double   *values);

/**
 * Adds to values in a particular row of the matrix.  Adds to any
 * previous values at the specified locations, or, if there was no
 * value there before, inserts the value.  Equivalent to AddValues
 * except can span more than one row at a time.
 *
 * Not collective.
 *
 * @return Error code.
 *
 * @param matrix [IN] the matrix to be operated on.
 *
 * @param n [IN] the number of values in the row to be added.
 *
 * @param rows [IN] m indexes of rows to be added.
 *
 * @param cols [IN] an integer vector of length n giving the indices
 * in the global matrix corresponding to the columns in "values".
 *
 * @param values [IN] The values to be added to the matrix.
 **/
int HYPRE_IJMatrixAddToBlockValues(HYPRE_IJMatrix IJmatrix,
                                   int             m,
                                   int             n,
                                   const int      *rows,
                                   const int      *cols,
                                   const double   *values);

int HYPRE_IJMatrixDistribute(HYPRE_IJMatrix  IJmatrix,
                             const int      *row_starts,
                             const int      *col_starts);

void *HYPRE_IJMatrixGetLocalStorage(HYPRE_IJMatrix IJmatrix);

int HYPRE_IJMatrixGetRowPartitioning(HYPRE_IJMatrix IJmatrix,
                                     const int **row_partitioning);
int HYPRE_IJMatrixGetColPartitioning(HYPRE_IJMatrix   IJmatrix,
                                     const int      **col_partitioning);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name IJ Vector Builder
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt HYPRE\_IJVector} object: This object follows the classic
 * builder pattern. An IJVector is *not* a vector data structure
 * itself, it *builds* one. This allows users to use the same
 * interface to build different underlying data structures.
 **/
struct hypre_IJVector_struct;
typedef struct hypre_IJVector_struct *HYPRE_IJVector;

/**
 * Create a vector builder that implements the IJ interface.
 * Collective.
 *
 * {\bf Note:} Must be the first function called using "vector" as an
 * actual argument.
 *
 * @return Error code.
 *
 * @param IJvector the vector to be initialized.
 *
 * @param comm a single MPI communicator that contains exactly the
 * MPI processes that are to participate in any collective operations.
 *
 * @param global_m [IN] global number of rows.
 *
 * @param global_n [IN] global number of columns.
 *
 * @param local_m [IN] local number of rows.
 *
 * @param local_n [IN] local number of columns.
 **/
int HYPRE_IJVectorCreate(MPI_Comm        comm,
                         HYPRE_IJVector *IJvector,
                         int             global_n);

/**
 * Destroy a vector builder object. An object should be explicitly
 * destroyed using this destructor when the user's code no longer
 * needs direct access to the grid description.  Once destroyed, the
 * object must not be referenced again.  Note that the object may not
 * be deallocated at the completion of this call, since there may be
 * internal package references to the object.  The object will then be
 * destroyed when all internal reference counts go to zero.
 *
 * @return Error code.
 *
 * @param IJvector the vector to be destroyed.
 **/
int HYPRE_IJVectorDestroy(HYPRE_IJVector IJvector);

/**
 * Indicate the number of elements of the vector assigned to each processor.
 *
 * @param HYPRE_IJVector IJvector [IN] The vector to be operated on.
 *
 * @param int *partitioning [IN]
 * pointer to array of integers specifying vector decomposition across processors
 * @note HYPRE_IJVectorSetLocalStorageType needs to be called before this function.
 **/
int HYPRE_IJVectorSetPartitioning(HYPRE_IJVector  IJvector,
                                  const int      *partitioning);

/**
 * Sets information about which rows are stored on this processor
 *
 * @param HYPRE_IJVector IJvector [IN] The vector to be operated on.
 * @param int vec_start_this_proc [IN]
 * integer specifying local index to first data element on calling processor
 * @param int vec_start_next_proc [IN]
 * integer specifying local index to first data element on next processor in
 * decomposition
 **/
int HYPRE_IJVectorSetLocalPartitioning(HYPRE_IJVector IJvector,
                                       int            vec_start_this_proc,
                                       int            vec_start_next_proc);

/**
 * Initialize an {\tt IJ\_Vector} object.  Must be called before any
 * of the {\tt Insert} or {\tt Add} routines are called.
 *
 * @return Error code.
 *
 **/
int HYPRE_IJVectorInitialize(HYPRE_IJVector IJvector);


/**
 *
 * Tells "vector" which underlying "storage type" it should build from
 * the IJVector interface calls. Conceptually, "IJVector" is just an
 * "interface" (an abstract base class in C++-speak), with no
 * implementation. Specific concrete classes implement this
 * interface, and this function chooses which one.

Not collective, but must be same on all processors in group that stores vector.
@return integer error code
@param HYPRE_IJVector IJvector
vector for which the type is to be set
@param int type [IN]
Possible types should be documented with each implementation. The first HYPRE implementation
will list its possible values in HYPRE.h (at least for now). Note that this function is optional,
as all implementations must have a default. Eventually the plan is to let solvers choose a storage
type that they can work with automatically, but this is in the future.
 **/
int HYPRE_IJVectorSetLocalStorageType(HYPRE_IJVector IJvector,
                                      int            type);

/**
zeros all of the local vector components, overwriting
all indexed coefficients.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector
vector, some components of which are to be set
@param int *glob_vec_indices [IN]
pointer to global indices of local vector components to be set
@param double value [IN]
value to which the local vector components are to be set
**/
int HYPRE_IJVectorZeroLocalComponents(HYPRE_IJVector IJvector);

/** 
From indexed values of an array, sets indexed components of an IJVector, 
overwriting all indexed components.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
vector, some components of which are to be inserted to
@param int *glob_vec_indices [IN]
pointer to indices of local vector specifying which local vector components
to be overwritten
@param int *value_indices [IN]
pointer to indices of values array specifying values from which to insert 
@param double *values [IN]
pointer to array from which vector values are inserted

**/
int HYPRE_IJVectorSetLocalComponents(HYPRE_IJVector  IJvector,
                                     int             num_values,
                                     const int      *glob_indices,
                                     const int      *value_indices,
                                     const double   *values);

/**
 *
From indexed values of an array, sets a block of components of an IJVector,
overwriting all components of the block.

Not collective.

@return integer error code
@param HYPRE_IJVector IJvector
the vector to be operated on
@param int glob_vec_index_start [IN]
global index of first vector component in block to be inserted 
@param int glob_vec_index_stop [IN]
global index of last vector component in block to be set 
@param int *value_indices [IN]
pointer to indices of values array specifying values from which to insert 
@param double *values [IN]
pointer to array from which vector values are inserted
 **/
int HYPRE_IJVectorSetLocalComponentsInBlock(HYPRE_IJVector  IJvector,
                                            int             glob_index_start,
                                            int             glob_index_stop,
                                            const int      *value_indices,
                                            const double   *values);

/**
 *
Adds indexed values of an array to an indexed set of components of
an IJVector, overwriting all indexed components.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
vector, some components of which are to be summed to
@param int *glob_vec_indices [IN]
pointer to global indices of local vector components to be set 
@param int *value_indices [IN]
pointer to indices of value array members to be summed to local vector
components
@param double *values [IN]
pointer to array from which values are taken for summing
 **/
int HYPRE_IJVectorAddToLocalComponents(HYPRE_IJVector  IJvector,
                                       int             num_values,
                                       const int      *glob_indices,
                                       const int      *value_indices,
                                       const double   *values);

/**
 *
Adds indexed values of an array into a block of components of an IJVector,
overwriting all components in the block.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector
vector, the components of which are to be summed to
@param int glob_vec_index_start [IN]
global index of first vector component in block to be summed to
@param int glob_vec_index_stop [IN]
global index of last vector component in block to be summed to
@param int *value_indices [IN]
pointer to indices of value array members to be summed to local vector
components
@param double *values [IN]
pointer to array from which values are taken for summing
**/
int HYPRE_IJVectorAddToLocalComponentsInBlock(HYPRE_IJVector  IJvector,
                                              int             glob_index_start,
                                              int             glob_index_stop,
                                              const int      *value_indices,
                                              const double   *values);

/**
 * Assemble an {\tt IJ\_Vector} object.  Must be called after all of
 * the Insert or add routines are called.  After assemble completes,
 * the builder has constructed a vector that can be used as the
 * righthand side in a linear solve, etc.
 *
 * @return Error code.
 *
 * @param IJvector the vector to be assembled.
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_IJVectorAssemble(HYPRE_IJVector IJvector);

/**
Puts indexed components of IJVector into indexed array of values,
overwriting any values in the event of a collision.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
vector from which components are to be fetched
@param double *values
 **/
int HYPRE_IJVectorGetLocalComponents(HYPRE_IJVector  IJvector,
                                     int             num_values,
                                     const int      *glob_indices,
                                     const int      *value_indices,
                                     double         *values);

/**
 *
Puts components in specified local block of IJVector into indexed array of values, overwriti
ng any values in the event of a collision.

Not collective.

@return integer error code
@param HYPRE_IJVector &vector [IN]
the vector from which components are to be fetched
@param int glob_vec_index_start [IN]
@param int glob_vec_index_stop [IN]
@param int *value_indices [IN]
@param double *values
 **/
int HYPRE_IJVectorGetLocalComponentsInBlock(HYPRE_IJVector  IJvector,
                                            int             glob_index_start,
                                            int             glob_index_stop,
                                            const int      *value_indices,
                                            double         *values);



/*@}*/
/*@}*/

/*--------------------------------------------------------------------------
 * Miscellaneous
 *--------------------------------------------------------------------------*/

int HYPRE_IJVectorGetLocalStorageType(HYPRE_IJVector  IJvector,
                                      int            *type);

void *HYPRE_IJVectorGetLocalStorage(HYPRE_IJVector IJvector);

/* Internal routine only. */
int HYPRE_IJVectorDistribute(HYPRE_IJVector  IJvector,
                             const int      *vec_starts);

/* Internal routine only. */
int hypre_RefIJMatrix(HYPRE_IJMatrix  IJmatrix,
                      HYPRE_IJMatrix *reference);

/* Internal routine only. */
int hypre_RefIJVector(HYPRE_IJVector  IJvector,
                      HYPRE_IJVector *reference);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif
