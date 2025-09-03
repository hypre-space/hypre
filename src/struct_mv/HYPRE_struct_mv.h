/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_STRUCT_MV_HEADER
#define HYPRE_STRUCT_MV_HEADER

#include "HYPRE_utilities.h"

#ifdef HYPRE_MIXED_PRECISION
#include "_hypre_struct_mv_mup_def.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_Index: public interface to hypre_Index, that is to define indices
 * in index space, or dimension sizes of boxes.
 *--------------------------------------------------------------------------*/

typedef HYPRE_Int  HYPRE_Index[HYPRE_MAXDIM];
typedef HYPRE_Int *HYPRE_IndexRef;

/* forward declarations */
#ifndef HYPRE_StructVector_defined
#define HYPRE_StructVector_defined
struct hypre_StructVector_struct;
typedef struct hypre_StructVector_struct *HYPRE_StructVector;
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @defgroup StructSystemInterface Struct System Interface
 *
 * A structured-grid conceptual interface. This interface represents a
 * structured-grid conceptual view of a linear system.
 *
 * @{
 **/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Grids
 *
 * @{
 **/

struct hypre_StructGrid_struct;
/**
 * A grid object is constructed out of several "boxes", defined on a global
 * abstract index space.
 **/
typedef struct hypre_StructGrid_struct *HYPRE_StructGrid;

/**
 * Create an <em>ndim</em>-dimensional grid object.
 **/
HYPRE_Int HYPRE_StructGridCreate(MPI_Comm          comm,
                                 HYPRE_Int         ndim,
                                 HYPRE_StructGrid *grid);

/**
 * Destroy a grid object.  An object should be explicitly destroyed using this
 * destructor when the user's code no longer needs direct access to it.  Once
 * destroyed, the object must not be referenced again.  Note that the object may
 * not be deallocated at the completion of this call, since there may be
 * internal package references to the object.  The object will then be destroyed
 * when all internal reference counts go to zero.
 **/
HYPRE_Int HYPRE_StructGridDestroy(HYPRE_StructGrid grid);

/**
 * Set the extents for a box on the grid.
 **/
HYPRE_Int HYPRE_StructGridSetExtents(HYPRE_StructGrid  grid,
                                     HYPRE_Int        *ilower,
                                     HYPRE_Int        *iupper);

/**
 * Finalize the construction of the grid before using.
 **/
HYPRE_Int HYPRE_StructGridAssemble(HYPRE_StructGrid grid);

/**
 * Prints a grid in VTK format.
 **/
HYPRE_Int HYPRE_StructGridPrintVTK(const char       *filename,
                                   HYPRE_StructGrid  grid);

/**
 * Set the periodicity for the grid.
 *
 * The argument \e periodic is an <em>ndim</em>-dimensional integer array that
 * contains the periodicity for each dimension.  A zero value for a dimension
 * means non-periodic, while a nonzero value means periodic and contains the
 * actual period.  For example, periodicity in the first and third dimensions
 * for a 10x11x12 grid is indicated by the array [10,0,12].
 *
 * NOTE: Some of the solvers in hypre have power-of-two restrictions on the size
 * of the periodic dimensions.
 **/
HYPRE_Int HYPRE_StructGridSetPeriodic(HYPRE_StructGrid  grid,
                                      HYPRE_Int        *periodic);

/**
 * Set the ghost layer in the grid object.
 **/
HYPRE_Int HYPRE_StructGridSetNumGhost(HYPRE_StructGrid  grid,
                                      HYPRE_Int        *num_ghost);

/**
 * Coarsen \e grid by factor \e stride to create \e cgrid.
 **/
HYPRE_Int HYPRE_StructGridCoarsen(HYPRE_StructGrid  grid,
                                  HYPRE_Int        *stride,
                                  HYPRE_StructGrid *cgrid);

/**
 * Project the box described by \e ilower and \e iupper onto the strided
 * index space that contains the index \e origin and has stride \e stride.
 * This routine is useful in combination with \ref HYPRE_StructGridCoarsen when
 * dealing with rectangular matrices.
 **/
HYPRE_Int
HYPRE_StructGridProjectBox(HYPRE_StructGrid  grid,
                           HYPRE_Int        *ilower,
                           HYPRE_Int        *iupper,
                           HYPRE_Int        *origin,
                           HYPRE_Int        *stride);


/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Stencils
 *
 * @{
 **/

struct hypre_StructStencil_struct;
/**
 * The stencil object.
 **/
typedef struct hypre_StructStencil_struct *HYPRE_StructStencil;

/**
 * Create a stencil object for the specified number of spatial dimensions and
 * stencil entries.
 **/
HYPRE_Int HYPRE_StructStencilCreate(HYPRE_Int            ndim,
                                    HYPRE_Int            size,
                                    HYPRE_StructStencil *stencil);

/**
 * Destroy a stencil object.
 **/
HYPRE_Int HYPRE_StructStencilDestroy(HYPRE_StructStencil stencil);

/**
 * Set a stencil entry.
 **/
HYPRE_Int HYPRE_StructStencilSetEntry(HYPRE_StructStencil  stencil,
                                      HYPRE_Int            entry,
                                      HYPRE_Int           *offset);

/*
 * OBSOLETE.  Use SetEntry instead.
 **/
HYPRE_Int HYPRE_StructStencilSetElement(HYPRE_StructStencil  stencil,
                                        HYPRE_Int            entry,
                                        HYPRE_Int           *offset);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Matrices
 *
 * @{
 **/

struct hypre_StructMatrix_struct;
/**
 * The matrix object.
 **/
typedef struct hypre_StructMatrix_struct *HYPRE_StructMatrix;

/**
 * Create a matrix object.  Matrices may have different range and domain grids,
 * that is, they need not be square.  By default, the range and domain grids are
 * the same as \e grid.  In general, the range is a coarsening of \e grid
 * as specified in \ref HYPRE_StructMatrixSetRangeStride, and similarly for the
 * domain.  Note that the range index space must either be a subspace of the
 * domain index space or vice versa.  Also, (currently) either the range or
 * domain coarsening factor (or both) must be all ones (i.e., no coarsening).
 **/
HYPRE_Int HYPRE_StructMatrixCreate(MPI_Comm             comm,
                                   HYPRE_StructGrid     grid,
                                   HYPRE_StructStencil  stencil,
                                   HYPRE_StructMatrix  *matrix);

/**
 * Destroy a matrix object.
 **/
HYPRE_Int HYPRE_StructMatrixDestroy(HYPRE_StructMatrix matrix);

#if 0
/**
 * (Optional) Set the domain grid.  By default, the range and domain grids are
 * the same as the argument \e grid in \ref HYPRE_StructMatrixCreate.  Both
 * grids live on a common fine index space and should have the same number of
 * boxes.  The actual range is a coarsening of the range grid with coarsening
 * factor \e rstride specified in \ref HYPRE_StructMatrixSetRStride.
 * Similarly, the actual domain is a coarsening of the domain grid with factor
 * \e dstride specified in \ref HYPRE_StructMatrixSetDStride.  Currently,
 * either \e rstride or \e dstride or both must be all ones (i.e., no
 * coarsening).
 **/
HYPRE_Int HYPRE_StructMatrixSetDomainGrid(HYPRE_StructMatrix matrix,
                                          HYPRE_StructGrid   domain_grid);
#endif

/* RDF: Need a good user interface for setting range/domain grids. */

/**
 * (Optional) Set the range coarsening stride.  For more information, see
 * \ref HYPRE_StructMatrixCreate.
 **/
HYPRE_Int HYPRE_StructMatrixSetRangeStride(HYPRE_StructMatrix matrix,
                                           HYPRE_Int         *range_stride);

/**
 * (Optional) Set the domain coarsening stride.  For more information, see
 * \ref HYPRE_StructMatrixCreate.
 **/
HYPRE_Int HYPRE_StructMatrixSetDomainStride(HYPRE_StructMatrix matrix,
                                            HYPRE_Int         *domain_stride);

/**
 * Prepare a matrix object for setting coefficient values.
 **/
HYPRE_Int HYPRE_StructMatrixInitialize(HYPRE_StructMatrix matrix);

/**
 * Set matrix coefficients index by index.  The \e values array is of length
 * \e nentries.
 *
 * NOTE: For better efficiency, use \ref HYPRE_StructMatrixSetBoxValues to set
 * coefficients a box at a time.
 **/
HYPRE_Int HYPRE_StructMatrixSetValues(HYPRE_StructMatrix  matrix,
                                      HYPRE_Int          *index,
                                      HYPRE_Int           nentries,
                                      HYPRE_Int          *entries,
                                      HYPRE_Complex      *values);

/**
 * Add to matrix coefficients index by index.  The \e values array is of
 * length \e nentries.
 *
 * NOTE: For better efficiency, use \ref HYPRE_StructMatrixAddToBoxValues to
 * set coefficients a box at a time.
 **/
HYPRE_Int HYPRE_StructMatrixAddToValues(HYPRE_StructMatrix  matrix,
                                        HYPRE_Int          *index,
                                        HYPRE_Int           nentries,
                                        HYPRE_Int          *entries,
                                        HYPRE_Complex      *values);

/**
 * Set matrix coefficients which are constant over the grid.  The \e values
 * array is of length \e nentries.
 **/
HYPRE_Int HYPRE_StructMatrixSetConstantValues(HYPRE_StructMatrix  matrix,
                                              HYPRE_Int           nentries,
                                              HYPRE_Int          *entries,
                                              HYPRE_Complex      *values);

/**
 * Add to matrix coefficients which are constant over the grid.  The \e
 * values array is of length \e nentries.
 **/
HYPRE_Int HYPRE_StructMatrixAddToConstantValues(HYPRE_StructMatrix  matrix,
                                                HYPRE_Int           nentries,
                                                HYPRE_Int          *entries,
                                                HYPRE_Complex      *values);

/**
 * Set matrix coefficients a box at a time.  The data in \e values is ordered
 * as follows:
 *
   \verbatim
   m = 0;
   for (k = ilower[2]; k <= iupper[2]; k++)
      for (j = ilower[1]; j <= iupper[1]; j++)
         for (i = ilower[0]; i <= iupper[0]; i++)
            for (entry = 0; entry < nentries; entry++)
            {
               values[m] = ...;
               m++;
            }
   \endverbatim
 **/
HYPRE_Int HYPRE_StructMatrixSetBoxValues(HYPRE_StructMatrix  matrix,
                                         HYPRE_Int          *ilower,
                                         HYPRE_Int          *iupper,
                                         HYPRE_Int           nentries,
                                         HYPRE_Int          *entries,
                                         HYPRE_Complex      *values);
/**
 * Add to matrix coefficients a box at a time.  The data in \e values is
 * ordered as in \ref HYPRE_StructMatrixSetBoxValues.
 **/
HYPRE_Int HYPRE_StructMatrixAddToBoxValues(HYPRE_StructMatrix  matrix,
                                           HYPRE_Int          *ilower,
                                           HYPRE_Int          *iupper,
                                           HYPRE_Int           nentries,
                                           HYPRE_Int          *entries,
                                           HYPRE_Complex      *values);

/**
 * Set matrix coefficients a box at a time.  The \e values array is logically
 * box shaped with value-box extents \e vilower and \e viupper that must
 * contain the set-box extents \e ilower and \e iupper .  The data in the
 * \e values array is ordered as in \ref HYPRE_StructMatrixSetBoxValues, but
 * based on the value-box extents.
 **/
HYPRE_Int HYPRE_StructMatrixSetBoxValues2(HYPRE_StructMatrix  matrix,
                                          HYPRE_Int          *ilower,
                                          HYPRE_Int          *iupper,
                                          HYPRE_Int           nentries,
                                          HYPRE_Int          *entries,
                                          HYPRE_Int          *vilower,
                                          HYPRE_Int          *viupper,
                                          HYPRE_Complex      *values);
/**
 * Add to matrix coefficients a box at a time.  The data in \e values is
 * ordered as in \ref HYPRE_StructMatrixSetBoxValues2.
 **/
HYPRE_Int HYPRE_StructMatrixAddToBoxValues2(HYPRE_StructMatrix  matrix,
                                            HYPRE_Int          *ilower,
                                            HYPRE_Int          *iupper,
                                            HYPRE_Int           nentries,
                                            HYPRE_Int          *entries,
                                            HYPRE_Int          *vilower,
                                            HYPRE_Int          *viupper,
                                            HYPRE_Complex      *values);

/**
 * Finalize the construction of the matrix before using.
 **/
HYPRE_Int HYPRE_StructMatrixAssemble(HYPRE_StructMatrix matrix);

/**
 * Get matrix coefficients index by index.  The \e values array is of length
 * \e nentries.
 *
 * NOTE: For better efficiency, use \ref HYPRE_StructMatrixGetBoxValues to get
 * coefficients a box at a time.
 **/
HYPRE_Int HYPRE_StructMatrixGetValues(HYPRE_StructMatrix  matrix,
                                      HYPRE_Int          *index,
                                      HYPRE_Int           nentries,
                                      HYPRE_Int          *entries,
                                      HYPRE_Complex      *values);

/**
 * Get matrix coefficients a box at a time.  The data in \e values is
 * ordered as in \ref HYPRE_StructMatrixSetBoxValues.
 **/
HYPRE_Int HYPRE_StructMatrixGetBoxValues(HYPRE_StructMatrix  matrix,
                                         HYPRE_Int          *ilower,
                                         HYPRE_Int          *iupper,
                                         HYPRE_Int           nentries,
                                         HYPRE_Int          *entries,
                                         HYPRE_Complex      *values);

/**
 * Get matrix coefficients a box at a time.  The data in \e values is
 * ordered as in \ref HYPRE_StructMatrixSetBoxValues2.
 **/
HYPRE_Int HYPRE_StructMatrixGetBoxValues2(HYPRE_StructMatrix  matrix,
                                          HYPRE_Int          *ilower,
                                          HYPRE_Int          *iupper,
                                          HYPRE_Int           nentries,
                                          HYPRE_Int          *entries,
                                          HYPRE_Int          *vilower,
                                          HYPRE_Int          *viupper,
                                          HYPRE_Complex      *values);

/**
 * Define symmetry properties of the matrix.  By default, matrices are assumed
 * to be nonsymmetric.  Significant storage savings can be made if the matrix is
 * symmetric.
 **/
HYPRE_Int HYPRE_StructMatrixSetSymmetric(HYPRE_StructMatrix  matrix,
                                         HYPRE_Int           symmetric);

/**
 * Specify which stencil entries are constant over the grid.  Declaring entries
 * to be "constant over the grid" yields significant memory savings because
 * the value for each declared entry will only be stored once.  However, not all
 * solvers are able to utilize this feature.
 *
 * Presently supported:
 *    - no entries constant (this function need not be called)
 *    - all entries constant
 *    - all but the diagonal entry constant
 **/
HYPRE_Int HYPRE_StructMatrixSetConstantEntries( HYPRE_StructMatrix matrix,
                                                HYPRE_Int          nentries,
                                                HYPRE_Int         *entries );

/**
 * Indicate whether the transpose coefficients should also be stored.
 **/
HYPRE_Int HYPRE_StructMatrixSetTranspose( HYPRE_StructMatrix  matrix,
                                          HYPRE_Int           transpose );

/**
 * Set the ghost layer in the matrix
 **/
HYPRE_Int HYPRE_StructMatrixSetNumGhost(HYPRE_StructMatrix  matrix,
                                        HYPRE_Int          *num_ghost);


/**
 * Print the matrix to file.  This is mainly for debugging purposes.
 **/
HYPRE_Int HYPRE_StructMatrixPrint(const char         *filename,
                                  HYPRE_StructMatrix  matrix,
                                  HYPRE_Int           all);

/**
 * Read the matrix from file.  This is mainly for debugging purposes.
 **/
HYPRE_Int HYPRE_StructMatrixRead( MPI_Comm             comm,
                                  const char          *filename,
                                  HYPRE_Int           *num_ghost,
                                  HYPRE_StructMatrix  *matrix );

/**
 * Matvec operator.  This operation is \f$y = \alpha A x + \beta y\f$ .
 * Note that you can do a simple matrix-vector multiply by setting
 * \f$\alpha=1\f$ and \f$\beta=0\f$.
 **/
HYPRE_Int HYPRE_StructMatrixMatvec( HYPRE_Complex alpha,
                                    HYPRE_StructMatrix A,
                                    HYPRE_StructVector x,
                                    HYPRE_Complex beta,
                                    HYPRE_StructVector y );

/**
 * Matvec transpose operation.  This operation is \f$y = \alpha A^T x + \beta y\f$.
 * Note that you can do a simple matrix-vector multiply by setting \f$\alpha=1\f$
 * and \f$\beta=0\f$.
 **/
HYPRE_Int HYPRE_StructMatrixMatvecT( HYPRE_Complex alpha,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector x,
                                     HYPRE_Complex beta,
                                     HYPRE_StructVector y );

/**
 * Matrix-matrix multiply.  Returns \f$C=AB\f$, \f$C=A^TB\f$, \f$C=AB^T\f$, or
 * \f$C=A^TB^T\f$, depending on the boolean arguments \e Atranspose and \e Btranspose.
 **/
HYPRE_Int HYPRE_StructMatrixMatmat( HYPRE_StructMatrix  A,
                                    HYPRE_Int           Atranspose,
                                    HYPRE_StructMatrix  B,
                                    HYPRE_Int           Btranspose,
                                    HYPRE_StructMatrix *C );

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Vectors
 *
 * @{
 **/

struct hypre_StructVector_struct;
/**
 * The vector object.
 **/
#ifndef HYPRE_StructVector_defined
typedef struct hypre_StructVector_struct *HYPRE_StructVector;
#endif

/**
 * Create a vector object.  Similarly to matrices, the grid is in general a
 * coarsening of \e grid as specified by \ref HYPRE_StructVectorSetStride.
 * By default, the two are the same (the stride is one).
 **/
HYPRE_Int HYPRE_StructVectorCreate(MPI_Comm            comm,
                                   HYPRE_StructGrid    grid,
                                   HYPRE_StructVector *vector);

/**
 * Destroy a vector object.
 **/
HYPRE_Int HYPRE_StructVectorDestroy(HYPRE_StructVector vector);

/* RDF: Need a good user interface for setting the grid. */

/**
 * (Optional) Set the coarsening stride.  For more information, see
 * \ref HYPRE_StructVectorCreate.
 **/
HYPRE_Int HYPRE_StructVectorSetStride(HYPRE_StructVector vector,
                                      HYPRE_Int         *stride);

/**
 * Prepare a vector object for setting coefficient values.
 **/
HYPRE_Int HYPRE_StructVectorInitialize(HYPRE_StructVector vector);

/**
 * Set vector coefficients index by index.
 *
 * NOTE: For better efficiency, use \ref HYPRE_StructVectorSetBoxValues to set
 * coefficients a box at a time.
 **/
HYPRE_Int HYPRE_StructVectorSetValues(HYPRE_StructVector  vector,
                                      HYPRE_Int          *index,
                                      HYPRE_Complex      *values);

/**
 * Set vector coefficients to a constant value over the grid.
 **/
HYPRE_Int HYPRE_StructVectorSetConstantValues(HYPRE_StructVector vector,
                                              HYPRE_Complex      value);

/**
 * Set vector coefficients to random values between -1.0 and 1.0 over the grid.
 * The parameter \e seed controls the generation of random numbers.
 **/
HYPRE_Int HYPRE_StructVectorSetRandomValues(HYPRE_StructVector  vector,
                                            HYPRE_Int           seed);

/**
 * Add to vector coefficients index by index.
 *
 * NOTE: For better efficiency, use \ref HYPRE_StructVectorAddToBoxValues to
 * set coefficients a box at a time.
 **/
HYPRE_Int HYPRE_StructVectorAddToValues(HYPRE_StructVector  vector,
                                        HYPRE_Int          *index,
                                        HYPRE_Complex      *values);

/**
 * Set vector coefficients a box at a time.  The data in \e values is ordered
 * as follows:
 *
   \verbatim
   m = 0;
   for (k = ilower[2]; k <= iupper[2]; k++)
      for (j = ilower[1]; j <= iupper[1]; j++)
         for (i = ilower[0]; i <= iupper[0]; i++)
         {
            values[m] = ...;
            m++;
         }
   \endverbatim
 **/
HYPRE_Int HYPRE_StructVectorSetBoxValues(HYPRE_StructVector  vector,
                                         HYPRE_Int          *ilower,
                                         HYPRE_Int          *iupper,
                                         HYPRE_Complex      *values);
/**
 * Add to vector coefficients a box at a time.  The data in \e values is
 * ordered as in \ref HYPRE_StructVectorSetBoxValues.
 **/
HYPRE_Int HYPRE_StructVectorAddToBoxValues(HYPRE_StructVector  vector,
                                           HYPRE_Int          *ilower,
                                           HYPRE_Int          *iupper,
                                           HYPRE_Complex      *values);

/**
 * Set vector coefficients a box at a time.  The \e values array is logically
 * box shaped with value-box extents \e vilower and \e viupper that must
 * contain the set-box extents \e ilower and \e iupper .  The data in the
 * \e values array is ordered as in \ref HYPRE_StructVectorSetBoxValues, but
 * based on the value-box extents.
 **/
HYPRE_Int HYPRE_StructVectorSetBoxValues2(HYPRE_StructVector  vector,
                                          HYPRE_Int          *ilower,
                                          HYPRE_Int          *iupper,
                                          HYPRE_Int          *vilower,
                                          HYPRE_Int          *viupper,
                                          HYPRE_Complex      *values);
/**
 * Add to vector coefficients a box at a time.  The data in \e values is
 * ordered as in \ref HYPRE_StructVectorSetBoxValues2.
 **/
HYPRE_Int HYPRE_StructVectorAddToBoxValues2(HYPRE_StructVector  vector,
                                            HYPRE_Int          *ilower,
                                            HYPRE_Int          *iupper,
                                            HYPRE_Int          *vilower,
                                            HYPRE_Int          *viupper,
                                            HYPRE_Complex      *values);

/**
 * Finalize the construction of the vector before using.
 **/
HYPRE_Int HYPRE_StructVectorAssemble(HYPRE_StructVector vector);

/**
 * Get vector coefficients index by index.
 *
 * NOTE: For better efficiency, use \ref HYPRE_StructVectorGetBoxValues to get
 * coefficients a box at a time.
 **/
HYPRE_Int HYPRE_StructVectorGetValues(HYPRE_StructVector  vector,
                                      HYPRE_Int          *index,
                                      HYPRE_Complex      *value);

/**
 * Get vector coefficients a box at a time.  The data in \e values is ordered
 * as in \ref HYPRE_StructVectorSetBoxValues.
 **/
HYPRE_Int HYPRE_StructVectorGetBoxValues(HYPRE_StructVector  vector,
                                         HYPRE_Int          *ilower,
                                         HYPRE_Int          *iupper,
                                         HYPRE_Complex      *values);

/**
 * Get vector coefficients a box at a time.  The data in \e values is ordered
 * as in \ref HYPRE_StructVectorSetBoxValues2.
 **/
HYPRE_Int HYPRE_StructVectorGetBoxValues2(HYPRE_StructVector  vector,
                                          HYPRE_Int          *ilower,
                                          HYPRE_Int          *iupper,
                                          HYPRE_Int          *vilower,
                                          HYPRE_Int          *viupper,
                                          HYPRE_Complex      *values);

/**
 * Print the vector to file.  This is mainly for debugging purposes.
 **/
HYPRE_Int HYPRE_StructVectorPrint(const char         *filename,
                                  HYPRE_StructVector  vector,
                                  HYPRE_Int           all);

/**
 * Read the vector from file.  This is mainly for debugging purposes.
 **/
HYPRE_Int HYPRE_StructVectorRead( MPI_Comm             comm,
                                  const char          *filename,
                                  HYPRE_Int           *num_ghost,
                                  HYPRE_StructVector  *vector );

/**
 * Clone a vector x.
 **/
HYPRE_Int HYPRE_StructVectorClone( HYPRE_StructVector x,
                                   HYPRE_StructVector *y_ptr );

/**
 * Compute \e result, the inner product of vectors \e x and \e y.
 **/
HYPRE_Int HYPRE_StructVectorInnerProd( HYPRE_StructVector  x,
                                       HYPRE_StructVector  y,
                                       HYPRE_Real         *result );

/* Revisit these interface routines */
HYPRE_Int HYPRE_StructVectorScaleValues ( HYPRE_StructVector vector, HYPRE_Complex factor );
HYPRE_Int HYPRE_StructVectorCopy ( HYPRE_StructVector x, HYPRE_StructVector y );

/*@}*/
/*@}*/

/*--------------------------------------------------------------------------
 * Miscellaneous: These probably do not belong in the interface.
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_StructMatrixGetGrid(HYPRE_StructMatrix  matrix,
                                    HYPRE_StructGrid   *grid);
HYPRE_Int HYPRE_StructMatrixClearBoundary( HYPRE_StructMatrix matrix );

struct hypre_CommPkg_struct;
typedef struct hypre_CommPkg_struct *HYPRE_CommPkg;

HYPRE_Int HYPRE_StructVectorSetNumGhost(HYPRE_StructVector  vector,
                                        HYPRE_Int          *num_ghost);

HYPRE_Int HYPRE_StructVectorGetMigrateCommPkg(HYPRE_StructVector  from_vector,
                                              HYPRE_StructVector  to_vector,
                                              HYPRE_CommPkg      *comm_pkg);

HYPRE_Int HYPRE_StructVectorMigrate(HYPRE_CommPkg      comm_pkg,
                                    HYPRE_StructVector from_vector,
                                    HYPRE_StructVector to_vector);

HYPRE_Int HYPRE_CommPkgDestroy(HYPRE_CommPkg comm_pkg);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
HYPRE_Int
HYPRE_StructGridSetDataLocation( HYPRE_StructGrid grid, HYPRE_MemoryLocation data_location );
#endif

#ifdef __cplusplus
}
#endif

#ifdef HYPRE_MIXED_PRECISION
/* The following is for user compiles and the order is important.  The first
 * header ensures that we do not change prototype names in user files or in the
 * second header file.  The second header contains all the prototypes needed by
 * users for mixed precision. */
#ifndef hypre_MP_BUILD
#include "_hypre_struct_mv_mup_undef.h"
#include "HYPRE_struct_mv_mup.h"
#include "HYPRE_struct_mv_mp.h"
#endif
#endif

#endif
