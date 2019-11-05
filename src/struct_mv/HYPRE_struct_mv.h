/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_STRUCT_MV_HEADER
#define HYPRE_STRUCT_MV_HEADER

#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/* forward declarations */
#ifndef HYPRE_StructVector_defined
#define HYPRE_StructVector_defined
struct hypre_StructVector_struct;
typedef struct hypre_StructVector_struct *HYPRE_StructVector;
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct System Interface
 *
 * This interface represents a structured-grid conceptual view of a linear
 * system.
 *
 * @memo A structured-grid conceptual interface
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Grids
 **/
/*@{*/

struct hypre_StructGrid_struct;
/**
 * A grid object is constructed out of several ``boxes'', defined on a global
 * abstract index space.
 **/
typedef struct hypre_StructGrid_struct *HYPRE_StructGrid;

/**
 * Create an {\tt ndim}-dimensional grid object.
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
 * Set the periodicity for the grid.
 *
 * The argument {\tt periodic} is an {\tt ndim}-dimensional integer array that
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
 * Set the ghost layer in the grid object
 **/
HYPRE_Int HYPRE_StructGridSetNumGhost(HYPRE_StructGrid  grid,
                                      HYPRE_Int        *num_ghost);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Stencils
 **/
/*@{*/

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
 *
 * NOTE: The name of this routine will eventually be changed to {\tt
 * HYPRE\_StructStencilSetEntry}.
 **/
HYPRE_Int HYPRE_StructStencilSetElement(HYPRE_StructStencil  stencil,
                                        HYPRE_Int            entry,
                                        HYPRE_Int           *offset);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Matrices
 **/
/*@{*/

struct hypre_StructMatrix_struct;
/**
 * The matrix object.
 **/
typedef struct hypre_StructMatrix_struct *HYPRE_StructMatrix;

/**
 * Create a matrix object.
 **/
HYPRE_Int HYPRE_StructMatrixCreate(MPI_Comm             comm,
                                   HYPRE_StructGrid     grid,
                                   HYPRE_StructStencil  stencil,
                                   HYPRE_StructMatrix  *matrix);

/**
 * Destroy a matrix object.
 **/
HYPRE_Int HYPRE_StructMatrixDestroy(HYPRE_StructMatrix matrix);

/**
 * Prepare a matrix object for setting coefficient values.
 **/
HYPRE_Int HYPRE_StructMatrixInitialize(HYPRE_StructMatrix matrix);

/**
 * Set matrix coefficients index by index.  The {\tt values} array is of length
 * {\tt nentries}.
 *
 * NOTE: For better efficiency, use \Ref{HYPRE_StructMatrixSetBoxValues} to set
 * coefficients a box at a time.
 **/
HYPRE_Int HYPRE_StructMatrixSetValues(HYPRE_StructMatrix  matrix,
                                      HYPRE_Int          *index,
                                      HYPRE_Int           nentries,
                                      HYPRE_Int          *entries,
                                      HYPRE_Complex      *values);

/**
 * Add to matrix coefficients index by index.  The {\tt values} array is of
 * length {\tt nentries}.
 *
 * NOTE: For better efficiency, use \Ref{HYPRE_StructMatrixAddToBoxValues} to
 * set coefficients a box at a time.
 **/
HYPRE_Int HYPRE_StructMatrixAddToValues(HYPRE_StructMatrix  matrix,
                                        HYPRE_Int          *index,
                                        HYPRE_Int           nentries,
                                        HYPRE_Int          *entries,
                                        HYPRE_Complex      *values);

/**
 * Set matrix coefficients which are constant over the grid.  The {\tt values}
 * array is of length {\tt nentries}.
 **/
HYPRE_Int HYPRE_StructMatrixSetConstantValues(HYPRE_StructMatrix  matrix,
                                              HYPRE_Int           nentries,
                                              HYPRE_Int          *entries,
                                              HYPRE_Complex      *values);
/**
 * Add to matrix coefficients which are constant over the grid.  The {\tt
 * values} array is of length {\tt nentries}.
 **/
HYPRE_Int HYPRE_StructMatrixAddToConstantValues(HYPRE_StructMatrix  matrix,
                                                HYPRE_Int           nentries,
                                                HYPRE_Int          *entries,
                                                HYPRE_Complex      *values);

/**
 * Set matrix coefficients a box at a time.  The data in {\tt values} is ordered
 * as follows:
 *
   \begin{verbatim}
   m = 0;
   for (k = ilower[2]; k <= iupper[2]; k++)
      for (j = ilower[1]; j <= iupper[1]; j++)
         for (i = ilower[0]; i <= iupper[0]; i++)
            for (entry = 0; entry < nentries; entry++)
            {
               values[m] = ...;
               m++;
            }
   \end{verbatim}
 **/
HYPRE_Int HYPRE_StructMatrixSetBoxValues(HYPRE_StructMatrix  matrix,
                                         HYPRE_Int          *ilower,
                                         HYPRE_Int          *iupper,
                                         HYPRE_Int           nentries,
                                         HYPRE_Int          *entries,
                                         HYPRE_Complex      *values);
/**
 * Add to matrix coefficients a box at a time.  The data in {\tt values} is
 * ordered as in \Ref{HYPRE_StructMatrixSetBoxValues}.
 **/
HYPRE_Int HYPRE_StructMatrixAddToBoxValues(HYPRE_StructMatrix  matrix,
                                           HYPRE_Int          *ilower,
                                           HYPRE_Int          *iupper,
                                           HYPRE_Int           nentries,
                                           HYPRE_Int          *entries,
                                           HYPRE_Complex      *values);

/**
 * Set matrix coefficients a box at a time.  The {\tt values} array is logically
 * box shaped with value-box extents {\tt vilower} and {\tt viupper} that must
 * contain the set-box extents {\tt ilower} and {\tt iupper} .  The data in the
 * {\tt values} array is ordered as in \Ref{HYPRE_StructMatrixSetBoxValues}, but
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
 * Add to matrix coefficients a box at a time.  The data in {\tt values} is
 * ordered as in \Ref{HYPRE_StructMatrixSetBoxValues2}.
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
 * Get matrix coefficients index by index.  The {\tt values} array is of length
 * {\tt nentries}.
 *
 * NOTE: For better efficiency, use \Ref{HYPRE_StructMatrixGetBoxValues} to get
 * coefficients a box at a time.
 **/
HYPRE_Int HYPRE_StructMatrixGetValues(HYPRE_StructMatrix  matrix,
                                      HYPRE_Int          *index,
                                      HYPRE_Int           nentries,
                                      HYPRE_Int          *entries,
                                      HYPRE_Complex      *values);

/**
 * Get matrix coefficients a box at a time.  The data in {\tt values} is
 * ordered as in \Ref{HYPRE_StructMatrixSetBoxValues}.
 **/
HYPRE_Int HYPRE_StructMatrixGetBoxValues(HYPRE_StructMatrix  matrix,
                                         HYPRE_Int          *ilower,
                                         HYPRE_Int          *iupper,
                                         HYPRE_Int           nentries,
                                         HYPRE_Int          *entries,
                                         HYPRE_Complex      *values);

/**
 * Get matrix coefficients a box at a time.  The data in {\tt values} is
 * ordered as in \Ref{HYPRE_StructMatrixSetBoxValues2}.
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
 * to be ``constant over the grid'' yields significant memory savings because
 * the value for each declared entry will only be stored once.  However, not all
 * solvers are able to utilize this feature.
 *
 * Presently supported:
 * \begin{itemize}
 * \item no entries constant (this function need not be called)
 * \item all entries constant
 * \item all but the diagonal entry constant
 * \end{itemize}
 **/
HYPRE_Int HYPRE_StructMatrixSetConstantEntries( HYPRE_StructMatrix matrix,
                                                HYPRE_Int          nentries,
                                                HYPRE_Int         *entries );

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
 * Matvec operator.  This operation is  $y = \alpha A x + \beta y$ .
 * Note that you can do a simple matrix-vector multiply by setting
 * $\alpha=1$ and $\beta=0$.
 **/
HYPRE_Int HYPRE_StructMatrixMatvec ( HYPRE_Complex alpha,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector x,
                                     HYPRE_Complex beta,
                                     HYPRE_StructVector y );

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Vectors
 **/
/*@{*/

struct hypre_StructVector_struct;
/**
 * The vector object.
 **/
#ifndef HYPRE_StructVector_defined
typedef struct hypre_StructVector_struct *HYPRE_StructVector;
#endif

/**
 * Create a vector object.
 **/
HYPRE_Int HYPRE_StructVectorCreate(MPI_Comm            comm,
                                   HYPRE_StructGrid    grid,
                                   HYPRE_StructVector *vector);

/**
 * Destroy a vector object.
 **/
HYPRE_Int HYPRE_StructVectorDestroy(HYPRE_StructVector vector);

/**
 * Prepare a vector object for setting coefficient values.
 **/
HYPRE_Int HYPRE_StructVectorInitialize(HYPRE_StructVector vector);

/**
 * Set vector coefficients index by index.
 *
 * NOTE: For better efficiency, use \Ref{HYPRE_StructVectorSetBoxValues} to set
 * coefficients a box at a time.
 **/
HYPRE_Int HYPRE_StructVectorSetValues(HYPRE_StructVector  vector,
                                      HYPRE_Int          *index,
                                      HYPRE_Complex       value);

/**
 * Add to vector coefficients index by index.
 *
 * NOTE: For better efficiency, use \Ref{HYPRE_StructVectorAddToBoxValues} to
 * set coefficients a box at a time.
 **/
HYPRE_Int HYPRE_StructVectorAddToValues(HYPRE_StructVector  vector,
                                        HYPRE_Int          *index,
                                        HYPRE_Complex       value);

/**
 * Set vector coefficients a box at a time.  The data in {\tt values} is ordered
 * as follows:
 *
   \begin{verbatim}
   m = 0;
   for (k = ilower[2]; k <= iupper[2]; k++)
      for (j = ilower[1]; j <= iupper[1]; j++)
         for (i = ilower[0]; i <= iupper[0]; i++)
         {
            values[m] = ...;
            m++;
         }
   \end{verbatim}
 **/
HYPRE_Int HYPRE_StructVectorSetBoxValues(HYPRE_StructVector  vector,
                                         HYPRE_Int          *ilower,
                                         HYPRE_Int          *iupper,
                                         HYPRE_Complex      *values);
/**
 * Add to vector coefficients a box at a time.  The data in {\tt values} is
 * ordered as in \Ref{HYPRE_StructVectorSetBoxValues}.
 **/
HYPRE_Int HYPRE_StructVectorAddToBoxValues(HYPRE_StructVector  vector,
                                           HYPRE_Int          *ilower,
                                           HYPRE_Int          *iupper,
                                           HYPRE_Complex      *values);

/**
 * Set vector coefficients a box at a time.  The {\tt values} array is logically
 * box shaped with value-box extents {\tt vilower} and {\tt viupper} that must
 * contain the set-box extents {\tt ilower} and {\tt iupper} .  The data in the
 * {\tt values} array is ordered as in \Ref{HYPRE_StructVectorSetBoxValues}, but
 * based on the value-box extents.
 **/
HYPRE_Int HYPRE_StructVectorSetBoxValues2(HYPRE_StructVector  vector,
                                          HYPRE_Int          *ilower,
                                          HYPRE_Int          *iupper,
                                          HYPRE_Int          *vilower,
                                          HYPRE_Int          *viupper,
                                          HYPRE_Complex      *values);
/**
 * Add to vector coefficients a box at a time.  The data in {\tt values} is
 * ordered as in \Ref{HYPRE_StructVectorSetBoxValues2}.
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
 * NOTE: For better efficiency, use \Ref{HYPRE_StructVectorGetBoxValues} to get
 * coefficients a box at a time.
 **/
HYPRE_Int HYPRE_StructVectorGetValues(HYPRE_StructVector  vector,
                                      HYPRE_Int          *index,
                                      HYPRE_Complex      *value);

/**
 * Get vector coefficients a box at a time.  The data in {\tt values} is ordered
 * as in \Ref{HYPRE_StructVectorSetBoxValues}.
 **/
HYPRE_Int HYPRE_StructVectorGetBoxValues(HYPRE_StructVector  vector,
                                         HYPRE_Int          *ilower,
                                         HYPRE_Int          *iupper,
                                         HYPRE_Complex      *values);

/**
 * Get vector coefficients a box at a time.  The data in {\tt values} is ordered
 * as in \Ref{HYPRE_StructVectorSetBoxValues2}.
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

/*@}*/
/*@}*/

/*--------------------------------------------------------------------------
 * Miscellaneous: These probably do not belong in the interface.
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_StructMatrixGetGrid(HYPRE_StructMatrix  matrix,
                                    HYPRE_StructGrid   *grid);

struct hypre_CommPkg_struct;
typedef struct hypre_CommPkg_struct *HYPRE_CommPkg;

HYPRE_Int HYPRE_StructVectorSetNumGhost(HYPRE_StructVector  vector,
                                        HYPRE_Int          *num_ghost);

HYPRE_Int HYPRE_StructVectorSetConstantValues(HYPRE_StructVector vector,
                                              HYPRE_Complex      values);

HYPRE_Int HYPRE_StructVectorGetMigrateCommPkg(HYPRE_StructVector  from_vector,
                                              HYPRE_StructVector  to_vector,
                                              HYPRE_CommPkg      *comm_pkg);

HYPRE_Int HYPRE_StructVectorMigrate(HYPRE_CommPkg      comm_pkg,
                                    HYPRE_StructVector from_vector,
                                    HYPRE_StructVector to_vector);

HYPRE_Int HYPRE_CommPkgDestroy(HYPRE_CommPkg comm_pkg);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#if defined(HYPRE_USING_CUDA)
HYPRE_Int
HYPRE_StructGridSetDataLocation( HYPRE_StructGrid grid, HYPRE_Int data_location );
#endif

#ifdef __cplusplus
}
#endif

#endif

