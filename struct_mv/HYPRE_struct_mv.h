/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#ifndef HYPRE_STRUCT_MV_HEADER
#define HYPRE_STRUCT_MV_HEADER

#include "HYPRE_config.h"
#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct System Interface
 *
 * This interface represents a structured-grid conceptual view of a
 * linear system.
 *
 * @memo A structured-grid conceptual interface
 * @author Robert D. Falgout
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
 * A grid object is constructed out of several ``boxes'', defined on a
 * global abstract index space.
 **/
typedef struct hypre_StructGrid_struct *HYPRE_StructGrid;

/**
 * Create an {\tt ndim}-dimensional grid object.
 **/
int HYPRE_StructGridCreate(MPI_Comm          comm,
                           int               ndim,
                           HYPRE_StructGrid *grid);

/**
 * Destroy a grid object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
int HYPRE_StructGridDestroy(HYPRE_StructGrid grid);

/**
 * Set the extents for a box on the grid.
 **/
int HYPRE_StructGridSetExtents(HYPRE_StructGrid  grid,
                               int              *ilower,
                               int              *iupper);

/**
 * Finalize the construction of the grid before using.
 **/
int HYPRE_StructGridAssemble(HYPRE_StructGrid grid);

/**
 * (Optional) Set periodic.
 **/
int HYPRE_StructGridSetPeriodic(HYPRE_StructGrid  grid,
                                int              *periodic);

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
 * Create a stencil object for the specified number of spatial dimensions
 * and stencil entries.
 **/
int HYPRE_StructStencilCreate(int                  ndim,
                              int                  size,
                              HYPRE_StructStencil *stencil);

/**
 * Destroy a stencil object.
 **/
int HYPRE_StructStencilDestroy(HYPRE_StructStencil stencil);

/**
 * Set a stencil entry.
 *
 * NOTE: The name of this routine will eventually be changed to
 * {\tt HYPRE\_StructStencilSetEntry}.
 **/
int HYPRE_StructStencilSetElement(HYPRE_StructStencil  stencil,
                                  int                  entry,
                                  int                 *offset);

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
int HYPRE_StructMatrixCreate(MPI_Comm             comm,
                             HYPRE_StructGrid     grid,
                             HYPRE_StructStencil  stencil,
                             HYPRE_StructMatrix  *matrix);

/**
 * Destroy a matrix object.
 **/
int HYPRE_StructMatrixDestroy(HYPRE_StructMatrix matrix);

/**
 * Prepare a matrix object for setting coefficient values.
 **/
int HYPRE_StructMatrixInitialize(HYPRE_StructMatrix matrix);

/**
 * Set matrix coefficients index by index.
 **/
int HYPRE_StructMatrixSetValues(HYPRE_StructMatrix  matrix,
                                int                *index,
                                int                 nentries,
                                int                *entries,
                                double             *values);

/**
 * Set matrix coefficients a box at a time.
 **/
int HYPRE_StructMatrixSetBoxValues(HYPRE_StructMatrix  matrix,
                                   int                *ilower,
                                   int                *iupper,
                                   int                 nentries,
                                   int                *entries,
                                   double             *values);
/**
 * Add to matrix coefficients index by index.
 **/
int HYPRE_StructMatrixAddToValues(HYPRE_StructMatrix  matrix,
                                  int                *index,
                                  int                 nentries,
                                  int                *entries,
                                  double             *values);

/**
 * Add to matrix coefficients a box at a time.
 **/
int HYPRE_StructMatrixAddToBoxValues(HYPRE_StructMatrix  matrix,
                                     int                *ilower,
                                     int                *iupper,
                                     int                 nentries,
                                     int                *entries,
                                     double             *values);

/**
 * Finalize the construction of the matrix before using.
 **/
int HYPRE_StructMatrixAssemble(HYPRE_StructMatrix matrix);

/**
 * (Optional) Define symmetry properties of the matrix.  By default,
 * matrices are assumed to be nonsymmetric.  Significant storage
 * savings can be made if the matrix is symmetric.
 **/
int HYPRE_StructMatrixSetSymmetric(HYPRE_StructMatrix  matrix,
                                   int                 symmetric);

/**
 * Print the matrix to file.  This is mainly for debugging purposes.
 **/
int HYPRE_StructMatrixPrint(char               *filename,
                            HYPRE_StructMatrix  matrix,
                            int                 all);

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
typedef struct hypre_StructVector_struct *HYPRE_StructVector;

/**
 * Create a vector object.
 **/
int HYPRE_StructVectorCreate(MPI_Comm            comm,
                             HYPRE_StructGrid    grid,
                             HYPRE_StructVector *vector);

/**
 * Destroy a vector object.
 **/
int HYPRE_StructVectorDestroy(HYPRE_StructVector vector);

/**
 * Prepare a vector object for setting coefficient values.
 **/
int HYPRE_StructVectorInitialize(HYPRE_StructVector vector);

/**
 * Set vector coefficients index by index.
 **/
int HYPRE_StructVectorSetValues(HYPRE_StructVector  vector,
                                int                *index,
                                double              value);

/**
 * Set vector coefficients a box at a time.
 **/
int HYPRE_StructVectorSetBoxValues(HYPRE_StructVector  vector,
                                   int                *ilower,
                                   int                *iupper,
                                   double             *values);
/**
 * Set vector coefficients index by index.
 **/
int HYPRE_StructVectorAddToValues(HYPRE_StructVector  vector,
                                  int                *index,
                                  double              value);

/**
 * Set vector coefficients a box at a time.
 **/
int HYPRE_StructVectorAddToBoxValues(HYPRE_StructVector  vector,
                                     int                *ilower,
                                     int                *iupper,
                                     double             *values);

/**
 * Finalize the construction of the vector before using.
 **/
int HYPRE_StructVectorAssemble(HYPRE_StructVector vector);

/**
 * Get vector coefficients index by index.
 **/
int HYPRE_StructVectorGetValues(HYPRE_StructVector  vector,
                                int                *index,
                                double             *value);

/**
 * Get vector coefficients a box at a time.
 **/
int HYPRE_StructVectorGetBoxValues(HYPRE_StructVector  vector,
                                   int                *ilower,
                                   int                *iupper,
                                   double             *values);

/**
 * Print the vector to file.  This is mainly for debugging purposes.
 **/
int HYPRE_StructVectorPrint(char               *filename,
                            HYPRE_StructVector  vector,
                            int                 all);

/*@}*/
/*@}*/

/*--------------------------------------------------------------------------
 * Miscellaneous: These probably do not belong in the interface.
 *--------------------------------------------------------------------------*/

int HYPRE_StructMatrixSetNumGhost(HYPRE_StructMatrix  matrix,
                                  int                *num_ghost);

int HYPRE_StructMatrixGetGrid(HYPRE_StructMatrix  matrix,
                              HYPRE_StructGrid   *grid);

struct hypre_CommPkg_struct;
typedef struct hypre_CommPkg_struct *HYPRE_CommPkg;

int HYPRE_StructVectorSetNumGhost(HYPRE_StructVector  vector,
                                  int                *num_ghost);

int HYPRE_StructVectorSetConstantValues(HYPRE_StructVector vector,
                                        double             values);

int HYPRE_StructVectorGetMigrateCommPkg(HYPRE_StructVector  from_vector,
                                        HYPRE_StructVector  to_vector,
                                        HYPRE_CommPkg      *comm_pkg);

int HYPRE_StructVectorMigrate(HYPRE_CommPkg      comm_pkg,
                              HYPRE_StructVector from_vector,
                              HYPRE_StructVector to_vector);

int HYPRE_CommPkgDestroy(HYPRE_CommPkg comm_pkg);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif

