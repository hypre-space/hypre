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
 * @name Struct Linear Solvers Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A linear solver interface for structured grids
 * @version 1.0
 * @author Robert D. Falgout
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Grids
 *
 * A grid object is constructed out of several ``boxes'', defined on a
 * global abstract index space.
 **/
/*@{*/

/**
 * The {\tt HYPRE\_StructGrid} object ...
 **/
struct hypre_StructGrid_struct;
typedef struct hypre_StructGrid_struct *HYPRE_StructGrid;

/**
 * Create an {\tt ndim}-dimensional grid object.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructGridCreate(MPI_Comm          comm,
                           int               ndim,
                           HYPRE_StructGrid *grid);

/**
 * Destroy a grid object.  A grid should be explicitly destroyed using
 * this destructor when the user's code no longer needs direct access
 * to the grid description.  Once destroyed, the object must not be
 * referenced again.  Note that the grid description may not be
 * deallocated at the completion of this call, since there may be
 * internal package references to the object.  The grid will then be
 * destroyed when all internal reference counts go to zero.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructGridDestroy(HYPRE_StructGrid grid);

/**
 * Set the extents for a box on the grid.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructGridSetExtents(HYPRE_StructGrid  grid,
                               int              *ilower,
                               int              *iupper);

/**
 * Finalize the construction of the grid before using.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructGridAssemble(HYPRE_StructGrid grid);

/**
 * Set periodic.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructGridSetPeriodic(HYPRE_StructGrid  grid,
                                int              *periodic);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Stencils
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt HYPRE\_StructStencil} object ...
 **/
struct hypre_StructStencil_struct;
typedef struct hypre_StructStencil_struct *HYPRE_StructStencil;

/**
 * Create a stencil object for the specified number of spatial dimensions
 * and stencil entries.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructStencilCreate(int                  ndim,
                              int                  size,
                              HYPRE_StructStencil *stencil);

/**
 * Destroy the stencil object.  The stencil object should be explicitly
 * destroyed using this destructor when the user's code no longer needs to
 * directly access the stencil description.  Once destroyed, the object
 * must not be referenced again.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructStencilDestroy(HYPRE_StructStencil stencil);

/**
 * Set a stencil entry ...
 *
 * NOTE: The name of this routine will eventually be changed to
 * HYPRE\_StructStencilSetEntry.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructStencilSetElement(HYPRE_StructStencil  stencil,
                                  int                  entry,
                                  int                 *offset);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Matrices
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt HYPRE\_StructMatrix} object ...
 **/
struct hypre_StructMatrix_struct;
typedef struct hypre_StructMatrix_struct *HYPRE_StructMatrix;

/**
 * Create a matrix object.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructMatrixCreate(MPI_Comm             comm,
                             HYPRE_StructGrid     grid,
                             HYPRE_StructStencil  stencil,
                             HYPRE_StructMatrix  *matrix);

/**
 * Destroy a matrix object.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructMatrixDestroy(HYPRE_StructMatrix matrix);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructMatrixInitialize(HYPRE_StructMatrix matrix);

/**
 * Set matrix coefficients index by index.
 *
 * @param param [IN] ... 
 **/
int HYPRE_StructMatrixSetValues(HYPRE_StructMatrix  matrix,
                                int                *index,
                                int                 nentries,
                                int                *entries,
                                double             *values);

/**
 * Set matrix coefficients a box at a time.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructMatrixSetBoxValues(HYPRE_StructMatrix  matrix,
                                   int                *ilower,
                                   int                *iupper,
                                   int                 nentries,
                                   int                *entries,
                                   double             *values);
/**
 * Add to matrix coefficients index by index.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructMatrixAddToValues(HYPRE_StructMatrix  matrix,
                                  int                *index,
                                  int                 nentries,
                                  int                *entries,
                                  double             *values);

/**
 * Add to matrix coefficients a box at a time.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructMatrixAddToBoxValues(HYPRE_StructMatrix  matrix,
                                     int                *ilower,
                                     int                *iupper,
                                     int                 nentries,
                                     int                *entries,
                                     double             *values);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructMatrixAssemble(HYPRE_StructMatrix matrix);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructMatrixSetSymmetric(HYPRE_StructMatrix  matrix,
                                   int                 symmetric);

/**
 * Description...
 *
 * NOTE: This routine should not be in the interface.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructMatrixSetNumGhost(HYPRE_StructMatrix  matrix,
                                  int                *num_ghost);

/**
 * Description...
 *
 * NOTE: Not sure if this needs to be in the interface.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructMatrixGetGrid(HYPRE_StructMatrix  matrix,
                              HYPRE_StructGrid   *grid);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructMatrixPrint(char               *filename,
                            HYPRE_StructMatrix  matrix,
                            int                 all);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Vectors
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt HYPRE\_StructVector} object ...
 **/
struct hypre_StructVector_struct;
typedef struct hypre_StructVector_struct *HYPRE_StructVector;

/**
 * Create a vector object.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructVectorCreate(MPI_Comm            comm,
                             HYPRE_StructGrid    grid,
                             HYPRE_StructVector *vector);

/**
 * Destroy a vector object.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructVectorDestroy(HYPRE_StructVector vector);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructVectorInitialize(HYPRE_StructVector vector);

/**
 * Set vector coefficients index by index.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructVectorSetValues(HYPRE_StructVector  vector,
                                int                *index,
                                double              value);

/**
 * Set vector coefficients a box at a time.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructVectorSetBoxValues(HYPRE_StructVector  vector,
                                   int                *ilower,
                                   int                *iupper,
                                   double             *values);
/**
 * Set vector coefficients index by index.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructVectorAddToValues(HYPRE_StructVector  vector,
                                  int                *index,
                                  double              value);

/**
 * Set vector coefficients a box at a time.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructVectorAddToBoxValues(HYPRE_StructVector  vector,
                                     int                *ilower,
                                     int                *iupper,
                                     double             *values);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructVectorAssemble(HYPRE_StructVector vector);

/**
 * Description...
 *
 * NOTE: This routine should not be in the interface.
 *
 * @param param [IN] ...
 **/
int HYPRE_StructVectorSetNumGhost(HYPRE_StructVector  vector,
                                  int                *num_ghost);

/**
 * Description...
 *
 * @param param [IN] ... 
 **/
int HYPRE_StructVectorGetValues(HYPRE_StructVector  vector,
                                int                *index,
                                double             *value);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructVectorGetBoxValues(HYPRE_StructVector  vector,
                                   int                *ilower,
                                   int                *iupper,
                                   double             *values);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructVectorPrint(char               *filename,
                            HYPRE_StructVector  vector,
                            int                 all);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Miscellaneous
 *
 * This does not belong in the interface, and will be removed.
 **/
/*@{*/

/**
 * The {\tt HYPRE\_CommPkg} object ...
 **/
struct hypre_CommPkg_struct;
typedef struct hypre_CommPkg_struct *HYPRE_CommPkg;

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructVectorSetConstantValues(HYPRE_StructVector vector,
                                        double             values);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructVectorGetMigrateCommPkg(HYPRE_StructVector  from_vector,
                                        HYPRE_StructVector  to_vector,
                                        HYPRE_CommPkg      *comm_pkg);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructVectorMigrate(HYPRE_CommPkg      comm_pkg,
                              HYPRE_StructVector from_vector,
                              HYPRE_StructVector to_vector);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_CommPkgDestroy(HYPRE_CommPkg comm_pkg);

/*@}*/
/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif

