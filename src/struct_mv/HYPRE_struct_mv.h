/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.11 $
 ***********************************************************************EHEADER*/



#ifndef HYPRE_STRUCT_MV_HEADER
#define HYPRE_STRUCT_MV_HEADER

#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
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
int HYPRE_StructGridCreate(MPI_Comm          comm,
                           int               ndim,
                           HYPRE_StructGrid *grid);

/**
 * Destroy a grid object.  An object should be explicitly destroyed using this
 * destructor when the user's code no longer needs direct access to it.  Once
 * destroyed, the object must not be referenced again.  Note that the object may
 * not be deallocated at the completion of this call, since there may be
 * internal package references to the object.  The object will then be destroyed
 * when all internal reference counts go to zero.
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
 * Set periodic.
 **/
int HYPRE_StructGridSetPeriodic(HYPRE_StructGrid  grid,
                                int              *periodic);

/**
 * Set the ghost layer in the grid object
 **/
int HYPRE_StructGridSetNumGhost(HYPRE_StructGrid  grid,
                                int              *num_ghost);

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
 * NOTE: The name of this routine will eventually be changed to {\tt
 * HYPRE\_StructStencilSetEntry}.
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
 * Set matrix coefficients index by index.  The {\tt values} array is of length
 * {\tt nentries}.
 *
 * NOTE: For better efficiency, use \Ref{HYPRE_StructMatrixSetBoxValues} to set
 * coefficients a box at a time.
 **/
int HYPRE_StructMatrixSetValues(HYPRE_StructMatrix  matrix,
                                int                *index,
                                int                 nentries,
                                int                *entries,
                                double             *values);

/**
 * Add to matrix coefficients index by index.  The {\tt values} array is of
 * length {\tt nentries}.
 *
 * NOTE: For better efficiency, use \Ref{HYPRE_StructMatrixAddToBoxValues} to
 * set coefficients a box at a time.
 **/
int HYPRE_StructMatrixAddToValues(HYPRE_StructMatrix  matrix,
                                  int                *index,
                                  int                 nentries,
                                  int                *entries,
                                  double             *values);

/**
 * Set matrix coefficients which are constant over the grid.  The {\tt values}
 * array is of length {\tt nentries}.
 **/
int HYPRE_StructMatrixSetConstantValues(HYPRE_StructMatrix  matrix,
                                   int                 nentries,
                                   int                *entries,
                                   double             *values);
/**
 * Add to matrix coefficients which are constant over the grid.  The {\tt
 * values} array is of length {\tt nentries}.
 **/
int HYPRE_StructMatrixAddToConstantValues(HYPRE_StructMatrix  matrix,
                                     int                 nentries,
                                     int                *entries,
                                     double             *values);

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
int HYPRE_StructMatrixSetBoxValues(HYPRE_StructMatrix  matrix,
                                   int                *ilower,
                                   int                *iupper,
                                   int                 nentries,
                                   int                *entries,
                                   double             *values);
/**
 * Add to matrix coefficients a box at a time.  The data in {\tt values} is
 * ordered as in \Ref{HYPRE_StructMatrixSetBoxValues}.
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
 * Define symmetry properties of the matrix.  By default, matrices are assumed
 * to be nonsymmetric.  Significant storage savings can be made if the matrix is
 * symmetric.
 **/
int HYPRE_StructMatrixSetSymmetric(HYPRE_StructMatrix  matrix,
                                   int                 symmetric);

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
int HYPRE_StructMatrixSetConstantEntries( HYPRE_StructMatrix matrix,
                                          int                nentries,
                                          int               *entries );

/**
 * Set the ghost layer in the matrix 
 **/
int HYPRE_StructMatrixSetNumGhost(HYPRE_StructMatrix  matrix,
                                  int                *num_ghost);


/**
 * Print the matrix to file.  This is mainly for debugging purposes.
 **/
int HYPRE_StructMatrixPrint(const char         *filename,
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
 * Clears the ghostvalues of vector object. Beneficial to users that re-assemble
 * a vector object (e.g., in time-stepping).
 **/
int HYPRE_StructVectorClearGhostValues(HYPRE_StructVector vector);


/**
 * Set vector coefficients index by index.
 *
 * NOTE: For better efficiency, use \Ref{HYPRE_StructVectorSetBoxValues} to set
 * coefficients a box at a time.
 **/
int HYPRE_StructVectorSetValues(HYPRE_StructVector  vector,
                                int                *index,
                                double              value);

/**
 * Add to vector coefficients index by index.
 *
 * NOTE: For better efficiency, use \Ref{HYPRE_StructVectorAddToBoxValues} to
 * set coefficients a box at a time.
 **/
int HYPRE_StructVectorAddToValues(HYPRE_StructVector  vector,
                                  int                *index,
                                  double              value);

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
int HYPRE_StructVectorSetBoxValues(HYPRE_StructVector  vector,
                                   int                *ilower,
                                   int                *iupper,
                                   double             *values);
/**
 * Add to vector coefficients a box at a time.  The data in {\tt values} is
 * ordered as in \Ref{HYPRE_StructVectorSetBoxValues}.
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
 *
 * NOTE: For better efficiency, use \Ref{HYPRE_StructVectorGetBoxValues} to get
 * coefficients a box at a time.
 **/
int HYPRE_StructVectorGetValues(HYPRE_StructVector  vector,
                                int                *index,
                                double             *value);

/**
 * Get vector coefficients a box at a time.  The data in {\tt values} is ordered
 * as in \Ref{HYPRE_StructVectorSetBoxValues}.
 **/
int HYPRE_StructVectorGetBoxValues(HYPRE_StructVector  vector,
                                   int                *ilower,
                                   int                *iupper,
                                   double             *values);

/**
 * Print the vector to file.  This is mainly for debugging purposes.
 **/
int HYPRE_StructVectorPrint(const char         *filename,
                            HYPRE_StructVector  vector,
                            int                 all);

/*@}*/
/*@}*/

/*--------------------------------------------------------------------------
 * Miscellaneous: These probably do not belong in the interface.
 *--------------------------------------------------------------------------*/

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

