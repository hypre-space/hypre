/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_StructMatrix interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_CreateStructMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_createstructmatrix)( int      *comm,
                                           long int *grid,
                                           long int *stencil,
                                           long int *matrix,
                                           int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_CreateStructMatrix( (MPI_Comm)             *comm,
                                  (HYPRE_StructGrid)     *grid,
                                  (HYPRE_StructStencil)  *stencil,
                                  (HYPRE_StructMatrix *) matrix   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DestroyStructMatrix
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_destroystructmatrix)( long int *matrix,
                                            int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_DestroyStructMatrix( (HYPRE_StructMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeStructMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_initializestructmatrix)( long int *matrix,
                                               int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_InitializeStructMatrix( (HYPRE_StructMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructMatrixValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setstructmatrixvalues)( long int *matrix,
                                              int      *grid_index,
                                              int      *num_stencil_indices,
                                              int      *stencil_indices,
                                              double   *values,
                                              int      *ierr                )
{
   *ierr = (int)
      ( HYPRE_SetStructMatrixValues( (HYPRE_StructMatrix) *matrix,
                                     (int *)              grid_index,
                                     (int)                *num_stencil_indices,
                                     (int *)              stencil_indices,
                                     (double *)           values           ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructMatrixBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setstructmatrixboxvalues)( long int *matrix,
                                                 int      *ilower,
                                                 int      *iupper,
                                                 int      *num_stencil_indices,
                                                 int      *stencil_indices,
                                                 double   *values,
                                                 int      *ierr              )
{
   *ierr = (int)
      ( HYPRE_SetStructMatrixBoxValues( (HYPRE_StructMatrix) *matrix,
                                        (int *)              ilower,
                                        (int *)              iupper,
                                        (int)                *num_stencil_indices,
                                        (int *)              stencil_indices,
                                        (double *)           values        ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AssembleStructMatrix
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_assemblestructmatrix)( long int *matrix,
                                             int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_AssembleStructMatrix( (HYPRE_StructMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructMatrixNumGhost
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_setstructmatrixnumghost)( long int *matrix,
                                                int      *num_ghost,
                                                int      *ierr      )
{
   *ierr = (int)
      ( HYPRE_SetStructMatrixNumGhost( (HYPRE_StructMatrix) *matrix,
                                       (int *)               num_ghost ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixGrid
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixgrid)( long int *matrix,
                                         long int *grid,
                                         int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructMatrixGrid( (HYPRE_StructMatrix) *matrix,
                                (HYPRE_StructGrid *) grid ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructMatrixSymmetric
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_setstructmatrixsymmetric)( long int *matrix,
                                                 int      *symmetric,
                                                 int      *ierr      )
{
   *ierr = (int)
      ( HYPRE_SetStructMatrixSymmetric( (HYPRE_StructMatrix) *matrix,
                                        (int)                *symmetric ) );
}
