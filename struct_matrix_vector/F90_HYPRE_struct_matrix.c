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
 * HYPRE_StructMatrixCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixcreate)( int      *comm,
                                           long int *grid,
                                           long int *stencil,
                                           long int *matrix,
                                           int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructMatrixCreate( (MPI_Comm)             *comm,
                                  (HYPRE_StructGrid)     *grid,
                                  (HYPRE_StructStencil)  *stencil,
                                  (HYPRE_StructMatrix *) matrix   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixdestroy)( long int *matrix,
                                            int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructMatrixDestroy( (HYPRE_StructMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixinitialize)( long int *matrix,
                                               int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructMatrixInitialize( (HYPRE_StructMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixsetvalues)( long int *matrix,
                                              int      *grid_index,
                                              int      *num_stencil_indices,
                                              int      *stencil_indices,
                                              double   *values,
                                              int      *ierr                )
{
   *ierr = (int)
      ( HYPRE_StructMatrixSetValues( (HYPRE_StructMatrix) *matrix,
                                     (int *)              grid_index,
                                     (int)                *num_stencil_indices,
                                     (int *)              stencil_indices,
                                     (double *)           values           ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixsetboxvalues)( long int *matrix,
                                                 int      *ilower,
                                                 int      *iupper,
                                                 int      *num_stencil_indices,
                                                 int      *stencil_indices,
                                                 double   *values,
                                                 int      *ierr              )
{
   *ierr = (int)
      ( HYPRE_StructMatrixSetBoxValues( (HYPRE_StructMatrix) *matrix,
                                        (int *)              ilower,
                                        (int *)              iupper,
                                        (int)                *num_stencil_indices,
                                        (int *)              stencil_indices,
                                        (double *)           values        ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAddToValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixaddtovalues)( long int *matrix,
                                                int      *grid_index,
                                                int      *num_stencil_indices,
                                                int      *stencil_indices,
                                                double   *values,
                                                int      *ierr                )
{
   *ierr = (int)
      ( HYPRE_StructMatrixAddToValues( (HYPRE_StructMatrix) *matrix,
                                       (int *)     grid_index,
                                       (int)       *num_stencil_indices,
                                       (int *)     stencil_indices,
                                       (double *)  values           ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAddToBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixaddtoboxvalues)( long int *matrix,
                                                   int    *ilower,
                                                   int    *iupper,
                                                   int    *num_stencil_indices,
                                                   int    *stencil_indices,
                                                   double *values,
                                                   int    *ierr              )
{
   *ierr = (int)
      ( HYPRE_StructMatrixAddToBoxValues( (HYPRE_StructMatrix) *matrix,
                                          (int *)     ilower,
                                          (int *)     iupper,
                                          (int)       *num_stencil_indices,
                                          (int *)     stencil_indices,
                                          (double *)  values        ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAssemble
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixassemble)( long int *matrix,
                                             int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructMatrixAssemble( (HYPRE_StructMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetNumGhost
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structmatrixsetnumghost)( long int *matrix,
                                                int      *num_ghost,
                                                int      *ierr      )
{
   *ierr = (int)
      ( HYPRE_StructMatrixSetNumGhost( (HYPRE_StructMatrix) *matrix,
                                       (int *)               num_ghost ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixGetGrid
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixgetgrid)( long int *matrix,
                                            long int *grid,
                                            int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructMatrixGetGrid( (HYPRE_StructMatrix) *matrix,
                                (HYPRE_StructGrid *) grid ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetSymmetric
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structmatrixsetsymmetric)( long int *matrix,
                                                 int      *symmetric,
                                                 int      *ierr      )
{
   *ierr = (int)
      ( HYPRE_StructMatrixSetSymmetric( (HYPRE_StructMatrix) *matrix,
                                        (int)                *symmetric ) );
}
