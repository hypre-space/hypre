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
 * HYPRE_StructGrid interface
 *
 *****************************************************************************/
#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_NewStructGrid
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_newstructgrid)( int      *comm,
                                      int      *dim,
                                      long int *grid,
                                      int      *ierr )
{
   *ierr = (int) ( HYPRE_NewStructGrid( (MPI_Comm)           *comm,
                                        (int)                *dim,
                                        (HYPRE_StructGrid *) grid ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeStructGrid
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_freestructgrid)( long int *grid,
                                       int      *ierr )
{
   *ierr = (int) ( HYPRE_FreeStructGrid( (HYPRE_StructGrid) *grid ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructGridExtents
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_setstructgridextents)( long int *grid,
                                             int      *ilower,
                                             int      *iupper,
                                             int      *ierr )
{
   *ierr = (int) ( HYPRE_SetStructGridExtents( (HYPRE_StructGrid) *grid,
                                               (int *)            ilower,
                                               (int *)            iupper ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructGridPeriodicity
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_setstructgridperiodic)( long int *grid,
                                              int      *periodic,
                                              int      *ierr)
{
   *ierr = (int) ( HYPRE_SetStructGridPeriodic( (HYPRE_StructGrid) *grid,
                                                (int *)             periodic) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AssembleStructGrid
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_assemblestructgrid)( long int *grid,
                                           int      *ierr )
{
   *ierr = (int) ( HYPRE_AssembleStructGrid( (HYPRE_StructGrid) *grid) );
}
