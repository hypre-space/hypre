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
 * HYPRE_StructGridCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_createstructgrid)( int      *comm,
                                         int      *dim,
                                         long int *grid,
                                         int      *ierr )
{
   *ierr = (int) ( HYPRE_StructGridCreate( (MPI_Comm)           *comm,
                                           (int)                *dim,
                                           (HYPRE_StructGrid *) grid ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGridDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_destroystructgrid)( long int *grid,
                                          int      *ierr )
{
   *ierr = (int) ( HYPRE_StructGridDestroy( (HYPRE_StructGrid) *grid ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGridSetExtents
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_setstructgridextents)( long int *grid,
                                             int      *ilower,
                                             int      *iupper,
                                             int      *ierr )
{
   *ierr = (int) ( HYPRE_StructGridSetExtents( (HYPRE_StructGrid) *grid,
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
   *ierr = (int) ( HYPRE_StructGridSetPeriodic( (HYPRE_StructGrid) *grid,
                                                (int *)             periodic) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGridAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_assemblestructgrid)( long int *grid,
                                           int      *ierr )
{
   *ierr = (int) ( HYPRE_StructGridAssemble( (HYPRE_StructGrid) *grid) );
}
