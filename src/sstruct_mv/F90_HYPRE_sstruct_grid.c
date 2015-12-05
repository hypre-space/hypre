/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_SStructGrid interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructGridCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridcreate, HYPRE_SSTRUCTGRIDCREATE)
                                                          (int        *comm,
                                                           int        *ndim,
                                                           int        *nparts,
                                                           long int   *grid_ptr,
                                                           int        *ierr)
{
   *ierr = (int) (HYPRE_SStructGridCreate( (MPI_Comm)           *comm,
                                           (int)                *ndim,
                                           (int)                *nparts,
                                           (HYPRE_SStructGrid *) grid_ptr ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgriddestroy, HYPRE_SSTRUCTGRIDDESTROY)
                                                           (long int   *grid,
                                                            int        *ierr)
{
   *ierr = (int) (HYPRE_SStructGridDestroy( (HYPRE_SStructGrid) *grid ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridSetExtents
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridsetextents, HYPRE_SSTRUCTGRIDSETEXTENTS)
                                                          (long int   *grid,
                                                           int        *part,
                                                           int        *ilower,
                                                           int        *iupper,
                                                           int        *ierr)
{
   *ierr = (int) (HYPRE_SStructGridSetExtents( (HYPRE_SStructGrid) *grid,
                                               (int)               *part,
                                               (int *)              ilower,
                                               (int *)              iupper ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridSetVariables
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridsetvariables, HYPRE_SSTRUCTGRIDSETVARIABLES)
                                                          (long int   *grid,
                                                           int        *part,
                                                           int        *nvars,
                                                           long int   *vartypes,
                                                           int        *ierr)
{
   *ierr = (int) (HYPRE_SStructGridSetVariables( (HYPRE_SStructGrid)      *grid,
                                                 (int)                    *part,
                                                 (int)                    *nvars,
                                                 (HYPRE_SStructVariable *) vartypes ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridAddVariables
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridaddvariables, HYPRE_SSTRUCTGRIDADDVARIABLES)
                                                          (long int   *grid,
                                                           int        *part,
                                                           int        *index,
                                                           int        *nvars,
                                                           long int   *vartypes,
                                                           int        *ierr)
{
   *ierr = (int) (HYPRE_SStructGridAddVariables( (HYPRE_SStructGrid)      *grid,
                                                 (int)                    *part,
                                                 (int *)                   index,
                                                 (int)                    *nvars,
                                                 (HYPRE_SStructVariable *) vartypes ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridSetNeighborBox
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridsetneighborbox, HYPRE_SSTRUCTGRIDSETNEIGHBORBOX)
                                                          (long int   *grid,
                                                           int        *part,
                                                           int        *ilower,
                                                           int        *iupper,
                                                           int        *nbor_part,
                                                           int        *nbor_ilower,
                                                           int        *nbor_iupper,
                                                           int        *index_map,
                                                           int        *ierr)
{
   *ierr = (int) (HYPRE_SStructGridSetNeighborBox( (HYPRE_SStructGrid) *grid,
                                                   (int)               *part,
                                                   (int *)              ilower,
                                                   (int *)              iupper,
                                                   (int)               *nbor_part,
                                                   (int *)              nbor_ilower,
                                                   (int *)              nbor_iupper,
                                                   (int *)              index_map ) );
}

/*--------------------------------------------------------------------------
 * *** placeholder ***
 *  HYPRE_SStructGridAddUnstructuredPart
 *--------------------------------------------------------------------------*/

#if 0

void
hypre_F90_IFACE(hypre_sstructgridaddunstructure, HYPRE_SSTRUCTGRIDADDUNSTRUCTURE)
                                                          (long int   *grid,
                                                           int        *ilower,
                                                           int        *iupper,
                                                           int        *ierr)
{
   *ierr = (int) (HYPRE_SStructGridAddUnstructuredPart( (HYPRE_SStructGrid) *grid,
                                                        (int *)              ilower,
                                                        (int *)              iupper) );
}
#endif

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridassemble, HYPRE_SSTRUCTGRIDASSEMBLE)
                                                          (long int   *grid,
                                                           int        *ierr)
{
   *ierr = (int) (HYPRE_SStructGridAssemble( (HYPRE_SStructGrid) *grid ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridSetPeriodic
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridsetperiodic, HYPRE_SSTRUCTGRIDSETPERIODIC)
                                                          (long int   *grid,
                                                           int        *part,
                                                           int        *periodic,
                                                           int        *ierr)
{
   *ierr = (int) (HYPRE_SStructGridSetPeriodic( (HYPRE_SStructGrid) *grid,
                                                (int)               *part,
                                                (int *)              periodic ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridSetNumGhost
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridsetnumghost, HYPRE_SSTRUCTGRIDSETNUMGHOST)
                                                          (long int   *grid,
                                                           int        *num_ghost,
                                                           int        *ierr)
{
   *ierr = (int) (HYPRE_SStructGridSetNumGhost( (HYPRE_SStructGrid) *grid,
                                                (int *)              num_ghost));       
}
