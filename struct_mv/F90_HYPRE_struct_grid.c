/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/



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
hypre_F90_IFACE(hypre_structgridcreate, HYPRE_STRUCTGRIDCREATE)( HYPRE_Int      *comm,
                                         HYPRE_Int      *dim,
                                         hypre_F90_Obj *grid,
                                         HYPRE_Int      *ierr )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructGridCreate( (MPI_Comm)           *comm,
                                           (HYPRE_Int)                *dim,
                                           (HYPRE_StructGrid *) grid ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGridDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgriddestroy, HYPRE_STRUCTGRIDDESTROY)( hypre_F90_Obj *grid,
                                          HYPRE_Int      *ierr )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructGridDestroy( (HYPRE_StructGrid) *grid ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGridSetExtents
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgridsetextents, HYPRE_STRUCTGRIDSETEXTENTS)( hypre_F90_Obj *grid,
                                             HYPRE_Int      *ilower,
                                             HYPRE_Int      *iupper,
                                             HYPRE_Int      *ierr )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructGridSetExtents( (HYPRE_StructGrid) *grid,
                                               (HYPRE_Int *)            ilower,
                                               (HYPRE_Int *)            iupper ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructGridPeriodicity
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgridsetperiodic, HYPRE_STRUCTGRIDSETPERIODIC)( hypre_F90_Obj *grid,
                                              HYPRE_Int      *periodic,
                                              HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_StructGridSetPeriodic( (HYPRE_StructGrid) *grid,
                                                (HYPRE_Int *)             periodic) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGridAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgridassemble, HYPRE_STRUCTGRIDASSEMBLE)( hypre_F90_Obj *grid,
                                           HYPRE_Int      *ierr )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructGridAssemble( (HYPRE_StructGrid) *grid) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGridSetNumGhost
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgridsetnumghost, HYPRE_STRUCTGRIDSETNUMGHOST)
                                         ( hypre_F90_Obj *grid,
                                           HYPRE_Int      *num_ghost,
                                           HYPRE_Int      *ierr )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructGridSetNumGhost( (HYPRE_StructGrid) *grid,
                                                (HYPRE_Int *)            num_ghost) );
}
