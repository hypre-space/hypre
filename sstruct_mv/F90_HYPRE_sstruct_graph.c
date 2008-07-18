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
 * HYPRE_SStructGraph interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphcreate, HYPRE_SSTRUCTGRAPHCREATE)
                                                            (int       *comm,
                                                             long int  *grid,
                                                             long int  *graph_ptr,
                                                             int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGraphCreate( (MPI_Comm)           *comm,
                                           (HYPRE_SStructGrid)   *grid,
                                           (HYPRE_SStructGraph *) graph_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphdestroy, HYPRE_SSTRUCTGRAPHDESTROY)
                                                             (long int *graph,
                                                             int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGraphDestroy( (HYPRE_SStructGraph) *graph ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphSetStencil
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphsetstencil, HYPRE_SSTRUCTGRAPHSETSTENCIL)
                                                               (long int *graph,
                                                                int      *part,
                                                                int      *var,
                                                                long int *stencil,
                                                                int      *ierr)
{
   *ierr = (int) (HYPRE_SStructGraphSetStencil( (HYPRE_SStructGraph)   *graph,
                                                (int)                  *part,
                                                (int)                  *var,
                                                (HYPRE_SStructStencil) *stencil ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphAddEntries-
 *   THIS IS FOR A NON-OVERLAPPING GRID GRAPH.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphaddentries, HYPRE_SSTRUCTGRAPHADDENTRIES)
                                                                (long int *graph,
                                                                 int      *part,
                                                                 int      *index,
                                                                 int      *var,
                                                                 int      *to_part,
                                                                 int      *to_index,
                                                                 int      *to_var,
                                                                 int      *ierr)
{
   *ierr = (int) (HYPRE_SStructGraphAddEntries( (HYPRE_SStructGraph)  *graph,
                                                (int)                 *part,
                                                (int *)                index,
                                                (int)                 *var,
                                                (int)                 *to_part,
                                                (int *)                to_index,
                                                (int)                 *to_var ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphassemble, HYPRE_SSTRUCTGRAPHASSEMBLE)
                                                              (long int *graph,
                                                               int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGraphAssemble( (HYPRE_SStructGraph) *graph ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphSetObjectType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphsetobjecttype, HYPRE_SSTRUCTGRAPHSETOBJECTTYPE)
                                                                (long int *graph,
                                                                 int      *type,
                                                                 int      *ierr)
{
   *ierr = (int) (HYPRE_SStructGraphSetObjectType((HYPRE_SStructGraph) *graph,
                                                  (int)                *type));
}
