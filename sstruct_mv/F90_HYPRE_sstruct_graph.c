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
   (HYPRE_Int       *comm,
    hypre_F90_Obj *grid,
    hypre_F90_Obj *graph_ptr,
    HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGraphCreate(
                     (MPI_Comm)            *comm,
                     (HYPRE_SStructGrid)   *grid,
                     (HYPRE_SStructGraph *) graph_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphdestroy, HYPRE_SSTRUCTGRAPHDESTROY)
   (hypre_F90_Obj *graph,
    HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGraphDestroy(
                     (HYPRE_SStructGraph) *graph ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphSetDomainGrid
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphsetdomaingrid, HYPRE_SSTRUCTGRAPHSETDOMAINGRID)
   (hypre_F90_Obj *graph,
    hypre_F90_Obj *domain_grid,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGraphSetDomainGrid(
                     (HYPRE_SStructGraph) *graph,
                     (HYPRE_SStructGrid)  *domain_grid ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphSetStencil
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphsetstencil, HYPRE_SSTRUCTGRAPHSETSTENCIL)
   (hypre_F90_Obj *graph,
    HYPRE_Int      *part,
    HYPRE_Int      *var,
    hypre_F90_Obj *stencil,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGraphSetStencil(
                     (HYPRE_SStructGraph)   *graph,
                     (HYPRE_Int)                  *part,
                     (HYPRE_Int)                  *var,
                     (HYPRE_SStructStencil) *stencil ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphSetFEM
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphsetfem, HYPRE_SSTRUCTGRAPHSETFEM)
   (hypre_F90_Obj *graph,
    HYPRE_Int      *part,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGraphSetFEM(
                     (HYPRE_SStructGraph) *graph,
                     (HYPRE_Int)                *part ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphSetFEMSparsity
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphsetfemsparsity, HYPRE_SSTRUCTGRAPHSETFEMSPARSITY)
   (hypre_F90_Obj *graph,
    HYPRE_Int      *part,
    HYPRE_Int      *nsparse,
    HYPRE_Int      *sparsity,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGraphSetFEMSparsity(
                     (HYPRE_SStructGraph) *graph,
                     (HYPRE_Int)                *part,
                     (HYPRE_Int)                *nsparse,
                     (HYPRE_Int *)               sparsity ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphAddEntries-
 *   THIS IS FOR A NON-OVERLAPPING GRID GRAPH.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphaddentries, HYPRE_SSTRUCTGRAPHADDENTRIES)
   (hypre_F90_Obj *graph,
    HYPRE_Int      *part,
    HYPRE_Int      *index,
    HYPRE_Int      *var,
    HYPRE_Int      *to_part,
    HYPRE_Int      *to_index,
    HYPRE_Int      *to_var,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGraphAddEntries(
                     (HYPRE_SStructGraph)  *graph,
                     (HYPRE_Int)                 *part,
                     (HYPRE_Int *)                index,
                     (HYPRE_Int)                 *var,
                     (HYPRE_Int)                 *to_part,
                     (HYPRE_Int *)                to_index,
                     (HYPRE_Int)                 *to_var ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphassemble, HYPRE_SSTRUCTGRAPHASSEMBLE)
   (hypre_F90_Obj *graph,
    HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGraphAssemble(
                     (HYPRE_SStructGraph) *graph ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphSetObjectType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphsetobjecttype, HYPRE_SSTRUCTGRAPHSETOBJECTTYPE)
   (hypre_F90_Obj *graph,
    HYPRE_Int      *type,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGraphSetObjectType(
                     (HYPRE_SStructGraph) *graph,
                     (HYPRE_Int)                *type ) );
}
