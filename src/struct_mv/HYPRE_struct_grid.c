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

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGridCreate( MPI_Comm          comm,
                        HYPRE_Int         dim,
                        HYPRE_StructGrid *grid )
{
   hypre_StructGridCreate(comm, dim, grid);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGridDestroy( HYPRE_StructGrid grid )
{
   return ( hypre_StructGridDestroy(grid) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGridSetExtents( HYPRE_StructGrid  grid,
                            HYPRE_Int        *ilower,
                            HYPRE_Int        *iupper )
{
   hypre_Index  new_ilower;
   hypre_Index  new_iupper;

   HYPRE_Int    d;

   hypre_SetIndex(new_ilower, 0);
   hypre_SetIndex(new_iupper, 0);
   for (d = 0; d < hypre_StructGridNDim((hypre_StructGrid *) grid); d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }

   return ( hypre_StructGridSetExtents(grid, new_ilower, new_iupper) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGridSetPeriodic( HYPRE_StructGrid  grid,
                             HYPRE_Int        *periodic )
{
   hypre_Index  new_periodic;

   HYPRE_Int    d;

   hypre_SetIndex(new_periodic, 0);
   for (d = 0; d < hypre_StructGridNDim(grid); d++)
   {
      hypre_IndexD(new_periodic, d) = periodic[d];
   }

   return ( hypre_StructGridSetPeriodic(grid, new_periodic) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGridAssemble( HYPRE_StructGrid grid )
{
   return ( hypre_StructGridAssemble(grid) );
}

/*---------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGridSetNumGhost( HYPRE_StructGrid  grid,
                             HYPRE_Int        *num_ghost )
{
   return ( hypre_StructGridSetNumGhost(grid, num_ghost) );
}

/*---------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGridCoarsen(HYPRE_StructGrid  grid,
                        HYPRE_Int        *stride,
                        HYPRE_StructGrid *cgrid)
{
   hypre_Index origin;

   hypre_SetIndex(origin, 0);
   hypre_StructCoarsen(grid, origin, stride, 1, cgrid);
   
   return hypre_error_flag;
}

/*---------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructGridProjectBox(HYPRE_StructGrid  grid,
                           HYPRE_Int        *ilower,
                           HYPRE_Int        *iupper,
                           HYPRE_Int        *origin,
                           HYPRE_Int        *stride)
{
   hypre_Box *box;

   box = hypre_BoxCreate(hypre_StructGridNDim(grid));
   hypre_CopyIndex(ilower, hypre_BoxIMin(box));
   hypre_CopyIndex(iupper, hypre_BoxIMax(box));
   hypre_ProjectBox(box, origin, stride);
   hypre_CopyIndex(hypre_BoxIMin(box), ilower);
   hypre_CopyIndex(hypre_BoxIMax(box), iupper);
   hypre_BoxDestroy(box);
   
   return hypre_error_flag;
}

