/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




/*--------------------------------------------------------------------------
 * hypre_RedBlackGSData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;

   double                  tol;                /* not yet used */
   int                     max_iter;
   int                     rel_change;         /* not yet used */
   int                     zero_guess;
   int                     rb_start;

   hypre_StructMatrix     *A;
   hypre_StructVector     *b;
   hypre_StructVector     *x;

   int                     diag_rank;

   hypre_ComputePkg       *compute_pkg;

   /* log info (always logged) */
   int                     num_iterations;
   int                     time_index;
   int                     flops;

} hypre_RedBlackGSData;

