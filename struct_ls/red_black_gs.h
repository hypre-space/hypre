/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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

