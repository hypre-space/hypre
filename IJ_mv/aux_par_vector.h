/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * Header info for Auxiliary Parallel Vector data structures
 *
 * Note: this vector currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_AUX_PAR_VECTOR_HEADER
#define hypre_AUX_PAR_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary Parallel Vector
 *--------------------------------------------------------------------------*/

typedef struct
{
   int	    max_off_proc_elmts; /* length of off processor stash for
					SetValues and AddToValues*/
   int	    current_num_elmts; /* current no. of elements stored in stash */
   int     *off_proc_i; /* contains column indices */
   double  *off_proc_data; /* contains corresponding data */
} hypre_AuxParVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_AuxParVectorMaxOffProcElmts(matrix)  ((matrix) -> max_off_proc_elmts)
#define hypre_AuxParVectorCurrentNumElmts(matrix)  ((matrix) -> current_num_elmts)
#define hypre_AuxParVectorOffProcI(matrix)  ((matrix) -> off_proc_i)
#define hypre_AuxParVectorOffProcData(matrix)  ((matrix) -> off_proc_data)

#endif
