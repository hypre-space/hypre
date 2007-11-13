/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
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


/*****************************************************************************
 *	Wrapper code for SMP compiler directives.  Translates 
 *	hypre SMP directives into the appropriate Open MP,
 *	IBM, SGI, or pgcc (Red) SMP compiler directives.
 ****************************************************************************/

#ifndef HYPRE_SMP_PRIVATE
#define HYPRE_SMP_PRIVATE
#endif

#ifdef HYPRE_USING_OPENMP
#ifndef HYPRE_SMP_REDUCTION_OP
#pragma omp parallel for private(HYPRE_SMP_PRIVATE) schedule(static)
#endif
#ifdef HYPRE_SMP_REDUCTION_OP
#pragma omp parallel for private(HYPRE_SMP_PRIVATE) \
reduction(HYPRE_SMP_REDUCTION_OP: HYPRE_SMP_REDUCTION_VARS) \
schedule(static)
#endif
#endif

#ifdef HYPRE_USING_SGI_SMP
#pragma parallel
#pragma pfor
#pragma schedtype(gss)
#pragma chunksize(10)
#endif

#ifdef HYPRE_USING_IBM_SMP
#pragma parallel_loop
#pragma schedule (guided,10)
#endif

#ifdef HYPRE_USING_PGCC_SMP
#ifndef HYPRE_SMP_REDUCTION_OP
#pragma parallel local(HYPRE_SMP_PRIVATE) pfor
#endif
#ifdef HYPRE_SMP_REDUCTION_OP
#endif
#endif

#undef HYPRE_SMP_PRIVATE
#undef HYPRE_SMP_REDUCTION_OP
#undef HYPRE_SMP_REDUCTION_VARS
