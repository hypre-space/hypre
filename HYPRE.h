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
 * Header file for HYPRE library
 *
 *****************************************************************************/

#ifndef HYPRE_HEADER
#define HYPRE_HEADER


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * Constants
 *--------------------------------------------------------------------------*/

#define HYPRE_UNITIALIZED -999

#define HYPRE_PETSC_MAT_PARILUT_SOLVER 222
#define HYPRE_PARILUT                  333

#define HYPRE_STRUCT  1111
#define HYPRE_SSTRUCT 3333
#define HYPRE_PARCSR  5555

#define HYPRE_ISIS    9911
#define HYPRE_PETSC   9933

#define HYPRE_PFMG    10
#define HYPRE_SMG     11

#endif
