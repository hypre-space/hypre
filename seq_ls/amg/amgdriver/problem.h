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





/******************************************************************************
 *
 * Header info for Problem data structures
 *
 *****************************************************************************/

#ifndef _PROBLEM_HEADER
#define _PROBLEM_HEADER


/*--------------------------------------------------------------------------
 * Problem
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      num_variables;

   hypre_Matrix  *A;
   char     A_input[256];

   hypre_Vector  *f;
   int      f_flag;
   char     f_input[256];

   hypre_Vector  *u;
   int      u_flag;
   char     u_input[256];

   int      num_unknowns;
   int      num_points;

   int     *iu;
   int     *ip;
   int     *iv;
   int      iupv_flag;
   char     iupv_input[256];

} Problem;

/*--------------------------------------------------------------------------
 * Accessor functions for the Problem structure
 *--------------------------------------------------------------------------*/

#define ProblemNumVariables(problem)   ((problem) -> num_variables)

#define ProblemA(problem)              ((problem) -> A)
#define ProblemAInput(problem)         ((problem) -> A_input)

#define ProblemF(problem)              ((problem) -> f)
#define ProblemFFlag(problem)          ((problem) -> f_flag)
#define ProblemFInput(problem)         ((problem) -> f_input)

#define ProblemU(problem)              ((problem) -> u)
#define ProblemUFlag(problem)          ((problem) -> u_flag)
#define ProblemUInput(problem)         ((problem) -> u_input)

#define ProblemNumUnknowns(problem)    ((problem) -> num_unknowns)
#define ProblemNumPoints(problem)      ((problem) -> num_points)

#define ProblemIU(problem)             ((problem) -> iu)
#define ProblemIP(problem)             ((problem) -> ip)
#define ProblemIV(problem)             ((problem) -> iv)
#define ProblemIUPVFlag(problem)       ((problem) -> iupv_flag)
#define ProblemIUPVInput(problem)      ((problem) -> iupv_input)

#endif
