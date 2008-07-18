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
