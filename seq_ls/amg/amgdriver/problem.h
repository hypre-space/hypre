/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

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

   Matrix  *A;
   char     A_input[256];

   Vector  *f;
   int      f_flag;
   char     f_input[256];

   Vector  *u;
   int      u_flag;
   char     u_input[256];

   int      num_unknowns;
   int      num_points;

   int     *iu;
   int     *ip;
   int     *iv;
   int      iupv_flag;
   char     iupv_input[256];

   double  *xp;
   double  *yp;
   double  *zp;
   int      xyzp_flag;
   char     xyzp_input[256];

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

#define ProblemXP(problem)             ((problem) -> xp)
#define ProblemYP(problem)             ((problem) -> yp)
#define ProblemZP(problem)             ((problem) -> zp)
#define ProblemXYZPFlag(problem)       ((problem) -> xyzp_flag)
#define ProblemXYZPInput(problem)      ((problem) -> xyzp_input)


#endif
