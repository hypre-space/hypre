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
   Vector  *f;
   Vector  *u;

   int      num_unknowns;
   int      num_points;

   int     *iu;
   int     *ip;
   int     *iv;

   double  *xp;
   double  *yp;
   double  *zp;

} Problem;

/*--------------------------------------------------------------------------
 * Accessor functions for the Problem structure
 *--------------------------------------------------------------------------*/

#define ProblemNumVariables(problem)   ((problem) -> num_variables)

#define ProblemA(problem)              ((problem) -> A)
#define ProblemF(problem)              ((problem) -> f)
#define ProblemU(problem)              ((problem) -> u)

#define ProblemNumUnknowns(problem)    ((problem) -> num_unknowns)
#define ProblemNumPoints(problem)      ((problem) -> num_points)

#define ProblemIU(problem)             ((problem) -> iu)
#define ProblemIP(problem)             ((problem) -> ip)
#define ProblemIV(problem)             ((problem) -> iv)

#define ProblemXP(problem)             ((problem) -> xp)
#define ProblemYP(problem)             ((problem) -> yp)
#define ProblemZP(problem)             ((problem) -> zp)


#endif
