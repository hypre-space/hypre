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
 * AMG solve routine
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * amg_Solve
 *--------------------------------------------------------------------------*/

int         amg_Solve(u, f, tol, data)
Vector      *u;
Vector      *f;
double       tol;
void        *data;
{
   int Solve_err_flag;

   AMGData  *amg_data = data;

   WriteSolverParams(tol, amg_data);

   CALL_SOLVE(Solve_err_flag, u, f, tol, amg_data);

   return(Solve_err_flag);
}

