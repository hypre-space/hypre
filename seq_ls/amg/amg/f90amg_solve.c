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
 * AMG solve routine (Fortran 90 interface)
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * HYPRE_AMGSolve
 *--------------------------------------------------------------------------*/

void      hypre_NAME_C_FOR_FORTRAN(amg_solve)(Solve_err_flag, u_data, f_data,
					n, tol, data)
int      *Solve_err_flag;
double   *u_data;
double   *f_data;
int      *n;
double   *tol;
int      *data;
{
   hypre_Vector   *u;
   hypre_Vector   *f;


   u = hypre_NewVector(u_data, *n);
   f = hypre_NewVector(f_data, *n);

   *Solve_err_flag =  HYPRE_AMGSolve(u, f, *tol, (void *) *data);
}

