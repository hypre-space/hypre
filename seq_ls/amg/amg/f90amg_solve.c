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
 * amg_Solve
 *--------------------------------------------------------------------------*/

void      amg_solve__(Solve_err_flag, u_data, f_data, n, tol, data)
int      *Solve_err_flag;
double   *u_data;
double   *f_data;
int      *n;
double   *tol;
int      *data;
{
   Vector   *u;
   Vector   *f;


   u = NewVector(u_data, *n);
   f = NewVector(f_data, *n);

   *Solve_err_flag =  amg_Solve(u, f, *tol, (void *) *data);
}

