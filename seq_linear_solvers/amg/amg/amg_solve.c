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

void         amg_Solve(u, f, tol, data)
Vector      *u;
Vector      *f;
double       tol;
void        *data;
{
   AMGData  *amg_data = data;


   CALL_SOLVE(u, f, tol, amg_data);
}

