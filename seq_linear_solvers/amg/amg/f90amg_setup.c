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
 * AMG setup routine (Fortran 90 interface)
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * amg_Setup
 *--------------------------------------------------------------------------*/

void     amg_Setup_(a_data, ia, ja, n, data)
double  *a_data;
int     *ia;
int     *ja;
int     *n;
int     *data;
{
   Matrix  *A;


   tfree(AMGDataA((AMGData *) data));
   A = NewMatrix(a_data, ia, ja, *n);

   amg_Setup(A, (void *) *data);
}


