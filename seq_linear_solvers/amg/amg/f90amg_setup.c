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

void     NAME_C_FOR_FORTRAN(amg_setup)(Setup_err_flag, a_data, ia, ja, n, data)
int     *Setup_err_flag;
double  *a_data;
int     *ia;
int     *ja;
int     *n;
int     *data;
{
   Matrix  *A;


   tfree(AMGDataA((AMGData *) *data));
   A = NewMatrix(a_data, ia, ja, *n);

   *Setup_err_flag = amg_Setup(A, (void *) *data);
}



