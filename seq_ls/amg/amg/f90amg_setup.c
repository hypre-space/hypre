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
 * HYPRE_AMGSetup
 *--------------------------------------------------------------------------*/

void     hypre_NAME_C_FOR_FORTRAN(amg_setup)(Setup_err_flag, a_data, ia, ja, n, data)
int     *Setup_err_flag;
double  *a_data;
int     *ia;
int     *ja;
int     *n;
int     *data;
{
   hypre_Matrix  *A;


   hypre_TFree(hypre_AMGDataA((hypre_AMGData *) *data));
   A = hypre_NewMatrix(a_data, ia, ja, *n);

   *Setup_err_flag = HYPRE_AMGSetup(A, (void *) *data);
}



