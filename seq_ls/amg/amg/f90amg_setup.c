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
 * AMG setup routine (Fortran 90 interface)
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * HYPRE_AMGSetup
 *--------------------------------------------------------------------------*/

void     hypre_NAME_C_FOR_FORTRAN(amg_setup)(Setup_err_flag, a_data, ia, ja, n, data)
HYPRE_Int     *Setup_err_flag;
double  *a_data;
HYPRE_Int     *ia;
HYPRE_Int     *ja;
HYPRE_Int     *n;
HYPRE_Int     *data;
{
   hypre_Matrix  *A;


   hypre_TFree(hypre_AMGDataA((hypre_AMGData *) *data));
   A = hypre_NewMatrix(a_data, ia, ja, *n);

   *Setup_err_flag = HYPRE_AMGSetup(A, (void *) *data);
}



