/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * ParCSRMatrix Fortran interface to macros
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixGlobalNumRows
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixglobalnumrows, HYPRE_PARCSRMATRIXGLOBALNUMROWS)( long int *matrix,
                                                  int      *num_rows,
                                                  int      *ierr      )
{
   *num_rows = (int) ( hypre_ParCSRMatrixGlobalNumRows
                          ( (hypre_ParCSRMatrix *) *matrix ) );

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixRowStarts
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixrowstarts, HYPRE_PARCSRMATRIXROWSTARTS)( long int *matrix,
                                              long int *row_starts,
                                              int      *ierr      )
{
   *row_starts = (long int) ( hypre_ParCSRMatrixRowStarts
                                 ( (hypre_ParCSRMatrix *) *matrix ) );

   *ierr = 0;
}

