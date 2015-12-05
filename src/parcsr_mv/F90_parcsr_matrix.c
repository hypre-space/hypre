/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
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
hypre_F90_IFACE(hypre_parcsrmatrixglobalnumrows, HYPRE_PARCSRMATRIXGLOBALNUMROWS)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *num_rows,
     hypre_F90_Int *ierr      )
{
   *num_rows = (hypre_F90_Int)
      ( hypre_ParCSRMatrixGlobalNumRows(
           (hypre_ParCSRMatrix *) *matrix ) );

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixRowStarts
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixrowstarts, HYPRE_PARCSRMATRIXROWSTARTS)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Obj *row_starts,
     hypre_F90_Int *ierr      )
{
   *row_starts = (hypre_F90_Obj)
      ( hypre_ParCSRMatrixRowStarts(
           (hypre_ParCSRMatrix *) *matrix ) );

   *ierr = 0;
}

