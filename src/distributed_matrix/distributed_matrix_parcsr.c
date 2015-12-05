/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Member functions for hypre_DistributedMatrix class for par_csr storage scheme.
 *
 *****************************************************************************/

#include "./distributed_matrix.h"

#include "HYPRE_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixDestroyParCSR
 *   Internal routine for freeing a matrix stored in Parcsr form.
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_DistributedMatrixDestroyParCSR( hypre_DistributedMatrix *distributed_matrix )
{

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixInitializeParCSR
 *--------------------------------------------------------------------------*/

  /* matrix must be set before calling this function*/

HYPRE_Int 
hypre_DistributedMatrixInitializeParCSR(hypre_DistributedMatrix *matrix)
{
   
   return 0;
}

/*--------------------------------------------------------------------------
 * Optional routines that depend on underlying storage type
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixPrintParCSR
 *   Internal routine for printing a matrix stored in Parcsr form.
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_DistributedMatrixPrintParCSR( hypre_DistributedMatrix *matrix )
{
   HYPRE_Int  ierr=0;
   HYPRE_ParCSRMatrix Parcsr_matrix = (HYPRE_ParCSRMatrix) hypre_DistributedMatrixLocalStorage(matrix);

   HYPRE_ParCSRMatrixPrint( Parcsr_matrix, "STDOUT" );
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetLocalRangeParCSR
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_DistributedMatrixGetLocalRangeParCSR( hypre_DistributedMatrix *matrix,
                             HYPRE_Int *row_start,
                             HYPRE_Int *row_end,
                             HYPRE_Int *col_start,
                             HYPRE_Int *col_end )
{
   HYPRE_Int ierr=0;
   HYPRE_ParCSRMatrix Parcsr_matrix = (HYPRE_ParCSRMatrix) hypre_DistributedMatrixLocalStorage(matrix);

   if (!Parcsr_matrix) return(-1);


   ierr = HYPRE_ParCSRMatrixGetLocalRange( Parcsr_matrix, row_start, row_end, 
					col_start, col_end );

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetRowParCSR
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_DistributedMatrixGetRowParCSR( hypre_DistributedMatrix *matrix,
                             HYPRE_Int row,
                             HYPRE_Int *size,
                             HYPRE_Int **col_ind,
                             double **values )
{
   HYPRE_Int ierr = 0;
   HYPRE_ParCSRMatrix Parcsr_matrix = (HYPRE_ParCSRMatrix) hypre_DistributedMatrixLocalStorage(matrix);

   if (!Parcsr_matrix) return(-1);

   ierr = HYPRE_ParCSRMatrixGetRow( Parcsr_matrix, row, size, col_ind, values);

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixRestoreRowParCSR
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_DistributedMatrixRestoreRowParCSR( hypre_DistributedMatrix *matrix,
                             HYPRE_Int row,
                             HYPRE_Int *size,
                             HYPRE_Int **col_ind,
                             double **values )
{
   HYPRE_Int ierr;
   HYPRE_ParCSRMatrix Parcsr_matrix = (HYPRE_ParCSRMatrix) hypre_DistributedMatrixLocalStorage(matrix);

   if (Parcsr_matrix == NULL) return(-1);

   ierr = HYPRE_ParCSRMatrixRestoreRow( Parcsr_matrix, row, size, col_ind, values); 

   return(ierr);
}
