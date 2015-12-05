/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.6 $
 ***********************************************************************EHEADER*/





#include "_hypre_parcsr_mv.h"

extern int MLI_Smoother_Apply_Schwarz(void *smoother_obj,hypre_ParCSRMatrix *A,
                                        hypre_ParVector *f,hypre_ParVector *u);

/******************************************************************************
 * Schwarz relaxation scheme 
 *****************************************************************************/

typedef struct MLI_Smoother_Schwarz_Struct
{
   hypre_ParCSRMatrix *Amat;
   ParaSails          *ps;
   int                factorized;
} MLI_Smoother_Schwarz;

/*--------------------------------------------------------------------------
 * MLI_Smoother_Create_Schwarz
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Create_Schwarz(void **smoother_obj) 
{
   MLI_Smoother_Schwarz *smoother;

   smoother = hypre_CTAlloc( MLI_Smoother_Schwarz, 1 );
   if ( smoother == NULL ) { (*smoother_obj) = NULL; return 1; }
   smoother->Amat = NULL;
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_Destroy_Schwarz
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Destroy_Schwarz(void *smoother_obj)
{
   MLI_Smoother_Schwarz *smoother;

   smoother = (MLI_Smoother_Schwarz *) smoother_obj;
   if ( smoother != NULL ) hypre_TFree( smoother );
   return 0;
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_Setup_Schwarz
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Setup_Schwarz(void *smoother_obj, 
                               int (**smoother_func)(void *smoother_obj, 
                                hypre_ParCSRMatrix *A,hypre_ParVector *f,
                                hypre_ParVector *u), hypre_ParCSRMatrix *A, 
{
   int                    *partition, mypid, start_row, end_row;
   int                    row, row_length, *col_indices;
   double                 *col_values;
   Matrix                 *mat;
   ParaSails              *ps;
   MLI_Smoother_ParaSails *smoother;
   MPI_Comm               comm;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   comm = hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);  
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   start_row = partition[mypid];
   end_row   = partition[mypid+1] - 1;

   /*-----------------------------------------------------------------
    * construct a ParaSails matrix
    *-----------------------------------------------------------------*/

   mat = MatrixCreate(comm, start_row, end_row);
   for (row = start_row; row <= end_row; row++)
   {
      hypre_ParCSRMatrixGetRow(A, row, &row_length, &col_indices, &col_values);
      MatrixSetRow(mat, row, row_length, col_indices, col_values);
      hypre_ParCSRMatrixRestoreRow(A,row,&row_length,&col_indices,&col_values);
   }
   MatrixComplete(mat);

   /*-----------------------------------------------------------------
    * construct a ParaSails smoother object
    *-----------------------------------------------------------------*/

   smoother = hypre_CTAlloc( MLI_Smoother_ParaSails, 1 );
   if ( smoother == NULL ) { (*smoother_obj) = NULL; return 1; }
   ps = ParaSailsCreate(comm, start_row, end_row, parasails_factorized);
   ps->loadbal_beta = parasails_loadbal;
   ParaSailsSetupPattern(ps, mat, thresh, num_levels);
   ParaSailsStatsPattern(ps, mat);
   ParaSailsSetupValues(ps, mat, filter);
   ParaSailsStatsValues(ps, mat);
   smoother->factorized = parasails_factorized;
   smoother->ps = ps;
   smoother->Amat = A;

   /*-----------------------------------------------------------------
    * clean up and return object and function
    *-----------------------------------------------------------------*/

   MatrixDestroy(mat);
   (*smoother_obj) = (void *) smoother;
   if ( trans ) (*smoother_func) = MLI_Smoother_Apply_ParaSailsTrans;
   else         (*smoother_func) = MLI_Smoother_Apply_ParaSails;
   return 0;
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_Apply_ParaSails
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Apply_ParaSails(void *smoother_obj, hypre_ParCSRMatrix *A,
                                 hypre_ParVector *f, hypre_ParVector    *u)
{
   hypre_CSRMatrix        *A_diag;
   hypre_ParVector        *Vtemp;
   hypre_Vector           *u_local, *Vtemp_local;
   double                 *u_data, *Vtemp_data;
   int                    i, n, relax_error = 0, global_size;
   int                    num_procs, *partition1, *partition2;
   int                    parasails_factorized;
   double                 *tmp_data;
   MPI_Comm               comm;
   MLI_Smoother_ParaSails *smoother;
   ParaSails              *ps;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   MPI_Comm_size(comm,&num_procs);  
   smoother      = (MLI_Smoother_ParaSails *) smoother_obj;
   A             = smoother->Amat;
   comm          = hypre_ParCSRMatrixComm(A);
   A_diag        = hypre_ParCSRMatrixDiag(A);
   n             = hypre_CSRMatrixNumRows(A_diag);
   u_local       = hypre_ParVectorLocalVector(u);
   u_data        = hypre_VectorData(u_local);

   /*-----------------------------------------------------------------
    * create temporary vector
    *-----------------------------------------------------------------*/

   global_size = hypre_ParVectorGlobalSize(f);
   partition1  = hypre_ParVectorPartitioning(f);
   partition2  = hypre_CTAlloc( int, num_procs+1 );
   for ( i = 0; i <= num_procs; i++ ) partition2[i] = partition1[i];
   Vtemp = hypre_ParVectorCreate(comm, global_size, partition2);
   Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   Vtemp_data  = hypre_VectorData(Vtemp_local);

   /*-----------------------------------------------------------------
    * perform smoothing
    *-----------------------------------------------------------------*/

   hypre_ParVectorCopy(f, Vtemp);
   hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, Vtemp);
   tmp_data = hypre_CTAlloc( double, n );

   parasails_factorized = smoother->factorized;

   if (!parasails_factorized)
   {
      MatrixMatvec(ps->M, Vtemp_data, tmp_data);
      for (i = 0; i < n; i++) u_data[i] += tmp_data[i];
   }
   else
   {
      MatrixMatvec(ps->M, Vtemp_data, tmp_data);
      MatrixMatvecTrans(ps->M, tmp_data, tmp_data);
      for (i = 0; i < n; i++) u_data[i] += tmp_data[i];
   }

   /*-----------------------------------------------------------------
    * clean up 
    *-----------------------------------------------------------------*/

   hypre_TFree( tmp_data );

   return(relax_error); 
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_Apply_ParaSailsTrans
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Apply_ParaSailsTrans(void *smoother_obj,hypre_ParCSRMatrix *A,
                                      hypre_ParVector *f,hypre_ParVector *u)
{
   hypre_CSRMatrix        *A_diag;
   hypre_ParVector        *Vtemp;
   hypre_Vector           *u_local, *Vtemp_local;
   double                 *u_data, *Vtemp_data;
   int                    i, n, relax_error = 0, global_size;
   int                    num_procs, *partition1, *partition2;
   int                    parasails_factorized;
   double                 *tmp_data;
   MPI_Comm               comm;
   MLI_Smoother_ParaSails *smoother;
   ParaSails              *ps;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   MPI_Comm_size(comm,&num_procs);  
   smoother      = (MLI_Smoother_ParaSails *) smoother_obj;
   A             = smoother->Amat;
   comm          = hypre_ParCSRMatrixComm(A);
   A_diag        = hypre_ParCSRMatrixDiag(A);
   n             = hypre_CSRMatrixNumRows(A_diag);
   u_local       = hypre_ParVectorLocalVector(u);
   u_data        = hypre_VectorData(u_local);

   /*-----------------------------------------------------------------
    * create temporary vector
    *-----------------------------------------------------------------*/

   global_size = hypre_ParVectorGlobalSize(f);
   partition1  = hypre_ParVectorPartitioning(f);
   partition2  = hypre_CTAlloc( int, num_procs+1 );
   for ( i = 0; i <= num_procs; i++ ) partition2[i] = partition1[i];
   Vtemp = hypre_ParVectorCreate(comm, global_size, partition2);
   Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   Vtemp_data  = hypre_VectorData(Vtemp_local);

   /*-----------------------------------------------------------------
    * perform smoothing
    *-----------------------------------------------------------------*/

   hypre_ParVectorCopy(f, Vtemp);
   hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, Vtemp);
   tmp_data = hypre_CTAlloc( double, n );

   parasails_factorized = smoother->factorized;

   if (!parasails_factorized)
   {
      MatrixMatvecTrans(ps->M, Vtemp_data, tmp_data);
      for (i = 0; i < n; i++) u_data[i] += tmp_data[i];
   }
   else
   {
      MatrixMatvec(ps->M, Vtemp_data, tmp_data);
      MatrixMatvecTrans(ps->M, tmp_data, tmp_data);
      for (i = 0; i < n; i++) u_data[i] += tmp_data[i];
   }

   /*-----------------------------------------------------------------
    * clean up 
    *-----------------------------------------------------------------*/

   hypre_TFree( tmp_data );

   return(relax_error); 
}
#endif

