/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#define PARASAILS

#ifdef PARASAILS
#include "parcsr_mv.h"
#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_smoother.h"
#include "Matrix.h"
#include "ParaSails.h"

extern int  MLI_Smoother_Apply_ParaSails(void *smoother_obj, MLI_Vector *f, 
                                         MLI_Vector *u);
extern int  MLI_Smoother_Apply_ParaSailsTrans(void *smoother_obj,
                                              MLI_Vector *f, MLI_Vector *u);

/******************************************************************************
 * ParaSails relaxation scheme
 *****************************************************************************/

typedef struct MLI_Smoother_ParaSails_Struct
{
   MLI_Matrix *Amat;
   ParaSails  *ps;
   int        factorized;
} 
MLI_Smoother_ParaSails;

/*--------------------------------------------------------------------------
 * MLI_Smoother_Destroy_ParaSails
 *--------------------------------------------------------------------------*/

void MLI_Smoother_Destroy_ParaSails(void *smoother_obj)
{
   MLI_Smoother_ParaSails *ps_smoother;
   ParaSails              *ps;

   ps_smoother = (MLI_Smoother_ParaSails *) smoother_obj;
   if ( ps_smoother != NULL )
   {
      ps = ps_smoother->ps;
      if ( ps != NULL ) ParaSailsDestroy(ps);
      if ( ps_smoother->Amat != NULL ) MLI_Matrix_Destroy(ps_smoother->Amat);
      if ( ps_smoother != NULL ) hypre_TFree( ps_smoother );
   }
   return;
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_Setup_ParaSails
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Setup_ParaSails(void *smoother_obj, MLI_Matrix *Amat,
                                 int sym, double thresh, int num_levels, 
                                 double filter, int parasails_loadbal,
                                 int parasails_factorized, int trans)
{
   hypre_ParCSRMatrix     *A;
   int                    *partition, mypid, start_row, end_row;
   int                    row, row_length, *col_indices;
   double                 *col_values;
   Matrix                 *mat;
   ParaSails              *ps;
   MLI_Smoother           *generic_smoother;
   MLI_Smoother_ParaSails *ps_smoother;
   MPI_Comm               comm;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   comm = hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);  
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   start_row = partition[mypid];
   end_row   = partition[mypid+1] - 1;
   A         = Amat->matrix;

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

   ps_smoother = hypre_CTAlloc( MLI_Smoother_ParaSails, 1 );
   if ( ps_smoother == NULL ) { return 1; }
   ps = ParaSailsCreate(comm, start_row, end_row, parasails_factorized);
   ps->loadbal_beta = parasails_loadbal;
   ParaSailsSetupPattern(ps, mat, thresh, num_levels);
   ParaSailsStatsPattern(ps, mat);
   ParaSailsSetupValues(ps, mat, filter);
   ParaSailsStatsValues(ps, mat);
   ps_smoother->factorized = parasails_factorized;
   ps_smoother->ps = ps;
   ps_smoother->Amat = Amat;

   /*-----------------------------------------------------------------
    * clean up and return object and function
    *-----------------------------------------------------------------*/

   MatrixDestroy(mat);
   generic_smoother         = (MLI_Smoother *) smoother_obj;
   generic_smoother->object = (void *) ps_smoother;
   if (trans) generic_smoother->apply_func = MLI_Smoother_Apply_ParaSailsTrans;
   else       generic_smoother->apply_func = MLI_Smoother_Apply_ParaSails;
   generic_smoother->destroy_func = MLI_Smoother_Destroy_ParaSails;
   return 0;
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_Apply_ParaSails
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Apply_ParaSails(void *smoother_obj, MLI_Vector *f_in, 
                                 MLI_Vector *u_in)
{
   hypre_ParCSRMatrix     *A;
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
   hypre_ParVector        *u, *f;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   MPI_Comm_size(comm,&num_procs);  
   smoother      = (MLI_Smoother_ParaSails *) smoother_obj;
   A             = smoother->Amat->matrix;
   comm          = hypre_ParCSRMatrixComm(A);
   A_diag        = hypre_ParCSRMatrixDiag(A);
   n             = hypre_CSRMatrixNumRows(A_diag);
   u             = (hypre_ParVector *) u_in->vector;
   u_local       = hypre_ParVectorLocalVector(u);
   u_data        = hypre_VectorData(u_local);

   /*-----------------------------------------------------------------
    * create temporary vector
    *-----------------------------------------------------------------*/

   f           = (hypre_ParVector *) f_in->vector;
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

int MLI_Smoother_Apply_ParaSailsTrans(void *smoother_obj, MLI_Vector *f_in,
                                      MLI_Vector *u_in)
{
   hypre_ParCSRMatrix     *A;
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
   hypre_ParVector        *u, *f;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   MPI_Comm_size(comm,&num_procs);  
   smoother      = (MLI_Smoother_ParaSails *) smoother_obj;
   A             = smoother->Amat->matrix;
   comm          = hypre_ParCSRMatrixComm(A);
   A_diag        = hypre_ParCSRMatrixDiag(A);
   n             = hypre_CSRMatrixNumRows(A_diag);
   u             = (hypre_ParVector *) u_in->vector;
   u_local       = hypre_ParVectorLocalVector(u);
   u_data        = hypre_VectorData(u_local);

   /*-----------------------------------------------------------------
    * create temporary vector
    *-----------------------------------------------------------------*/

   f           = (hypre_ParVector *) f_in->vector;
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

