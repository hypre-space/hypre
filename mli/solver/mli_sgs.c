/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <stdio.h>
#include "parcsr_mv.h"
#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_smoother.h"

extern int  MLI_Smoother_Apply_SymGaussSeidel(void *smoother_obj, 
                                              MLI_Vector *f, MLI_Vector *u);

/******************************************************************************
 * Symmetric Gauss-Seidel relaxation scheme
 *****************************************************************************/

typedef struct MLI_Smoother_SGS_Struct
{
   MLI_Matrix *Amat;
   int        nsweeps;
   double     *relax_weights;
} 
MLI_Smoother_SGS;

/*--------------------------------------------------------------------------
 * MLI_Smoother_Destroy_SymGaussSeidel
 *--------------------------------------------------------------------------*/

void MLI_Smoother_Destroy_SymGaussSeidel(void *smoother_obj)
{
   MLI_Smoother_SGS *sgs_smoother;

   sgs_smoother = (MLI_Smoother_SGS *) smoother_obj;
   if ( sgs_smoother != NULL ) 
   {
      if ( sgs_smoother->Amat != NULL ) 
         MLI_Matrix_Destroy( sgs_smoother->Amat );
      if ( sgs_smoother->relax_weights != NULL ) 
         hypre_TFree(sgs_smoother->relax_weights);
      hypre_TFree( sgs_smoother );
   }
   return;
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_Setup_SymGaussSeidel
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Setup_SymGaussSeidel(void *smoother_obj, MLI_Matrix *Amat, 
                                      int ntimes, double *relax_weights)
{
   int              i, nsweeps;
   MLI_Smoother_SGS *sgs_smoother;
   MLI_Smoother     *generic_smoother = (MLI_Smoother *) smoother_obj;

   sgs_smoother = hypre_CTAlloc(MLI_Smoother_SGS, 1); 
   if ( sgs_smoother == NULL ) { return 1; }
   if ( ntimes <= 0 ) sgs_smoother->nsweeps = 1;
   else               sgs_smoother->nsweeps = ntimes;
   if ( relax_weights != NULL )
   {
      nsweeps = sgs_smoother->nsweeps;
      sgs_smoother->relax_weights = hypre_CTAlloc(double, nsweeps);
      for (i = 0; i < nsweeps; i++) 
      {
         printf("MLI_Smoother_Setup_SymGaussSeidel : weight %d = %e ?\n", i,
                relax_weights[i]);
         sgs_smoother->relax_weights[i] = relax_weights[i];
      }
   }
   else sgs_smoother->relax_weights = NULL;
   sgs_smoother->Amat = Amat;

   generic_smoother->apply_func = MLI_Smoother_Apply_SymGaussSeidel;
   generic_smoother->destroy_func = MLI_Smoother_Destroy_SymGaussSeidel;
   generic_smoother->object = sgs_smoother;
   return 0;
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_Apply_SymGaussSeidel
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Apply_SymGaussSeidel(void *smoother_obj, MLI_Vector *f_in,
                                      MLI_Vector *u_in)
{
   hypre_ParCSRMatrix  *A;
   hypre_CSRMatrix     *A_diag, *A_offd;
   int                 *A_diag_i, *A_diag_j, *A_offd_i, *A_offd_j;
   double              *A_diag_data, *A_offd_data;
   hypre_Vector        *u_local;
   double              *u_data;
   hypre_Vector        *f_local;
   double              *f_data;
   int                 i, j, n, is, relax_error = 0, global_size;
   int                 num_procs, *partitioning1, *partitioning2, nsweeps;
   int                 ii, jj, index, num_sends, num_cols_offd;
   int                 start;
   double              zero = 0.0, *relax_weights, relax_weight, res;
   double              *v_buf_data;
   double              *Vext_data;
   hypre_ParCSRCommPkg *comm_pkg;
   MPI_Comm            comm;
   MLI_Smoother_SGS    *smoother;
   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParVector     *f, *u;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   MPI_Comm_size(comm,&num_procs);  
   smoother      = (MLI_Smoother_SGS *) smoother_obj;
   relax_weights = smoother->relax_weights;
   nsweeps       = smoother->nsweeps;
   A             = smoother->Amat->matrix;
   comm          = hypre_ParCSRMatrixComm(A);
   comm_pkg      = hypre_ParCSRMatrixCommPkg(A);
   A_diag        = hypre_ParCSRMatrixDiag(A);
   n             = hypre_CSRMatrixNumRows(A_diag);
   A_diag_i      = hypre_CSRMatrixI(A_diag);
   A_diag_j      = hypre_CSRMatrixJ(A_diag);
   A_diag_data   = hypre_CSRMatrixData(A_diag);
   A_offd        = hypre_ParCSRMatrixOffd(A);
   num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   A_offd_i      = hypre_CSRMatrixI(A_offd);
   A_offd_j      = hypre_CSRMatrixJ(A_offd);
   A_offd_data   = hypre_CSRMatrixData(A_offd);
   u             = u_in->vector;
   u_local       = hypre_ParVectorLocalVector(u);
   u_data        = hypre_VectorData(u_local);
   f             = f_in->vector;
   f_local       = hypre_ParVectorLocalVector(f);
   f_data        = hypre_VectorData(f_local);

   /*-----------------------------------------------------------------
    * create temporary vector
    *-----------------------------------------------------------------*/

   global_size   = hypre_ParVectorGlobalSize(f);
   partitioning1 = hypre_ParVectorPartitioning(f);
   partitioning2 = hypre_CTAlloc( int, num_procs+1 );
   for ( i = 0; i <= num_procs; i++ ) partitioning2[i] = partitioning1[i];

   /*-----------------------------------------------------------------
    * setting up for interprocessor communication
    *-----------------------------------------------------------------*/

   if (num_procs > 1)
   {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      v_buf_data = hypre_CTAlloc(double,
                   hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
      Vext_data = hypre_CTAlloc(double,num_cols_offd);

      if (num_cols_offd)
      {
         A_offd_j = hypre_CSRMatrixJ(A_offd);
         A_offd_data = hypre_CSRMatrixData(A_offd);
      }
   }

   /*-----------------------------------------------------------------
    * perform GS sweeps
    *-----------------------------------------------------------------*/
 
   for( is = 0; is < nsweeps; is++ )
   {
      if ( relax_weights != NULL ) relax_weight = relax_weights[is];
      else                         relax_weight = 1.0;

      /*-----------------------------------------------------------------
       * communicate data on processor boundaries
       *-----------------------------------------------------------------*/

      if (num_procs > 1)
      {
         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j=start;j<hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1);j++)
               v_buf_data[index++]
                      = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
         }
         comm_handle = hypre_ParCSRCommHandleCreate(1,comm_pkg,v_buf_data,
                                                    Vext_data);
         hypre_ParCSRCommHandleDestroy(comm_handle);
         comm_handle = NULL;
      }

      /*-----------------------------------------------------------------
       * forward sweep
       *-----------------------------------------------------------------*/

      for (i = 0; i < n; i++)     /* interior points first */
      {
         /*-----------------------------------------------------------
          * If diagonal is nonzero, relax point i; otherwise, skip it.
          *-----------------------------------------------------------*/

         if ( A_diag_data[A_diag_i[i]] != zero)
         {
            res = f_data[i];
            for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
            {
               ii = A_diag_j[jj];
               res -= A_diag_data[jj] * u_data[ii];
            }
            for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
            {
               ii = A_offd_j[jj];
               res -= A_offd_data[jj] * Vext_data[ii];
            }
            u_data[i] += relax_weight * res / A_diag_data[A_diag_i[i]];
         }
      }

      /*-----------------------------------------------------------------
       * backward sweep
       *-----------------------------------------------------------------*/

      for (i = n-1; i > -1; i--)  /* interior points first */
      {
         /*-----------------------------------------------------------
          * If diagonal is nonzero, relax point i; otherwise, skip it.
          *-----------------------------------------------------------*/

         if ( A_diag_data[A_diag_i[i]] != zero)
         {
            res = f_data[i];
            for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
            {
               ii = A_diag_j[jj];
               res -= A_diag_data[jj] * u_data[ii];
            }
            for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
            {
               ii = A_offd_j[jj];
               res -= A_offd_data[jj] * Vext_data[ii];
            }
            u_data[i] += relax_weight * res / A_diag_data[A_diag_i[i]];
         }
      }
   }

   /*-----------------------------------------------------------------
    * clean up and return
    *-----------------------------------------------------------------*/

   if (num_procs > 1)
   {
      hypre_TFree(Vext_data);
      hypre_TFree(v_buf_data);
   }
   return(relax_error); 
}

