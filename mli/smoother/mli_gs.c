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

extern int  MLI_Smoother_Apply_GaussSeidel(void *smoother_obj, MLI_Vector *f, 
                                           MLI_Vector *u);

/******************************************************************************
 * Gauss-Seidel relaxation scheme
 *****************************************************************************/

typedef struct MLI_Smoother_GS_Struct
{
   MLI_Matrix *Amat;
   int        nsweeps;
   double     *relax_weights;
} 
MLI_Smoother_GS;

/*--------------------------------------------------------------------------
 * MLI_Smoother_Destroy_GaussSeidel
 *--------------------------------------------------------------------------*/

void MLI_Smoother_Destroy_GaussSeidel(void *smoother_obj)
{
   MLI_Smoother_GS *gs_smoother;

   gs_smoother = (MLI_Smoother_GS *) smoother_obj;
   if ( gs_smoother != NULL )
   {
      if ( gs_smoother->Amat != NULL ) 
         MLI_Matrix_Destroy( gs_smoother->Amat );
      if ( gs_smoother->relax_weights != NULL ) 
         hypre_TFree( gs_smoother->relax_weights );
      hypre_TFree( gs_smoother );
   }
   return;
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_Setup_GaussSeidel
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Setup_GaussSeidel(void *smoother_obj, MLI_Matrix *Amat, 
                                   int ntimes, double *relax_weights)
{
   int                i;
   MLI_Smoother       *generic_smoother;
   MLI_Smoother_GS    *gs_smoother ;

   generic_smoother = (MLI_Smoother *) smoother_obj;
   gs_smoother = hypre_CTAlloc(MLI_Smoother_GS, 1); 
   if ( gs_smoother == NULL ) { return 1; }
   if ( ntimes <= 0 ) gs_smoother->nsweeps = 1;
   else               gs_smoother->nsweeps = ntimes;
   if ( relax_weights != NULL )
   {
      gs_smoother->relax_weights = hypre_CTAlloc(double, gs_smoother->nsweeps); 
      for ( i = 0; i < gs_smoother->nsweeps; i++ )
      {
         if ( relax_weights[i] < 0 || relax_weights[i] > 2. )
            printf("MLI_Smoother_Setup_GaussSeidel : weight %d = %e ?\n", i,
                   relax_weights[i]);
         gs_smoother->relax_weights[i] = relax_weights[i];
      }
   }
   else gs_smoother->relax_weights = NULL;
   gs_smoother->Amat = Amat;

   generic_smoother->apply_func = MLI_Smoother_Apply_GaussSeidel;
   generic_smoother->destroy_func = MLI_Smoother_Destroy_GaussSeidel;
   generic_smoother->object = (void *) gs_smoother;
   return 0;
}


/*--------------------------------------------------------------------------
 * MLI_Smoother_Apply_GaussSeidel
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Apply_GaussSeidel(void *smoother_obj, MLI_Vector *f_in, 
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
   hypre_ParVector     *Vtemp;
   hypre_Vector        *Vtemp_local;
   double              *Vtemp_data;
   int                 i, j, n, is, relax_error = 0, global_size, *partitioning1;
   int                 ii, jj, num_procs, num_threads, *partitioning2, nsweeps;
   int                 num_sends, num_cols_offd, index, size, ns, ne, rest;
   int                 start;
   double              zero = 0.0, *relax_weights, relax_weight, res;
   double              *v_buf_data;
   double              *Vext_data, *tmp_data;
   hypre_ParCSRCommPkg *comm_pkg;
   MPI_Comm            comm;
   MLI_Smoother_GS     *gs_smoother;
   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParVector     *f, *u;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   MPI_Comm_size(comm,&num_procs);  
   num_threads   = hypre_NumThreads();
   gs_smoother   = (MLI_Smoother_GS *) smoother_obj;
   relax_weights = gs_smoother->relax_weights;
   nsweeps       = gs_smoother->nsweeps;
   A             = gs_smoother->Amat->matrix;
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
   Vtemp = hypre_ParVectorCreate(comm, global_size, partitioning2);
   Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   Vtemp_data  = hypre_VectorData(Vtemp_local);

   /*-----------------------------------------------------------------
    * Copy f into temporary vector.
    *-----------------------------------------------------------------*/
        
   hypre_ParVectorCopy(f,Vtemp); 
 
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
   if (num_threads > 1) tmp_data = hypre_CTAlloc(double,n);

   /*-----------------------------------------------------------------
    * perform GS sweeps
    *-----------------------------------------------------------------*/
 
   for( is = 0; is < nsweeps; is++ )
   {
      if ( relax_weights != NULL ) relax_weight = relax_weights[is];
      else                         relax_weight = 1.0;

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

      if (num_threads > 1)
      {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
         for (i = 0; i < n; i++) tmp_data[i] = u_data[i];

#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
         for (j = 0; j < num_threads; j++)
         {
            size = n/num_threads;
            rest = n - size*num_threads;
            if (j < rest)
            {
               ns = j*size+j;
               ne = (j+1)*size+j+1;
            }
            else
            {
               ns = j*size+rest;
               ne = (j+1)*size+rest;
            }
            for (i = ns; i < ne; i++)   /* interior points first */
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
                     if (ii >= ns && ii < ne)
                        res -= A_diag_data[jj] * u_data[ii];
                     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] += relax_weight * (res / A_diag_data[A_diag_i[i]]);
               }
            }
         }
      }
      else
      {
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
               u_data[i] += relax_weight * (res / A_diag_data[A_diag_i[i]]);
            }
         }
      }
   }

   /*-----------------------------------------------------------------
    * clean up and return
    *-----------------------------------------------------------------*/

   hypre_ParVectorDestroy( Vtemp ); 
   if (num_procs > 1)
   {
      hypre_TFree(Vext_data);
      hypre_TFree(v_buf_data);
   }
   if (num_threads > 1) hypre_TFree(tmp_data);
   return(relax_error); 
}

