/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_LSI_SSOR interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities/utilities.h"
#include "HYPRE.h"
#include "IJ_matrix_vector/HYPRE_IJ_mv.h"
#include "parcsr_matrix_vector/HYPRE_parcsr_mv.h"
#include "parcsr_matrix_vector/parcsr_matrix_vector.h"
#include "parcsr_linear_solvers/HYPRE_parcsr_ls.h"
#include "parcsr_linear_solvers/parcsr_linear_solvers.h"
#include "HYPRE_MHMatrix.h"

typedef struct HYPRE_LSI_SSOR_Struct
{
   double    omega;
   int       outputLevel;
}
HYPRE_LSI_SSOR;

/*--------------------------------------------------------------------------
 * HYPRE_LSI_SSORCreate - Return a SSOR preconditioner object "solver".  
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_SSORCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_LSI_SSOR *ssor_ptr;
   
   ssor_ptr = (HYPRE_LSI_SSOR *) malloc(sizeof(HYPRE_LSI_SSOR));
   if (ssor_ptr == NULL) return 1;

   ssor_ptr->omega = 1.0;
   *solver = (HYPRE_Solver) ssor_ptr;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_SSORDestroy - Destroy a SSOR object.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_SSORDestroy( HYPRE_Solver solver )
{
   HYPRE_LSI_SSOR *ssor_ptr;

   ssor_ptr = (HYPRE_LSI_SSOR *) solver;
   free(ssor_ptr);

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_LSI_SSORSolve - Destroy a SSOR object.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_SSORSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,   HYPRE_ParVector x )
{
   int             i, index, ierr, num_procs, my_id, num_sends;
   int             start, m, ii, j, jj;
   double          zero = 0.0, res, omega;
   HYPRE_LSI_SSOR  *ssor_ptr;
   double          *v_buf_data, *Vext_data;
   MPI_Comm        comm = hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *) A);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag((hypre_ParCSRMatrix *) A);
   double          *A_diag_data  = hypre_CSRMatrixData(A_diag);
   int             *A_diag_i     = hypre_CSRMatrixI(A_diag);
   int             *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   hypre_CSRMatrix *A_offd       = hypre_ParCSRMatrixOffd((hypre_ParCSRMatrix *) A);
   int             *A_offd_i     = hypre_CSRMatrixI(A_offd);
   double          *A_offd_data  = hypre_CSRMatrixData(A_offd);
   int             *A_offd_j     = hypre_CSRMatrixJ(A_offd);
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg((hypre_ParCSRMatrix*)A);
   hypre_ParCSRCommHandle *comm_handle;
   int             n_global= hypre_ParCSRMatrixGlobalNumRows((hypre_ParCSRMatrix*)A);
   int             n       = hypre_CSRMatrixNumRows(A_diag);
   int             num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   hypre_Vector    *u_local = hypre_ParVectorLocalVector((hypre_ParVector *)x);
   double          *u_data  = hypre_VectorData(u_local);

   hypre_Vector    *f_local = hypre_ParVectorLocalVector((hypre_ParVector *)b);
   double          *f_data  = hypre_VectorData(f_local);

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   ssor_ptr = (HYPRE_LSI_SSOR *) solver;
   omega = ssor_ptr->omega;

   if ( num_procs > 1 )
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
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1); j++)
            v_buf_data[index++]
                        = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
      comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data,
                    Vext_data);

      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

   for (i = 0; i < n; i++) u_data[i] = f_data[i];
   for (m = 0; m < 1; m++)
   {
      for (i = 0; i < n; i++)
      {
         if ( A_diag_data[A_diag_i[i]] != zero)
         {
            res = 0.0;
            for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
            {
               ii = A_diag_j[jj];
               if ( ii < i ) res += (A_diag_data[jj] * u_data[ii]);
            }
            u_data[i] = (u_data[i] - omega * res ) / A_diag_data[A_diag_i[i]];
         }
      }
      for (i = n-1; i > -1; i--)
      {
         if ( A_diag_data[A_diag_i[i]] != zero)
         {
            res = 0.0;
            for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
            {
               ii = A_diag_j[jj];
               if ( ii > i ) res += (A_diag_data[jj] * u_data[ii]);
            }
            u_data[i] -= ( omega * res / A_diag_data[A_diag_i[i]]);
         }
      }
   }
   return 0;
}

