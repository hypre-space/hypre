/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <string.h>
#include <strings.h>
#include "parcsr_mv/parcsr_mv.h"
#include "solver/mli_solver_sgs.h"
#include "base/mli_defs.h"

/******************************************************************************
 * symmetric Gauss-Seidel relaxation scheme
 *****************************************************************************/

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_SGS::MLI_Solver_SGS() : MLI_Solver(MLI_SOLVER_SGS_ID)
{
   Amat             = NULL;
   zeroInitialGuess = 0;
   nsweeps          = 1;
   relax_weights    = new double[1];
   relax_weights[0] = 0.5;
   useCG_           = 0;
   useOffProcData_  = 1;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_SGS::~MLI_Solver_SGS()
{
   if ( relax_weights != NULL ) delete [] relax_weights;
   relax_weights = NULL;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_SGS::setup(MLI_Matrix *mat)
{
   Amat = mat;
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_SGS::solve(MLI_Vector *f_in, MLI_Vector *u_in)
{
   register int        i, j, is, i_start, i_end, jj, *tmp_j;
   int                 relax_error=0, index, num_procs, num_sends;
   int                 num_cols_offd, start, local_nrows;
   int                 *A_diag_i, *A_diag_j, *A_offd_i, *A_offd_j;
   register double     res;
   double              *u_data, *f_data, *v_buf_data, *tmp_data;
   double              zero = 0.0, relax_weight, dtemp;
   double              *A_diag_data, *A_offd_data, *Vext_data;
   double              *r_vec, *z_vec, *p_vec, *ap_vec, alpha, sigma, rho;
   hypre_ParCSRMatrix  *A;
   hypre_CSRMatrix     *A_diag, *A_offd;
   hypre_Vector        *u_local, *f_local;
   hypre_ParCSRCommPkg *comm_pkg;
   hypre_ParVector     *f, *u;
   MPI_Comm            comm;
   hypre_ParCSRCommHandle *comm_handle;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   A             = (hypre_ParCSRMatrix *) Amat->getMatrix();
   comm          = hypre_ParCSRMatrixComm(A);
   comm_pkg      = hypre_ParCSRMatrixCommPkg(A);
   A_diag        = hypre_ParCSRMatrixDiag(A);
   local_nrows   = hypre_CSRMatrixNumRows(A_diag);
   A_diag_i      = hypre_CSRMatrixI(A_diag);
   A_diag_j      = hypre_CSRMatrixJ(A_diag);
   A_diag_data   = hypre_CSRMatrixData(A_diag);
   A_offd        = hypre_ParCSRMatrixOffd(A);
   num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   A_offd_i      = hypre_CSRMatrixI(A_offd);
   A_offd_j      = hypre_CSRMatrixJ(A_offd);
   A_offd_data   = hypre_CSRMatrixData(A_offd);
   u             = (hypre_ParVector *) u_in->getVector();
   u_local       = hypre_ParVectorLocalVector(u);
   u_data        = hypre_VectorData(u_local);
   f             = (hypre_ParVector *) f_in->getVector();
   f_local       = hypre_ParVectorLocalVector(f);
   f_data        = hypre_VectorData(f_local);
   MPI_Comm_size(comm,&num_procs);  

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
    * outer CG
    *-----------------------------------------------------------------*/

   if ( useCG_ == 1 )
   {
      r_vec  = new double[local_nrows];
      p_vec  = new double[local_nrows];
      z_vec  = new double[local_nrows];
      ap_vec = new double[local_nrows];
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start;j<hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1);j++)
         {
            jj = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            v_buf_data[index++] = u_data[jj];
         }
      }
      comm_handle = hypre_ParCSRCommHandleCreate(1,comm_pkg,v_buf_data,
                          Vext_data);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
      for ( i = 0; i < local_nrows; i++ ) 
      {
         i_start  = A_diag_i[i];
         i_end    = A_diag_i[i+1];
         tmp_j    = &(A_diag_j[i_start]);
         tmp_data = &(A_diag_data[i_start]);
         res      = 0.0;
         for (j = i_start; j < i_end; j++)
            res += (*tmp_data++) * u_data[*tmp_j++];
         if ( num_procs > 1 )
         {
            i_start  = A_offd_i[i];
            i_end    = A_offd_i[i+1];
            tmp_j    = &(A_offd_j[i_start]);
            tmp_data = &(A_offd_data[i_start]);
            for (j = i_start; j < i_end; j++)
               res += (*tmp_data++) * Vext_data[*tmp_j++];
         }
         r_vec[i] = f_data[i] - res;
      }
      for ( i = 0; i < local_nrows; i++ ) z_vec[i] = 0.0;
      zeroInitialGuess = 1;
   }
   else 
   {
      z_vec = u_data;
      r_vec = f_data;
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
         if ( ! zeroInitialGuess && useOffProcData_ == 1 )
         {
            index = 0;
            for (i = 0; i < num_sends; i++)
            {
               start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j=start;j<hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1);
                    j++)
               {
                  jj = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
                  v_buf_data[index++] = z_vec[jj];
               }
            }
            comm_handle = hypre_ParCSRCommHandleCreate(1,comm_pkg,v_buf_data,
                                                       Vext_data);
            hypre_ParCSRCommHandleDestroy(comm_handle);
            comm_handle = NULL;
         }
      }

      /*-----------------------------------------------------------------
       * forward sweep
       *-----------------------------------------------------------------*/

      for (i = 0; i < local_nrows; i++)     /* interior points first */
      {
         /*-----------------------------------------------------------
          * If diagonal is nonzero, relax point i; otherwise, skip it.
          *-----------------------------------------------------------*/

         if ( A_diag_data[A_diag_i[i]] != zero)
         {
            res      = r_vec[i];
            i_start  = A_diag_i[i];
            i_end    = A_diag_i[i+1];
            tmp_j    = &(A_diag_j[i_start]);
            tmp_data = &(A_diag_data[i_start]);
            for (j = i_start; j < i_end; j++)
               res -= (*tmp_data++) * z_vec[*tmp_j++];
            if ( ! zeroInitialGuess && num_procs > 1 && useOffProcData_ == 1)
            {
               i_start  = A_offd_i[i];
               i_end    = A_offd_i[i+1];
               tmp_j    = &(A_offd_j[i_start]);
               tmp_data = &(A_offd_data[i_start]);
               for (j = i_start; j < i_end; j++)
                  res -= (*tmp_data++) * Vext_data[*tmp_j++];
            }
            z_vec[i] += relax_weight * res / A_diag_data[A_diag_i[i]];
         }
      }

      /*-----------------------------------------------------------------
       * backward sweep
       *-----------------------------------------------------------------*/

      for (i = local_nrows-1; i > -1; i--)  /* interior points first */
      {
         /*-----------------------------------------------------------
          * If diagonal is nonzero, relax point i; otherwise, skip it.
          *-----------------------------------------------------------*/

         if ( A_diag_data[A_diag_i[i]] != zero)
         {
            res      = r_vec[i];
            i_start  = A_diag_i[i];
            i_end    = A_diag_i[i+1];
            tmp_j    = &(A_diag_j[i_start]);
            tmp_data = &(A_diag_data[i_start]);
            for (j = i_start; j < i_end; j++)
               res -= (*tmp_data++) * z_vec[*tmp_j++];
            if ( ! zeroInitialGuess && num_procs > 1 && useOffProcData_ == 1)
            {
               i_start  = A_offd_i[i];
               i_end    = A_offd_i[i+1];
               tmp_j    = &(A_offd_j[i_start]);
               tmp_data = &(A_offd_data[i_start]);
               for (j = i_start; j < i_end; j++)
                  res -= (*tmp_data++) * Vext_data[*tmp_j++];
            }
            z_vec[i] += relax_weight * res / A_diag_data[A_diag_i[i]];
         }
      }
      zeroInitialGuess = 0;
   }

   /*-----------------------------------------------------------------
    * continue outer CG
    *-----------------------------------------------------------------*/

   if ( useCG_ == 1 )
   {
      dtemp = 0.0;
      for ( i = 0; i < local_nrows; i++ ) dtemp += (r_vec[i] * z_vec[i]);
      MPI_Allreduce(&dtemp, &rho, 1, MPI_DOUBLE, MPI_SUM, comm);
      for ( i = 0; i < local_nrows; i++ ) p_vec[i] = z_vec[i];
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start;j<hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1);j++)
         {
            jj = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            v_buf_data[index++] = p_vec[jj];
         }
      }
      comm_handle = hypre_ParCSRCommHandleCreate(1,comm_pkg,v_buf_data,
                                                 Vext_data);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
      for ( i = 0; i < local_nrows; i++ ) 
      {
         i_start   = A_diag_i[i];
         i_end     = A_diag_i[i+1];
         tmp_j    = &(A_diag_j[i_start]);
         tmp_data = &(A_diag_data[i_start]);
         res      = 0.0;
         for (j = i_start; j < i_end; j++)
            res += (*tmp_data++) * p_vec[*tmp_j++];
         if ( num_procs > 1 )
         {
            i_start   = A_offd_i[i];
            i_end     = A_offd_i[i+1];
            tmp_j    = &(A_offd_j[i_start]);
            tmp_data = &(A_offd_data[i_start]);
            for (j = i_start; j < i_end; j++)
               res += (*tmp_data++) * Vext_data[*tmp_j++];
         }
         ap_vec[i] = res;
      }
      dtemp = 0.0;
      for ( i = 0; i < local_nrows; i++ ) dtemp += (p_vec[i] * ap_vec[i]);
      MPI_Allreduce(&dtemp, &sigma, 1, MPI_DOUBLE, MPI_SUM, comm);
      alpha = rho /sigma;
      for ( i = 0; i < local_nrows; i++ ) u_data[i] += alpha * p_vec[i];
      delete [] r_vec;
      delete [] p_vec;
      delete [] ap_vec;
      delete [] z_vec;
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

/******************************************************************************
 * set SGS parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_SGS::setParams( char *param_string, int argc, char **argv )
{
   int    i;
   double *weights;
   char   param1[200];

   if ( !strcasecmp(param_string, "useCG") )
   {
      useCG_ = 1; 
      useOffProcData_  = 0;
      return 0;
   }
   else if ( !strcasecmp(param_string, "useOffProcData") )
   {
      useOffProcData_  = 1;
      return 0;
   }
   else if ( !strcasecmp(param_string, "numSweeps") )
   {
      sscanf(param_string, "%s %d", param1, &nsweeps);
      if ( nsweeps < 1 ) nsweeps = 1;
      
      return 0;
   }
   else if ( !strcasecmp(param_string, "relaxWeight") )
   {
      if ( argc != 2 && argc != 1 ) 
      {
         printf("MLI_Solver_SGS::setParams ERROR : needs 1 or 2 args.\n");
         return 1;
      }
      if ( argc >= 1 ) nsweeps = *(int*)   argv[0];
      if ( argc == 2 ) weights = (double*) argv[1];
      if ( nsweeps < 1 ) nsweeps = 1;
      if ( relax_weights != NULL ) delete [] relax_weights;
      relax_weights = NULL;
      if ( weights != NULL )
      {
         relax_weights = new double[nsweeps];
         for ( i = 0; i < nsweeps; i++ ) relax_weights[i] = weights[i];
      }
   }
   else if ( !strcasecmp(param_string, "zeroInitialGuess") )
   {
      zeroInitialGuess = 1;
      return 0;
   }
   else
   {   
      printf("MLI_Solver_SGS::setParams - parameter not recognized.\n");
      printf("                 Params = %s\n", param_string);
      return 1;
   }
   return 0;
}

/******************************************************************************
 * set SGS parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_SGS::setParams( int ntimes, double *weights )
{
   if ( ntimes <= 0 )
   {
      printf("MLI_Solver_SGS::setParams WARNING : nsweeps set to 1.\n");
      ntimes = 1;
   }
   nsweeps = ntimes;
   if ( relax_weights != NULL ) delete [] relax_weights;
   relax_weights = new double[ntimes];
   if ( weights == NULL )
   {
      printf("MLI_Solver_SGS::setParams - relax_weights set to 0.5.\n");
      for ( int i = 0; i < ntimes; i++ ) relax_weights[i] = 0.5;
   }
   else
   {
      for ( int j = 0; j < ntimes; j++ ) 
      {
         if (weights[j] >= 0. && weights[j] <= 2.) 
            relax_weights[j] = weights[j];
         else 
         {
            printf("MLI_Solver_SGS::setParams - some weights set to 0.5.\n");
            relax_weights[j] = 0.5;
         }
      }
   }
   return 0;
}

