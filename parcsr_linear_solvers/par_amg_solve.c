/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * AMG solve routine
 *
 *****************************************************************************/

#include "headers.h"
#include "par_amg.h"

/*--------------------------------------------------------------------
 * hypre_ParAMGSolve
 *--------------------------------------------------------------------*/

int
hypre_ParAMGSolve( void               *amg_vdata,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector    *f,
                   hypre_ParVector    *u         )
{

   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(A);   

   hypre_ParAMGData   *amg_data = amg_vdata;

   /* Data Structure variables */

   int      amg_ioutdat;
   int     *num_coeffs;
   int     *num_variables;
   int      cycle_op_count;
   int      num_levels;
   int      num_unknowns;
   double   tol;
   char    *file_name;
   hypre_ParCSRMatrix **A_array;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;

   /*  Local variables  */

   FILE    *fp;

   int      j;
   int      Solve_err_flag;
   int      max_iter;
   int      cycle_count;
   int      total_coeffs;
   int      total_variables;
   int      num_procs, my_id;

   double   alpha = 1.0;
   double   beta = -1.0;
   double   cycle_cmplxty;
   double   operat_cmplxty;
   double   grid_cmplxty;
   double   conv_factor;
   double   resid_nrm;
   double   resid_nrm_init;
   double   relative_resid;
   double   rhs_norm;
   double   old_resid;

   hypre_ParVector  *Vtemp;

   MPI_Comm_size(comm, &num_procs);   
   MPI_Comm_rank(comm,&my_id);

   amg_ioutdat   = hypre_ParAMGDataIOutDat(amg_data);
   file_name     = hypre_ParAMGDataLogFileName(amg_data);
   num_unknowns  = hypre_ParAMGDataNumUnknowns(amg_data);
   num_levels    = hypre_ParAMGDataNumLevels(amg_data);
   A_array       = hypre_ParAMGDataAArray(amg_data);
   F_array       = hypre_ParAMGDataFArray(amg_data);
   U_array       = hypre_ParAMGDataUArray(amg_data);

   tol           = hypre_ParAMGDataTol(amg_data);
   max_iter      = hypre_ParAMGDataMaxIter(amg_data);

   num_coeffs = hypre_CTAlloc(int, num_levels);
   num_variables = hypre_CTAlloc(int, num_levels);
   num_coeffs[0]    = hypre_ParCSRMatrixNumNonzeros(A_array[0]);
   num_variables[0] = hypre_ParCSRMatrixGlobalNumRows(A_array[0]);
 
   A_array[0] = A;
   F_array[0] = f;
   U_array[0] = u;

   Vtemp = hypre_CreateParVector(hypre_ParCSRMatrixComm(A_array[0]),
                                 hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                 hypre_ParCSRMatrixRowStarts(A_array[0]));
   hypre_InitializeParVector(Vtemp);
   hypre_SetParVectorPartitioningOwner(Vtemp,0);
   hypre_ParAMGDataVtemp(amg_data) = Vtemp;

   for (j = 1; j < num_levels; j++)
   {
      num_coeffs[j]    = hypre_ParCSRMatrixNumNonzeros(A_array[j]);
      num_variables[j] = hypre_ParCSRMatrixGlobalNumRows(A_array[j]);
   }

   /*-----------------------------------------------------------------------
    *    Write the solver parameters
    *-----------------------------------------------------------------------*/


   if (my_id == 0 && amg_ioutdat > 1)
      hypre_WriteParAMGSolverParams(amg_data); 



   /*-----------------------------------------------------------------------
    *    Initialize the solver error flag and assorted bookkeeping variables
    *-----------------------------------------------------------------------*/

   Solve_err_flag = 0;

   total_coeffs = 0;
   total_variables = 0;
   cycle_count = 0;
   operat_cmplxty = 0;
   grid_cmplxty = 0;

   /*-----------------------------------------------------------------------
    *     open the log file and write some initial info
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && amg_ioutdat >= 1)
   { 
      fp = fopen(file_name, "a");

      fprintf(fp,"\n\nAMG SOLUTION INFO:\n");

   }

   /*-----------------------------------------------------------------------
    *    Compute initial fine-grid residual and print to logfile
    *-----------------------------------------------------------------------*/

   hypre_CopyParVector(F_array[0], Vtemp);
   hypre_ParMatvec(alpha, A_array[0], U_array[0], beta, Vtemp);
   resid_nrm = sqrt(hypre_ParInnerProd(Vtemp, Vtemp));

   resid_nrm_init = resid_nrm;
   rhs_norm = sqrt(hypre_ParInnerProd(f, f));
   relative_resid = 9999;
   if (rhs_norm)
   {
      relative_resid = resid_nrm_init / rhs_norm;
   }

   if (my_id ==0 && (amg_ioutdat > 1))
   {     
      fprintf(fp,"                                            relative\n");
      fprintf(fp,"               residual        factor       residual\n");
      fprintf(fp,"               --------        ------       --------\n");
      fprintf(fp,"    Initial    %e                 %e\n",resid_nrm_init,
              relative_resid);
   }

   /*-----------------------------------------------------------------------
    *    Main V-cycle loop
    *-----------------------------------------------------------------------*/
   
   while (relative_resid >= tol && cycle_count < max_iter 
          && Solve_err_flag == 0)
   {
      hypre_ParAMGDataCycleOpCount(amg_data) = 0;   
      /* Op count only needed for one cycle */

      Solve_err_flag = hypre_ParAMGCycle(amg_data, F_array, U_array); 

      old_resid = resid_nrm;

      /*---------------------------------------------------------------
       *    Compute  fine-grid residual and residual norm
       *----------------------------------------------------------------*/

      hypre_CopyParVector(F_array[0], Vtemp);
      hypre_ParMatvec(alpha, A_array[0], U_array[0], beta, Vtemp);
      resid_nrm = sqrt(hypre_ParInnerProd(Vtemp, Vtemp));

      conv_factor = resid_nrm / old_resid;
      relative_resid = 9999;
      if (rhs_norm)
      {
         relative_resid = resid_nrm / rhs_norm;
      }

      ++cycle_count;

      if (my_id == 0 && (amg_ioutdat > 1))
      { 
         fprintf(fp,"    Cycle %2d   %e    %f     %e \n", cycle_count,
                 resid_nrm, conv_factor, relative_resid);
      }
   }

   if (cycle_count == max_iter) Solve_err_flag = 1;

   /*-----------------------------------------------------------------------
    *    Compute closing statistics
    *-----------------------------------------------------------------------*/

   conv_factor = pow((resid_nrm/resid_nrm_init),(1.0/((double) cycle_count)));


   for (j=0;j<hypre_ParAMGDataNumLevels(amg_data);j++)
   {
      total_coeffs += num_coeffs[j];
      total_variables += num_variables[j];
   }

   cycle_op_count = hypre_ParAMGDataCycleOpCount(amg_data);

   if (num_variables[0])
      grid_cmplxty = ((double) total_variables) / ((double) num_variables[0]);
   if (num_coeffs[0])
   {
      operat_cmplxty = ((double) total_coeffs) / ((double) num_coeffs[0]);
      cycle_cmplxty = ((double) cycle_op_count) / ((double) num_coeffs[0]);
   }

   if (my_id == 0 && amg_ioutdat >= 1)
   {
      if (Solve_err_flag == 1)
      {
         fprintf(fp,"\n\n==============================================");
         fprintf(fp,"\n NOTE: Convergence tolerance was not achieved\n");
         fprintf(fp,"      within the allowed %d V-cycles\n",max_iter);
         fprintf(fp,"==============================================");
      }
      fprintf(fp,"\n\n Average Convergence Factor = %f",conv_factor);
      fprintf(fp,"\n\n     Complexity:    grid = %f\n",grid_cmplxty);
      fprintf(fp,"                operator = %f\n",operat_cmplxty);
      fprintf(fp,"                   cycle = %f\n\n",cycle_cmplxty);
   }

   /*----------------------------------------------------------
    * Close the output file (if open)
    *----------------------------------------------------------*/

   if (my_id == 0 && amg_ioutdat >= 1)
   { 
      fclose(fp); 
   }

   hypre_TFree(num_coeffs);
   hypre_TFree(num_variables);

   return(Solve_err_flag);
}

