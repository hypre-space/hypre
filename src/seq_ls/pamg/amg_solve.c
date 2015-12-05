/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * AMG solve routine
 *
 *****************************************************************************/

#include "headers.h"
#include "amg.h"

/*--------------------------------------------------------------------
 * hypre_AMGSolve
 *--------------------------------------------------------------------*/


HYPRE_Int
hypre_AMGSolve( void            *amg_vdata,
                hypre_CSRMatrix *A,
                hypre_Vector    *f,
                hypre_Vector    *u         )
{
   hypre_AMGData   *amg_data = amg_vdata;
   
   /* Data Structure variables */

   HYPRE_Int      amg_ioutdat;
   HYPRE_Int     *num_coeffs;
   HYPRE_Int     *num_variables;
   /* HYPRE_Int      cycle_op_count; */
   HYPRE_Int      num_levels;
   /* HYPRE_Int      num_functions; */
   double   tol;
   /* char    *file_name; */
   hypre_CSRMatrix **A_array;
   hypre_Vector    **F_array;
   hypre_Vector    **U_array;

   /*  Local variables  */

   HYPRE_Int      j;
   HYPRE_Int      Solve_err_flag;
   HYPRE_Int      max_iter;
   HYPRE_Int      cycle_count;
   HYPRE_Int      total_coeffs;
   HYPRE_Int      total_variables;

   double   alpha = 1.0;
   double   beta = -1.0;
   /* double   cycle_cmplxty; */
   double   operat_cmplxty;
   double   grid_cmplxty;
   double   conv_factor;
   double   resid_nrm;
   double   resid_nrm_init;
   double   relative_resid;
   double   rhs_norm;
   double   old_resid;

   hypre_Vector  *Vtemp;

   amg_ioutdat   = hypre_AMGDataIOutDat(amg_data);
   /* file_name     = hypre_AMGDataLogFileName(amg_data); */
   /* num_functions  = hypre_AMGDataNumFunctions(amg_data); */
   num_levels    = hypre_AMGDataNumLevels(amg_data);
   A_array       = hypre_AMGDataAArray(amg_data);
   F_array       = hypre_AMGDataFArray(amg_data);
   U_array       = hypre_AMGDataUArray(amg_data);

   tol           = hypre_AMGDataTol(amg_data);
   max_iter      = hypre_AMGDataMaxIter(amg_data);

   num_coeffs = hypre_CTAlloc(HYPRE_Int, num_levels);
   num_variables = hypre_CTAlloc(HYPRE_Int, num_levels);
   num_coeffs[0]    = hypre_CSRMatrixNumNonzeros(A_array[0]);
   num_variables[0] = hypre_CSRMatrixNumRows(A_array[0]);
 
   A_array[0] = A;
   F_array[0] = f;
   U_array[0] = u;

   Vtemp = hypre_SeqVectorCreate(num_variables[0]);
   hypre_SeqVectorInitialize(Vtemp);
   hypre_AMGDataVtemp(amg_data) = Vtemp;

   for (j = 1; j < num_levels; j++)
   {
      num_coeffs[j]    = hypre_CSRMatrixNumNonzeros(A_array[j]);
      num_variables[j] = hypre_CSRMatrixNumRows(A_array[j]);
   }

   /*-----------------------------------------------------------------------
    *    Write the solver parameters
    *-----------------------------------------------------------------------*/

   /*if (amg_ioutdat > 0)
      hypre_WriteSolverParams(amg_data); */


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

   if (amg_ioutdat > 1)
   { 

      hypre_printf("\n\nAMG SOLUTION INFO:\n");

   }

   /*-----------------------------------------------------------------------
    *    Compute initial fine-grid residual and print to logfile
    *-----------------------------------------------------------------------*/

   hypre_SeqVectorCopy(F_array[0],Vtemp);
   hypre_CSRMatrixMatvec(alpha,A_array[0],U_array[0],beta,Vtemp);
   resid_nrm = sqrt(hypre_SeqVectorInnerProd(Vtemp,Vtemp));

   resid_nrm_init = resid_nrm;
   rhs_norm = sqrt(hypre_SeqVectorInnerProd(f,f));
   relative_resid = 9999;
   if (rhs_norm)
   {
      relative_resid = resid_nrm_init / rhs_norm;
   }
   else
   {
      relative_resid = resid_nrm_init;
   }
   

   if (amg_ioutdat == 2 || amg_ioutdat == 3)
   {     
      hypre_printf("                                            relative\n");
      hypre_printf("               residual        factor       residual\n");
      hypre_printf("               --------        ------       --------\n");
      hypre_printf("    Initial    %e                 %e\n",resid_nrm_init,
              relative_resid);
   }

   /*-----------------------------------------------------------------------
    *    Main V-cycle loop
    *-----------------------------------------------------------------------*/
   
   while (relative_resid >= tol && cycle_count < max_iter 
          && Solve_err_flag == 0)
   {
      hypre_AMGDataCycleOpCount(amg_data) = 0;   
      /* Op count only needed for one cycle */

      Solve_err_flag = hypre_AMGCycle(amg_data, F_array, U_array); 

      old_resid = resid_nrm;

      /*---------------------------------------------------------------
       *    Compute  fine-grid residual and residual norm
       *----------------------------------------------------------------*/

      hypre_SeqVectorCopy(F_array[0],Vtemp);
      hypre_CSRMatrixMatvec(alpha,A_array[0],U_array[0],beta,Vtemp);
      resid_nrm = sqrt(hypre_SeqVectorInnerProd(Vtemp,Vtemp));

      conv_factor = resid_nrm / old_resid;
      relative_resid = 9999;
      if (rhs_norm)
      {
         relative_resid = resid_nrm / rhs_norm;
      }
      else
      {
         relative_resid = resid_nrm;
      }

      ++cycle_count;

      if (amg_ioutdat == 2 || amg_ioutdat == 3)
      { 
         hypre_printf("    Cycle %2d   %e    %f     %e \n",cycle_count,
                 resid_nrm,conv_factor,relative_resid);
      }
   }

   if (cycle_count == max_iter) Solve_err_flag = 1;

   /*-----------------------------------------------------------------------
    *    Compute closing statistics
    *-----------------------------------------------------------------------*/

   conv_factor = pow((resid_nrm/resid_nrm_init),(1.0/((double) cycle_count)));


   for (j=0;j<hypre_AMGDataNumLevels(amg_data);j++)
   {
      total_coeffs += num_coeffs[j];
      total_variables += num_variables[j];
   }

   /* cycle_op_count = hypre_AMGDataCycleOpCount(amg_data); */

   grid_cmplxty = ((double) total_variables) / ((double) num_variables[0]);
   operat_cmplxty = ((double) total_coeffs) / ((double) num_coeffs[0]);
   /* cycle_cmplxty = ((double) cycle_op_count) / ((double) num_coeffs[0]); */

   if (amg_ioutdat > 1)
   {
      if (Solve_err_flag == 1)
      {
         hypre_printf("\n\n==============================================");
         hypre_printf("\n NOTE: Convergence tolerance was not achieved\n");
         hypre_printf("      within the allowed %d V-cycles\n",max_iter);
         hypre_printf("==============================================");
      }
      hypre_printf("\n\n Average Convergence Factor = %f",conv_factor);
      hypre_printf("\n\n     Complexity:    grid = %f\n",grid_cmplxty);
      hypre_printf("                operator = %f\n",operat_cmplxty);
      /*  hypre_printf("                   cycle = %f\n\n",cycle_cmplxty); */
   }

   /*----------------------------------------------------------
    * Close the output file (if open)
    *----------------------------------------------------------*/

   hypre_TFree(num_coeffs);
   hypre_TFree(num_variables);

   return(Solve_err_flag);
}

