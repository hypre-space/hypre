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
 * hypre_BoomerAMGSolve
 *--------------------------------------------------------------------*/

int
hypre_BoomerAMGSolve( void               *amg_vdata,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector    *f,
                   hypre_ParVector    *u         )
{

   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(A);   

   hypre_ParAMGData   *amg_data = amg_vdata;

   /* Data Structure variables */

   int      amg_print_level;
   int      amg_log_level;
   int     *num_coeffs;
   int     *num_variables;
   int      cycle_op_count;
   int      num_levels;
   /* int      num_unknowns; */
   double   tol;

   hypre_ParCSRMatrix **A_array;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;

   /*  Local variables  */

   int      j;
   int      Solve_err_flag;
   int      min_iter;
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
   double   ieee_check = 0.;

   hypre_ParVector  *Vtemp;
   hypre_ParVector  *Residual;

   MPI_Comm_size(comm, &num_procs);   
   MPI_Comm_rank(comm,&my_id);

   amg_print_level    = hypre_ParAMGDataPrintLevel(amg_data);
   amg_log_level      = hypre_ParAMGDataLogLevel(amg_data);
   if ( amg_log_level > 2 )
      Residual = hypre_ParAMGDataResidual(amg_data);
   /* num_unknowns  = hypre_ParAMGDataNumUnknowns(amg_data); */
   num_levels       = hypre_ParAMGDataNumLevels(amg_data);
   A_array          = hypre_ParAMGDataAArray(amg_data);
   F_array          = hypre_ParAMGDataFArray(amg_data);
   U_array          = hypre_ParAMGDataUArray(amg_data);

   tol              = hypre_ParAMGDataTol(amg_data);
   min_iter         = hypre_ParAMGDataMinIter(amg_data);
   max_iter         = hypre_ParAMGDataMaxIter(amg_data);

   num_coeffs       = hypre_CTAlloc(int, num_levels);
   num_variables    = hypre_CTAlloc(int, num_levels);
   num_coeffs[0]    = hypre_ParCSRMatrixNumNonzeros(A);
   num_variables[0] = hypre_ParCSRMatrixGlobalNumRows(A);
 
   A_array[0] = A;
   F_array[0] = f;
   U_array[0] = u;

/*   Vtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                 hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                 hypre_ParCSRMatrixRowStarts(A_array[0]));
   hypre_ParVectorInitialize(Vtemp);
   hypre_ParVectorSetPartitioningOwner(Vtemp,0);
   hypre_ParAMGDataVtemp(amg_data) = Vtemp;
*/
   Vtemp = hypre_ParAMGDataVtemp(amg_data);
   for (j = 1; j < num_levels; j++)
   {
      num_coeffs[j]    = hypre_ParCSRMatrixNumNonzeros(A_array[j]);
      num_variables[j] = hypre_ParCSRMatrixGlobalNumRows(A_array[j]);
   }

   /*-----------------------------------------------------------------------
    *    Write the solver parameters
    *-----------------------------------------------------------------------*/


   if (my_id == 0 && amg_print_level > 1)
      hypre_BoomerAMGWriteSolverParams(amg_data); 

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
    *     write some initial info
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && amg_print_level > 1 && tol > 0.)
     printf("\n\nAMG SOLUTION INFO:\n");


   /*-----------------------------------------------------------------------
    *    Compute initial fine-grid residual and print 
    *-----------------------------------------------------------------------*/

   if (tol > 0.)
   {
     if ( amg_log_level > 2 ) {
        hypre_ParVectorCopy(F_array[0], Residual );
        hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Residual );
        resid_nrm = sqrt(hypre_ParVectorInnerProd( Residual, Residual ));
     }
     else {
        hypre_ParVectorCopy(F_array[0], Vtemp);
        hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Vtemp);
        resid_nrm = sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
     }

     /* Since it is does not diminish performance, attempt to return an error flag
        and notify users when they supply bad input. */
     if (resid_nrm != 0.) ieee_check = resid_nrm/resid_nrm; /* INF -> NaN conversion */
     if (ieee_check != ieee_check)
     {
        /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
           for ieee_check self-equality works on all IEEE-compliant compilers/
           machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
           by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
           found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
        if (amg_print_level > 0)
        {
          printf("\n\nERROR detected by Hypre ...  BEGIN\n");
          printf("ERROR -- hypre_BoomerAMGSolve: INFs and/or NaNs detected in input.\n");
          printf("User probably placed non-numerics in supplied A, x_0, or b.\n");
          printf("Returning error flag += 101.  Program not terminated.\n");
          printf("ERROR detected by Hypre ...  END\n\n\n");
        }
        Solve_err_flag += 101;
        return Solve_err_flag;
     }

     resid_nrm_init = resid_nrm;
     rhs_norm = sqrt(hypre_ParVectorInnerProd(f, f));
     if (rhs_norm)
     {
       relative_resid = resid_nrm_init / rhs_norm;
     }
     else
     {
       relative_resid = resid_nrm_init;
     }
   }
   else
   {
     relative_resid = 1.;
   }

   if (my_id == 0 && amg_print_level > 1 && tol > 0.)
   {     
      printf("                                            relative\n");
      printf("               residual        factor       residual\n");
      printf("               --------        ------       --------\n");
      printf("    Initial    %e                 %e\n",resid_nrm_init,
              relative_resid);
   }

   /*-----------------------------------------------------------------------
    *    Main V-cycle loop
    *-----------------------------------------------------------------------*/
   
   while ((relative_resid >= tol || cycle_count < min_iter)
          && cycle_count < max_iter
          && Solve_err_flag == 0)
   {
      hypre_ParAMGDataCycleOpCount(amg_data) = 0;   
      /* Op count only needed for one cycle */

      Solve_err_flag = hypre_BoomerAMGCycle(amg_data, F_array, U_array); 

      /*---------------------------------------------------------------
       *    Compute  fine-grid residual and residual norm
       *----------------------------------------------------------------*/

      if (tol > 0.)
      {
        old_resid = resid_nrm;

        if ( amg_log_level > 2 ) {
           hypre_ParVectorCopy(F_array[0], Residual);
           hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Residual );
           resid_nrm = sqrt(hypre_ParVectorInnerProd( Residual, Residual ));
        }
        else {
           hypre_ParVectorCopy(F_array[0], Vtemp);
           hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Vtemp);
           resid_nrm = sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
        }

        conv_factor = resid_nrm / old_resid;
        if (rhs_norm)
        {
           relative_resid = resid_nrm / rhs_norm;
        }
        else
        {
           relative_resid = resid_nrm;
        }

        hypre_ParAMGDataRelativeResidualNorm(amg_data) = relative_resid;
      }

      ++cycle_count;

      hypre_ParAMGDataNumIterations(amg_data) = cycle_count;

      if (my_id == 0 && amg_print_level > 1 && tol > 0.)
      { 
         printf("    Cycle %2d   %e    %f     %e \n", cycle_count,
                 resid_nrm, conv_factor, relative_resid);
      }
   }

   if (cycle_count == max_iter && tol > 0.) Solve_err_flag = 1;

   /*-----------------------------------------------------------------------
    *    Compute closing statistics
    *-----------------------------------------------------------------------*/

   if (cycle_count > 0 && tol > 0.) 
     conv_factor = pow((resid_nrm/resid_nrm_init),(1.0/((double) cycle_count)));
   else
     conv_factor = 1.;


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

   if (my_id == 0 && amg_print_level > 1)
   {
      if (Solve_err_flag == 1)
      {
         printf("\n\n==============================================");
         printf("\n NOTE: Convergence tolerance was not achieved\n");
         printf("      within the allowed %d V-cycles\n",max_iter);
         printf("==============================================");
      }
      if (tol > 0.)
        printf("\n\n Average Convergence Factor = %f",conv_factor);
      printf("\n\n     Complexity:    grid = %f\n",grid_cmplxty);
      printf("                operator = %f\n",operat_cmplxty);
      printf("                   cycle = %f\n\n\n\n",cycle_cmplxty);
   }

   hypre_TFree(num_coeffs);
   hypre_TFree(num_variables);

   return(Solve_err_flag);
}

