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


/*--------------------------------------------------------------------------
 * amg_Solve
 *--------------------------------------------------------------------------*/

int         amg_Solve(u, f, tol, data)
Vector      *u;
Vector      *f;
double       tol;
void        *data;
{

/* Data Structure variables */
   int      amg_ioutdat;
   int      num_cycles;
   char    *file_name;
   Matrix **A_array;

/*  Local variables  */
   FILE    *fp;
   int      j;
   int      Solve_err_flag;
   int     *iarr;
   int     *num_variables;
   int     *num_coeffs;
   int      num_unknowns;
   int      num_digits;
   int      num_integers;
   int      Vstar_flag;
   int      Fcycle_flag;
   int      num_Vcycles;
   int      cycle_count;
   int      total_coeffs;
   int      total_variables;
   int      cycle_op_count;
   double   cycle_cmplxty;
   double   operat_cmplxty;
   double   grid_cmplxty;
   double   conv_factor;
   double   resid_nrm;
   double   resid_nrm_init;
   double   relative_resid;
   double   rhs_norm;
   double   energy;
   double  *resv;
   double   old_energy;
   double   old_resid;

   AMGData  *amg_data = data;

   amg_ioutdat = AMGDataIOutDat(amg_data);
   num_cycles = AMGDataNCyc(amg_data);
   file_name = AMGDataLogFileName(amg_data);
   num_unknowns = AMGDataNumUnknowns(amg_data);
   A_array = AMGDataAArray(amg_data);
   num_coeffs = AMGDataNumA(amg_data);
   num_variables = AMGDataNumV(amg_data);

   iarr = ctalloc(int, 10);
   resv = ctalloc(double, num_unknowns);

/*--------------------------------------------------------------------------
 *    Write the solver parameters
 *--------------------------------------------------------------------------*/

   WriteSolverParams(tol, amg_data);


/*    CALL_SOLVE(Solve_err_flag, u, f, tol, amg_data);   */

/*--------------------------------------------------------------------------
 *    Initialize the solver error flag and assorted bookkeeping variables
 *--------------------------------------------------------------------------*/

   Solve_err_flag = 0;

   total_coeffs = 0;
   total_variables = 0;
   cycle_count = 0;
   operat_cmplxty = 0;
   grid_cmplxty = 0;

/*--------------------------------------------------------------------------
 *     open the log file and write some initial info
 *--------------------------------------------------------------------------*/

   if (amg_ioutdat >= 0)
   { 
      fp = fopen(file_name, "a");

      fprintf(fp,"\n\nAMG SOLUTION INFO:\n");

    }

/*--------------------------------------------------------------------------
 *    Decode num_cycles
 *--------------------------------------------------------------------------*/
    
    num_integers = 3;
    idec_(&num_cycles,&num_integers,&num_digits,iarr);
    Vstar_flag = iarr[0]-1;
    Fcycle_flag = iarr[1];
    num_Vcycles = iarr[2];


/*--------------------------------------------------------------------------
 *    Find initial residual and print to logfile
 *--------------------------------------------------------------------------*/

   CALL_RESIDUAL(energy,resid_nrm,resv,u,f,A_array,amg_data);

   resid_nrm_init = resid_nrm;
   rhs_norm = sqrt(InnerProd(f,f));
   relative_resid = resid_nrm_init / rhs_norm;

   if (amg_ioutdat == 1 || amg_ioutdat == 3)
   {     
      fprintf(fp,"                                           relative\n");
      fprintf(fp,"              residual        factor       residual\n");
      fprintf(fp,"              --------        ------       --------\n");
      fprintf(fp,"    Initial   %e                 %e\n",resid_nrm_init,
                                                        relative_resid);
   }

/*--------------------------------------------------------------------------
 *    Main V-cycle loop
 *--------------------------------------------------------------------------*/
   
   while (relative_resid >= tol && cycle_count < num_Vcycles 
                                && Solve_err_flag == 0)
   {
         cycle_op_count = 0;   /* Count only needed for one cycle */

         CALL_CYCLE(Solve_err_flag, u, f, tol, amg_data);

         old_energy = energy;
         old_resid = resid_nrm;

         CALL_RESIDUAL(energy,resid_nrm,resv,u,f,A_array,amg_data);

         conv_factor = resid_nrm / old_resid;
         relative_resid = resid_nrm / rhs_norm;

         ++cycle_count;

         if (amg_ioutdat == 1 || amg_ioutdat == 3)
         { 
            fprintf(fp,"    Cycle %d   %e    %f     %e \n",cycle_count,
                             resid_nrm,conv_factor,relative_resid);
         }
   }

   if (cycle_count == num_Vcycles) Solve_err_flag = 1;

/*--------------------------------------------------------------------------
 *    Compute closing statistics
 *--------------------------------------------------------------------------*/

   conv_factor = pow((resid_nrm/resid_nrm_init),(1.0/((double) cycle_count)));


   for (j=0;j<AMGDataNumLevels(amg_data);j++)
   {
       total_coeffs += num_coeffs[j];
       total_variables += num_variables[j];
   }

   grid_cmplxty = ((double) total_variables) / ((double) num_variables[0]);
   operat_cmplxty = ((double) total_coeffs) / ((double) num_coeffs[0]);
   cycle_cmplxty = ((double) cycle_op_count) / ((double) num_coeffs[0]);

   if (amg_ioutdat >= 0)
   {
       if (Solve_err_flag == 1)
       {
           fprintf(fp,"\n\n==============================================");
           fprintf(fp,"\n NOTE: Convergence tolerance was not achieved\n");
           fprintf(fp,"      within the allowed %d V-cycles\n",num_Vcycles);
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

    if (amg_ioutdat >= 0)
    { 
       fclose(fp);
    }

   return(Solve_err_flag);
}

