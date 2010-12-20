/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * AMG solve routine
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * HYPRE_AMGSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int         HYPRE_AMGSolve(u, f, tol, data)
hypre_Vector      *u;
hypre_Vector      *f;
double       tol;
void        *data;
{

/* Data Structure variables */

   HYPRE_Int      amg_ioutdat;
   HYPRE_Int      cycle_control;
   HYPRE_Int     *levv;
   HYPRE_Int     *num_coeffs;
   HYPRE_Int     *num_variables;
   HYPRE_Int      cycle_op_count;
   HYPRE_Int      Vstar_flag;
   HYPRE_Int      Fcycle_flag;
   HYPRE_Int      num_levels;
   HYPRE_Int      num_unknowns;
   char    *file_name;
   hypre_Matrix **A_array;

/*  Local variables  */

   FILE    *fp;

   HYPRE_Int      j;
   HYPRE_Int      Solve_err_flag;
   HYPRE_Int     *iarr;
   HYPRE_Int      num_digits;
   HYPRE_Int      num_integers;
   HYPRE_Int      num_Vcycles;
   HYPRE_Int      cycle_count;
   HYPRE_Int      total_coeffs;
   HYPRE_Int      total_variables;

   double   alpha = 1.0;
   double   beta = -1.0;
   double   cycle_cmplxty;
   double   operat_cmplxty;
   double   grid_cmplxty;
   double   conv_factor;
   double   resid_nrm;
   double   resid_nrm_tmp;
   double   resid_nrm_init;
   double   relative_resid;
   double   rhs_norm;
   double   energy;
   double  *tmpvec;
   double   old_energy;
   double   old_resid;

   hypre_Vector **F_array;
   hypre_Vector **U_array;
   hypre_Vector  *Vtemp;

   hypre_AMGData  *amg_data = data;

   amg_ioutdat   = hypre_AMGDataIOutDat(amg_data);
   cycle_control = hypre_AMGDataNCyc(amg_data);
   file_name     = hypre_AMGDataLogFileName(amg_data);
   num_unknowns  = hypre_AMGDataNumUnknowns(amg_data);
   num_levels    = hypre_AMGDataNumLevels(amg_data);
   A_array       = hypre_AMGDataAArray(amg_data);
   num_coeffs    = hypre_AMGDataNumA(amg_data);
   num_variables = hypre_AMGDataNumV(amg_data);
   levv          = hypre_AMGDataLevV(amg_data);
   
   iarr = hypre_CTAlloc(HYPRE_Int, 10);

   F_array = hypre_TAlloc(hypre_Vector*, num_levels);
   U_array = hypre_TAlloc(hypre_Vector*, num_levels);
 
   F_array[0] = f;
   U_array[0] = u;


   Vtemp = hypre_AMGDataVtemp(amg_data);

   for (j = 1; j < num_levels; j++)
   {
       F_array[j] = hypre_NewVector(&(f->data[levv[j]-1]), num_variables[j]);
       U_array[j] = hypre_NewVector(&(u->data[levv[j]-1]), num_variables[j]);
   }

/*********  the following does not work at this time 

   for (j = 1; j < num_levels; j++)
   {
       tmpvec = hypre_CTAlloc(double, num_variables[j]);
       F_array[j] = hypre_NewVector(tmpvec, num_variables[j]);

       tmpvec = hypre_CTAlloc(double, num_variables[j]);
       U_array[j] = hypre_NewVector(tmpvec, num_variables[j]);
   }

*************/ 
  

/*--------------------------------------------------------------------------
 *    Write the solver parameters
 *--------------------------------------------------------------------------*/

   hypre_WriteSolverParams(tol, amg_data);


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

      hypre_fprintf(fp,"\n\nAMG SOLUTION INFO:\n");

    }

/*--------------------------------------------------------------------------
 *    Decode cycle_control, load flags into data structure
 *--------------------------------------------------------------------------*/
    
    num_integers = 3;
    idec_(&cycle_control,&num_integers,&num_digits,iarr);
    Vstar_flag = iarr[0]-1;
    Fcycle_flag = iarr[1];
    num_Vcycles = iarr[2];

    hypre_AMGDataFcycleFlag(amg_data) = Fcycle_flag;
    hypre_AMGDataVstarFlag(amg_data) = Vstar_flag;


/*--------------------------------------------------------------------------
 *    Compute initial fine-grid residual and print to logfile
 *--------------------------------------------------------------------------*/

   hypre_CopyVector(F_array[0],Vtemp);
   hypre_Matvec(alpha,A_array[0],U_array[0],beta,Vtemp);
   resid_nrm = sqrt(hypre_InnerProd(Vtemp,Vtemp));

   resid_nrm_init = resid_nrm;
   rhs_norm = sqrt(hypre_InnerProd(f,f));
   relative_resid = resid_nrm_init / rhs_norm;

   if (amg_ioutdat == 1 || amg_ioutdat == 3)
   {     
      hypre_fprintf(fp,"                                            relative\n");
      hypre_fprintf(fp,"               residual        factor       residual\n");
      hypre_fprintf(fp,"               --------        ------       --------\n");
      hypre_fprintf(fp,"    Initial    %e                 %e\n",resid_nrm_init,
                                                        relative_resid);
   }

/*--------------------------------------------------------------------------
 *    Main V-cycle loop
 *--------------------------------------------------------------------------*/
   
   while (relative_resid >= tol && cycle_count < num_Vcycles 
                                && Solve_err_flag == 0)
   {
         hypre_AMGDataCycleOpCount(amg_data) = 0;   
                        /* Op count only needed for one cycle */

         Solve_err_flag = hypre_AMGCycle(U_array,F_array,tol,amg_data);

         old_energy = energy;
         old_resid = resid_nrm;

         /*---------------------------------------------------------------
          *    Compute  fine-grid residual and residual norm
          *----------------------------------------------------------------*/

         hypre_CopyVector(F_array[0],Vtemp);
         hypre_Matvec(alpha,A_array[0],U_array[0],beta,Vtemp);
         resid_nrm = sqrt(hypre_InnerProd(Vtemp,Vtemp));

         conv_factor = resid_nrm / old_resid;
         relative_resid = resid_nrm / rhs_norm;

         ++cycle_count;

         if (amg_ioutdat == 1 || amg_ioutdat == 3)
         { 
            hypre_fprintf(fp,"    Cycle %2d   %e    %f     %e \n",cycle_count,
                             resid_nrm,conv_factor,relative_resid);
         }
   }

   if (cycle_count == num_Vcycles) Solve_err_flag = 1;

/*--------------------------------------------------------------------------
 *    Compute closing statistics
 *--------------------------------------------------------------------------*/

   conv_factor = pow((resid_nrm/resid_nrm_init),(1.0/((double) cycle_count)));


   for (j=0;j<hypre_AMGDataNumLevels(amg_data);j++)
   {
       total_coeffs += num_coeffs[j];
       total_variables += num_variables[j];
   }

   cycle_op_count = hypre_AMGDataCycleOpCount(amg_data);

   grid_cmplxty = ((double) total_variables) / ((double) num_variables[0]);
   operat_cmplxty = ((double) total_coeffs) / ((double) num_coeffs[0]);
   cycle_cmplxty = ((double) cycle_op_count) / ((double) num_coeffs[0]);

   if (amg_ioutdat >= 0)
   {
       if (Solve_err_flag == 1)
       {
           hypre_fprintf(fp,"\n\n==============================================");
           hypre_fprintf(fp,"\n NOTE: Convergence tolerance was not achieved\n");
           hypre_fprintf(fp,"      within the allowed %d V-cycles\n",num_Vcycles);
           hypre_fprintf(fp,"==============================================");
       }
       hypre_fprintf(fp,"\n\n Average Convergence Factor = %f",conv_factor);
       hypre_fprintf(fp,"\n\n     Complexity:    grid = %f\n",grid_cmplxty);
       hypre_fprintf(fp,"                operator = %f\n",operat_cmplxty);
       hypre_fprintf(fp,"                   cycle = %f\n\n",cycle_cmplxty);
   }

   

/*----------------------------------------------------------
 * Close the output file (if open)
 *----------------------------------------------------------*/

    if (amg_ioutdat >= 0)
    { 
       fclose(fp);
    }

   hypre_TFree(iarr);

   hypre_TFree(F_array);
   hypre_TFree(U_array);

   return(Solve_err_flag);
}

