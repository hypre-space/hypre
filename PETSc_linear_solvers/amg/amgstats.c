/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "headers.h"

/******************************************************************************
 *
 * Routine for getting matrix statistics from setup
 *
 ******************************************************************************/


int hypre_AMGSetupStats(hypre_AMGData *amg_data)

{

   /* Data Structure variables */

   hypre_CSRMatrix **A_array;
   hypre_CSRMatrix **P_array;

   int      num_levels; 
   int      num_nonzeros;
   int      amg_ioutdat;
   char    *log_file_name;
 
   /* Local variables */

   FILE      *fp;

   int      *A_i;
   int      *A_j;
   double   *A_data;

   int      *P_i;
   int      *P_j;
   double   *P_data;

   int       level;
   int       i,j;
   int       fine_size;
   int       coarse_size;
   int       entries;
   int       total_entries;
   int       min_entries;
   int       max_entries;
   double    avg_entries;
   double    rowsum;
   double    min_rowsum;
   double    max_rowsum;
   double    sparse;
   double    min_weight;
   double    max_weight;

   A_array = hypre_AMGDataAArray(amg_data);
   P_array = hypre_AMGDataPArray(amg_data);
   num_levels = hypre_AMGDataNumLevels(amg_data);
   amg_ioutdat = hypre_AMGDataIOutDat(amg_data);
   log_file_name = hypre_AMGDataLogFileName(amg_data);
   
   fp = fopen(hypre_AMGDataLogFileName(amg_data),"a");
 
   fprintf(fp,"\n  AMG SETUP PARAMETERS:\n\n");
   fprintf(fp," Max levels = %d\n",hypre_AMGDataMaxLevels(amg_data));
   fprintf(fp," Num levels = %d\n\n",num_levels);

   fprintf(fp, "\nOperator Matrix Information:\n\n");

   fprintf(fp,"         nonzero         entries p");
   fprintf(fp,"er row        row sums\n");
   fprintf(fp,"lev rows entries  sparse  min max  ");
   fprintf(fp,"avg       min         max\n");
   fprintf(fp,"=======================================");
   fprintf(fp,"==========================\n");

  
   /*-----------------------------------------------------
    *  Enter Statistics Loop
    *-----------------------------------------------------*/

   for (level = 0; level < num_levels; level++)
   {
       A_i = hypre_CSRMatrixI(A_array[level]);
       A_j = hypre_CSRMatrixJ(A_array[level]);
       A_data = hypre_CSRMatrixData(A_array[level]);

       fine_size = hypre_CSRMatrixNumRows(A_array[level]);
       num_nonzeros = hypre_CSRMatrixNumNonzeros(A_array[level]);
       sparse = num_nonzeros /((double) fine_size * (double) fine_size);

       min_entries = A_i[1]-A_i[0];
       max_entries = 0;
       total_entries = 0;
       min_rowsum = 0.0;
       max_rowsum = 0.0;

       for (j = A_i[0]; j < A_i[1]; j++)
                    min_rowsum += A_data[j];

       max_rowsum = min_rowsum;

       for (j = 0; j < fine_size; j++)
       {
           entries = A_i[j+1] - A_i[j];
           min_entries = min(entries, min_entries);
           max_entries = max(entries, max_entries);
           total_entries += entries;

           rowsum = 0.0;
           for (i = A_i[j]; i < A_i[j+1]; i++)
               rowsum += A_data[i];

           min_rowsum = min(rowsum, min_rowsum);
           max_rowsum = max(rowsum, max_rowsum);
       }

       avg_entries = ((double) total_entries) / ((double) fine_size);

       fprintf(fp, "%2d %5d %7d  %0.3f  %3d %3d",
                 level, fine_size, num_nonzeros, sparse, min_entries, 
                 max_entries);
       fprintf(fp,"  %4.1f  %10.3e  %10.3e\n", avg_entries,
                                 min_rowsum, max_rowsum);
   }
       
   fprintf(fp, "\n\nInterpolation Matrix Information:\n\n");

   fprintf(fp,"                 entries/row    min     max");
   fprintf(fp,"         row sums\n");
   fprintf(fp,"lev  rows cols    min max  ");
   fprintf(fp,"   weight   weight     min       max \n");
   fprintf(fp,"=======================================");
   fprintf(fp,"==========================\n");

  
   /*-----------------------------------------------------
    *  Enter Statistics Loop
    *-----------------------------------------------------*/

   for (level = 0; level < num_levels-1; level++)
   {
       P_i = hypre_CSRMatrixI(P_array[level]);
       P_j = hypre_CSRMatrixJ(P_array[level]);
       P_data = hypre_CSRMatrixData(P_array[level]);

       fine_size = hypre_CSRMatrixNumRows(P_array[level]);
       coarse_size = hypre_CSRMatrixNumCols(P_array[level]);
       num_nonzeros = hypre_CSRMatrixNumNonzeros(P_array[level]);

       min_entries = P_i[1]-P_i[0];
       max_entries = 0;
       total_entries = 0;
       min_rowsum = 0.0;
       max_rowsum = 0.0;
       min_weight = P_data[0];
       max_weight = P_data[0];

       for (j = P_i[0]; j < P_i[1]; j++)
                    min_rowsum += P_data[j];

       max_rowsum = min_rowsum;

       for (j = 0; j < num_nonzeros; j++)
       {
          if (P_data[j] != 1.0)
          {
             min_weight = min(min_weight,P_data[j]);
             max_weight = max(max_weight,P_data[j]);
          }
       }

       for (j = 0; j < fine_size; j++)
       {
           entries = P_i[j+1] - P_i[j];
           min_entries = min(entries, min_entries);
           max_entries = max(entries, max_entries);
           total_entries += entries;

           rowsum = 0.0;
           for (i = P_i[j]; i < P_i[j+1]; i++)
               rowsum += P_data[i];

           min_rowsum = min(rowsum, min_rowsum);
           max_rowsum = max(rowsum, max_rowsum);
       }

       fprintf(fp, "%2d %5d x %-5d %3d %3d",
             level, fine_size, coarse_size,  min_entries, max_entries);
       fprintf(fp,"  %5.3e  %5.3e %5.3e  %5.3e\n",
                 min_weight, max_weight, min_rowsum, max_rowsum);
   }
       
   
   fclose(fp);
   return(0);
}  



/*---------------------------------------------------------------
 * hypre_WriteSolveParams
 *---------------------------------------------------------------*/


void     hypre_WriteSolverParams(data)
void    *data;
 
{
   FILE    *fp;
   char    *file_name;
 
   hypre_AMGData  *amg_data = data;
 
   /* amg solve params */
   int      max_iter;
   int      cycle_type;    
   int     *num_grid_sweeps;  
   int     *grid_relax_type;   
   int    **grid_relax_points; 
   double   tol;
 
   /* amg output params */
   int      amg_ioutdat;
 
   int      j;
 
 
   /*----------------------------------------------------------
    * Get the amg_data data
    *----------------------------------------------------------*/
 
   file_name = hypre_AMGDataLogFileName(amg_data);
 
 
   max_iter   = hypre_AMGDataMaxIter(amg_data);
   cycle_type = hypre_AMGDataCycleType(amg_data);    
   num_grid_sweeps = hypre_AMGDataNumGridSweeps(amg_data);  
   grid_relax_type = hypre_AMGDataGridRelaxType(amg_data);
   grid_relax_points = hypre_AMGDataGridRelaxPoints(amg_data); 
   tol = hypre_AMGDataTol(amg_data);
 
   amg_ioutdat = hypre_AMGDataIOutDat(amg_data);
 
   /*----------------------------------------------------------
    * Open the output file
    *----------------------------------------------------------*/
 
   if (amg_ioutdat == 1 || amg_ioutdat == 3)
   { 
      fp = fopen(file_name, "a");
 
      fprintf(fp,"\n\nAMG SOLVER PARAMETERS:\n\n");
   
   /*----------------------------------------------------------
    * AMG info
    *----------------------------------------------------------*/
 
      fprintf(fp, "  Maximum number of cycles:         %d \n",max_iter);
      fprintf(fp, "  Stopping Tolerance:               %e \n",tol); 
      fprintf(fp, "  Cycle type (1 = V, 2 = W, etc.):  %d\n\n", cycle_type);
      fprintf(fp, "  Relaxation Parameters:\n");
      fprintf(fp, "   Visiting Grid:            fine  down   up  coarse\n");
      fprintf(fp, "   Number of partial sweeps:%4d  %4d   %2d  %4d \n",
              num_grid_sweeps[0],num_grid_sweeps[2],
              num_grid_sweeps[2],num_grid_sweeps[3]);
      fprintf(fp, "   Type 0=Jac, 1=GS, 9=GE:  %4d  %4d   %2d  %4d \n",
              grid_relax_type[0],grid_relax_type[2],
              grid_relax_type[2],grid_relax_type[3]);
      fprintf(fp, "   Point types, partial sweeps (1=C, -1=F):\n");
      fprintf(fp, "                               Finest grid:");
      for (j = 0; j < num_grid_sweeps[0]; j++)
              fprintf(fp,"  %2d", grid_relax_points[0][j]);
      fprintf(fp, "\n");
      fprintf(fp, "                  Pre-CG relaxation (down):");
      for (j = 0; j < num_grid_sweeps[1]; j++)
              fprintf(fp,"  %2d", grid_relax_points[1][j]);
      fprintf(fp, "\n");
      fprintf(fp, "                   Post-CG relaxation (up):");
      for (j = 0; j < num_grid_sweeps[2]; j++)
              fprintf(fp,"  %2d", grid_relax_points[2][j]);
      fprintf(fp, "\n");
      fprintf(fp, "                             Coarsest grid:");
      for (j = 0; j < num_grid_sweeps[3]; j++)
              fprintf(fp,"  %2d", grid_relax_points[3][j]);
      fprintf(fp, "\n\n");

      fprintf(fp, " Output flag (ioutdat): %d \n", amg_ioutdat);
 
   /*----------------------------------------------------------
    * Close the output file
    *----------------------------------------------------------*/
 
      fclose(fp);
   }
 
   return;
}
