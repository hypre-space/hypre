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

   int      num_levels; 
   int      num_nonzeros;
   int      amg_ioutdat;
   char    *log_file_name;
 
   /* Local variables */

   FILE      *fp;

   int      *A_i;
   int      *A_j;
   double   *A_data;

   int       level;
   int       i,j;
   int       fine_size;
   int       entries;
   int       total_entries;
   int       min_entries;
   int       max_entries;
   double    avg_entries;
   double    rowsum;
   double    min_rowsum;
   double    max_rowsum;
   double    sparse;

   A_array = hypre_AMGDataAArray(amg_data);
   num_levels = hypre_AMGDataNumLevels(amg_data);
   amg_ioutdat = hypre_AMGDataIOutDat(amg_data);
   log_file_name = hypre_AMGDataLogFileName(amg_data);
   
   fp = fopen(hypre_AMGDataLogFileName(amg_data),"a");
 
   fprintf(fp,"\n  AMG SETUP PARAMETERS:\n\n");
   fprintf(fp," Max levels = %d\n",hypre_AMGDataMaxLevels(amg_data));
   fprintf(fp," Num levels = %d\n\n",num_levels);

   fprintf(fp,"           nonzero          entries p");
   fprintf(fp,"er row        row sums\n");
   fprintf(fp,"lev rows entries sparse min max  ");
   fprintf(fp,"avg    min      max  avg\n");
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

       for (j = 0; j < fine_size; j++)
       {
           entries = A_i[j+1] - A_i[j];
           min_entries = min(entries, min_entries);
           max_entries = max(entries, max_entries);
           total_entries += entries;

           rowsum = 0.0;
           for (i = A_i[j]; i < A_i[j+1]; i++)
               rowsum += A_data[j];

           min_rowsum = min(rowsum, min_rowsum);
           max_rowsum = max(rowsum, max_rowsum);
       }

       avg_entries = ((double) total_entries) / ((double) fine_size);

       fprintf(fp, " %d  %d  %d  %f  %d  %d",
                 level, fine_size, num_nonzeros, sparse, min_entries, 
                 max_entries);
       fprintf(fp,"  %e  %e  %e\n", avg_entries,
                                 min_rowsum, max_rowsum);
   }
       
   
   fclose(fp);
   return(0);
}  


