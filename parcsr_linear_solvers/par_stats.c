/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "headers.h"
#include "par_amg.h"

/*****************************************************************************
 *
 * Routine for getting matrix statistics from setup
 *
 *****************************************************************************/


int
hypre_ParAMGSetupStats( void               *amg_vdata,
                        hypre_ParCSRMatrix *A         )
{
   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(A);   

   hypre_ParAMGData *amg_data = amg_vdata;

   /* Data Structure variables */

   hypre_ParCSRMatrix **A_array;
   hypre_ParCSRMatrix **P_array;

   hypre_CSRMatrix *A_diag;
   double          *A_diag_data;
   int             *A_diag_i;
   int             *A_diag_j;

   hypre_CSRMatrix *A_offd;   
   double          *A_offd_data;
   int             *A_offd_i;
   int             *A_offd_j;
   int              num_cols_A_offd;

   hypre_CSRMatrix *P_diag;
   double          *P_diag_data;
   int             *P_diag_i;
   int             *P_diag_j;

   hypre_CSRMatrix *P_offd;   
   double          *P_offd_data;
   int             *P_offd_i;
   int             *P_offd_j;
   int              num_cols_P_offd;

   int      num_levels; 
   int      global_nonzeros;
   int      amg_ioutdat;

   double  *send_buff;
   double  *gather_buff;

   char    *log_file_name;
 
   /* Local variables */

   FILE      *fp;

   int       level;
   int       i,j;
   int       fine_size;
   int       coarse_size;
   int       entries;
   int       total_entries;
   int       min_entries;
   int       max_entries;

   int       num_procs,my_id;

   double    avg_entries;
   double    rowsum;
   double    min_rowsum;
   double    max_rowsum;
   double    sparse;
   double    min_weight;
   double    max_weight;

   int       global_min_e;
   int       global_max_e;
   double    global_min_rsum;
   double    global_max_rsum;
   double    global_min_wt;
   double    global_max_wt;

   MPI_Comm_size(comm, &num_procs);   
   MPI_Comm_rank(comm,&my_id);

   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   num_levels = hypre_ParAMGDataNumLevels(amg_data);
   amg_ioutdat = hypre_ParAMGDataIOutDat(amg_data);
   log_file_name = hypre_ParAMGDataLogFileName(amg_data);

   send_buff     = hypre_CTAlloc(double, 6);
   gather_buff = hypre_CTAlloc(double,6*num_procs); 
   
   fp = fopen(hypre_ParAMGDataLogFileName(amg_data),"a");
 
   if (my_id==0)
   {
      fprintf(fp,"\nBoomerAMG SETUP PARAMETERS:\n\n");
      fprintf(fp," Max levels = %d\n",hypre_ParAMGDataMaxLevels(amg_data));
      fprintf(fp," Num levels = %d\n\n",num_levels);
      fprintf(fp," Strength Threshhold = %f\n\n", 
                         hypre_ParAMGDataStrongThreshold(amg_data));

      fprintf(fp, "\nOperator Matrix Information:\n\n");

      fprintf(fp,"         nonzero         entries p");
      fprintf(fp,"er row        row sums\n");
      fprintf(fp,"lev rows entries  sparse  min max  ");
      fprintf(fp,"avg       min         max\n");
      fprintf(fp,"=======================================");
      fprintf(fp,"==========================\n");
   }
  
   /*-----------------------------------------------------
    *  Enter Statistics Loop
    *-----------------------------------------------------*/


   for (level = 0; level < num_levels; level++)
   { 
       A_diag = hypre_ParCSRMatrixDiag(A_array[level]);
       A_diag_data = hypre_CSRMatrixData(A_diag);
       A_diag_i = hypre_CSRMatrixI(A_diag);
       A_diag_j = hypre_CSRMatrixJ(A_diag);

       A_offd = hypre_ParCSRMatrixOffd(A_array[level]);   
       A_offd_data = hypre_CSRMatrixData(A_offd);
       A_offd_i = hypre_CSRMatrixI(A_offd);
       A_offd_j = hypre_CSRMatrixJ(A_offd);
       num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);

       fine_size = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
       global_nonzeros = hypre_ParCSRMatrixNumNonzeros(A_array[level]);

       sparse = global_nonzeros /((double) fine_size * (double) fine_size);


       min_entries = 0;
       max_entries = 0;
       total_entries = 0;
       min_rowsum = 0.0;
       max_rowsum = 0.0;

  if (hypre_CSRMatrixNumNonzeros(A_diag))
  {
       min_entries = (A_diag_i[1]-A_diag_i[0])+(A_offd_i[1]-A_offd_i[0]);
       for (j = A_diag_i[0]; j < A_diag_i[1]; j++)
                    min_rowsum += A_diag_data[j];
       for (j = A_offd_i[0]; j < A_offd_i[1]; j++)
                    min_rowsum += A_offd_data[j];

       max_rowsum = min_rowsum;

       for (j = 0; j < hypre_CSRMatrixNumRows(A_diag); j++)
       {
           entries = (A_diag_i[j+1]-A_diag_i[j])+(A_offd_i[j+1]-A_offd_i[j]);
           min_entries = min(entries, min_entries);
           max_entries = max(entries, max_entries);

           rowsum = 0.0;
           for (i = A_diag_i[j]; i < A_diag_i[j+1]; i++)
               rowsum += A_diag_data[i];

           for (i = A_offd_i[j]; i < A_offd_i[j+1]; i++)
               rowsum += A_offd_data[i];

           min_rowsum = min(rowsum, min_rowsum);
           max_rowsum = max(rowsum, max_rowsum);
       }
  }
       avg_entries = ((double) global_nonzeros) / ((double) fine_size);

       send_buff[0] = (double) min_entries;
       send_buff[1] = (double) max_entries;
       send_buff[2] = min_rowsum;
       send_buff[3] = max_rowsum;

       MPI_Gather(send_buff,4,MPI_DOUBLE,gather_buff,4,MPI_DOUBLE,0,comm);

       if (my_id == 0)
       {
          global_min_e = (int) gather_buff[0];
          global_max_e = (int) gather_buff[1];
          global_min_rsum = gather_buff[2];
          global_max_rsum = gather_buff[3];
          
          for (j = 0; j < num_procs; j++)
          {
              global_min_e = min(global_min_e, (int) gather_buff[my_id*4]);
              global_max_e = max(global_max_e, (int) gather_buff[my_id*4 +1]);
              global_min_rsum = min(global_min_rsum, gather_buff[my_id*4 +2]);
              global_max_rsum = max(global_max_rsum, gather_buff[my_id*4 +3]);
          }

          fprintf(fp, "%2d %5d %7d  %0.3f  %3d %3d",
                    level, fine_size, global_nonzeros, sparse, global_min_e, 
                    global_max_e);
          fprintf(fp,"  %4.1f  %10.3e  %10.3e\n", avg_entries,
                                    global_min_rsum, global_max_rsum);
       }

   }


       
   if (my_id == 0)
   {
      fprintf(fp, "\n\nInterpolation Matrix Information:\n\n");

      fprintf(fp,"                 entries/row    min     max");
      fprintf(fp,"         row sums\n");
      fprintf(fp,"lev  rows cols    min max  ");
      fprintf(fp,"   weight   weight     min       max \n");
      fprintf(fp,"=======================================");
      fprintf(fp,"==========================\n");
   }
  
   /*-----------------------------------------------------
    *  Enter Statistics Loop
    *-----------------------------------------------------*/


   for (level = 0; level < num_levels-1; level++)
   {

       P_diag = hypre_ParCSRMatrixDiag(P_array[level]);
       P_diag_data = hypre_CSRMatrixData(P_diag);
       P_diag_i = hypre_CSRMatrixI(P_diag);
       P_diag_j = hypre_CSRMatrixJ(P_diag);

       P_offd = hypre_ParCSRMatrixOffd(P_array[level]);   
       P_offd_data = hypre_CSRMatrixData(P_offd);
       P_offd_i = hypre_CSRMatrixI(P_offd);
       P_offd_j = hypre_CSRMatrixJ(P_offd);
       num_cols_P_offd = hypre_CSRMatrixNumCols(P_offd);

       fine_size = hypre_ParCSRMatrixGlobalNumRows(P_array[level]);
       coarse_size = hypre_ParCSRMatrixGlobalNumCols(P_array[level]);
       global_nonzeros = hypre_ParCSRMatrixNumNonzeros(P_array[level]);

       min_weight = 0.0;
       max_weight = 0.0;
       max_rowsum = 0.0;
       min_rowsum = 0.0;
       min_entries = 0;
       max_entries = 0;
 
  if (hypre_CSRMatrixNumNonzeros(P_diag))
  {
       min_weight = P_diag_data[0];
       for (j = P_diag_i[0]; j < P_diag_i[1]; j++)
       {
            min_weight = min(min_weight, P_diag_data[j]);
            if (P_diag_data[j] != 1.0)
                max_weight = max(max_weight, P_diag_data[j]);
            min_rowsum += P_diag_data[j];
       }
       for (j = P_offd_i[0]; j < P_offd_i[1]; j++)
       {        
             min_weight = min(min_weight, P_offd_data[j]); 
             if (P_offd_data[j] != 1.0)
                  max_weight = max(max_weight, P_offd_data[j]);     
             min_rowsum += P_offd_data[j];
       }

       max_rowsum = min_rowsum;

/*       min_entries = (P_diag_i[1]-P_diag_i[0])+(P_offd_i[1]-P_offd_i[0]); */
       min_entries = 2;
       max_entries = 0;

       for (j = 0; j < hypre_CSRMatrixNumRows(P_diag); j++)
       {
           entries = (P_diag_i[j+1]-P_diag_i[j])+(P_offd_i[j+1]-P_offd_i[j]);
           if (entries > 1)
              min_entries = min(entries, min_entries);
           max_entries = max(entries, max_entries);

           rowsum = 0.0;
           for (i = P_diag_i[j]; i < P_diag_i[j+1]; i++)
           {
               min_weight = min(min_weight, P_diag_data[i]);
               if (P_diag_data[i] != 1.0)
                     max_weight = max(max_weight, P_diag_data[i]);
               rowsum += P_diag_data[i];
           }

           for (i = P_offd_i[j]; i < P_offd_i[j+1]; i++)
           {
               min_weight = min(min_weight, P_offd_data[i]);
               if (P_offd_data[i] != 1.0) 
                     max_weight = max(max_weight, P_offd_data[i]);
               rowsum += P_offd_data[i];
           }
           min_rowsum = min(rowsum, min_rowsum);
           max_rowsum = max(rowsum, max_rowsum);
       }

  }
       avg_entries = ((double) global_nonzeros) / ((double) fine_size);

       send_buff[0] = (double) min_entries;
       send_buff[1] = (double) max_entries;
       send_buff[2] = min_rowsum;
       send_buff[3] = max_rowsum;
       send_buff[4] = min_weight;
       send_buff[5] = max_weight;

       MPI_Gather(send_buff,6,MPI_DOUBLE,gather_buff,6,MPI_DOUBLE,0,comm);

       if (my_id == 0)
       {
          global_min_e = (int) gather_buff[0];
          global_max_e = (int) gather_buff[1];
          global_min_rsum = gather_buff[2];
          global_max_rsum = gather_buff[3];
          global_min_wt = gather_buff[4];
          global_max_wt = gather_buff[5];
          
          for (j = 0; j < num_procs; j++)
          {
              global_min_e = min(global_min_e, (int) gather_buff[my_id*6]);
              global_max_e = max(global_max_e, (int) gather_buff[my_id*6+1]);
              global_min_rsum = min(global_min_rsum, gather_buff[my_id*6+2]);
              global_max_rsum = max(global_max_rsum, gather_buff[my_id*6+3]);
              global_min_wt = min(global_min_wt, gather_buff[my_id*6+4]);
              global_max_wt = max(global_max_wt, gather_buff[my_id*6+5]);
          }

          fprintf(fp, "%2d %5d x %-5d %3d %3d",
                level, fine_size, coarse_size,  global_min_e, global_max_e);
          fprintf(fp,"  %10.3e %9.3e %9.3e %9.3e\n",
                    global_min_wt, global_max_wt, 
                    global_min_rsum, global_max_rsum);
       }
   }
       

   
   fclose(fp);
   return(0);
}  




/*---------------------------------------------------------------
 * hypre_WriteParAMGSolverParams
 *---------------------------------------------------------------*/


void     hypre_WriteParAMGSolverParams(data)
void    *data;
 
{
   FILE    *fp;
   char    *file_name;
 
   hypre_ParAMGData  *amg_data = data;
 
   /* amg solve params */
   int      max_iter;
   int      cycle_type;    
   int     *num_grid_sweeps;  
   int     *grid_relax_type;   
   int    **grid_relax_points; 
   double   relax_weight;
   double   tol;
 
   /* amg output params */
   int      amg_ioutdat;
 
   int      j;
 
 
   /*----------------------------------------------------------
    * Get the amg_data data
    *----------------------------------------------------------*/
 
   file_name = hypre_ParAMGDataLogFileName(amg_data);
 
 
   max_iter   = hypre_ParAMGDataMaxIter(amg_data);
   cycle_type = hypre_ParAMGDataCycleType(amg_data);    
   num_grid_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data);  
   grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);
   grid_relax_points = hypre_ParAMGDataGridRelaxPoints(amg_data);
   relax_weight = hypre_ParAMGDataRelaxWeight(amg_data); 
   tol = hypre_ParAMGDataTol(amg_data);
 
   amg_ioutdat = hypre_ParAMGDataIOutDat(amg_data);
 
   /*----------------------------------------------------------
    * Open the output file
    *----------------------------------------------------------*/
 
   if (amg_ioutdat == 1 || amg_ioutdat == 3)
   { 
      fp = fopen(file_name, "a");
 
      fprintf(fp,"\n\nBoomerAMG SOLVER PARAMETERS:\n\n");
   
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
      fprintf(fp, "  Relaxation Weight (Jacobi) %f\n",relax_weight);

      fprintf(fp, " Output flag (ioutdat): %d \n", amg_ioutdat);
 
   /*----------------------------------------------------------
    * Close the output file
    *----------------------------------------------------------*/
 
      fclose(fp);
   }
 
   return;
}
