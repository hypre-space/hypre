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
 * Routine for driving the setup phase of AMG
 *
 *****************************************************************************/

int
hypre_ParAMGSetup(void               *vamg_data,
                  hypre_ParCSRMatrix *A         )

{
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData *) vamg_data;

   /* Data Structure variables */

   hypre_ParCSRMatrix **A_array;
   hypre_ParCSRMatrix **P_array;
   int             **CF_marker_array;   
   double            strong_threshold;

   int      global_num_variables;
   int      max_levels; 
   int      amg_ioutdat;
 
   /* Local variables */
   int                 *CF_marker;
   int                 *coarse_partitioning;
   hypre_ParCSRMatrix  *S;
   hypre_ParCSRMatrix  *P;
   hypre_ParCSRMatrix  *A_H;


   int       level;
   int       coarse_size;
   int       fine_size;
   int       not_finished_coarsening = 1;

   int       Setup_err_flag;

   int       coarse_threshold = 9;

   max_levels = hypre_ParAMGDataMaxLevels(amg_data);
   amg_ioutdat = hypre_ParAMGDataIOutDat(amg_data);
   
   A_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_levels);
   P_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_levels-1);
   CF_marker_array = hypre_CTAlloc(int*, max_levels-1);

   A_array[0] = A;

   /*----------------------------------------------------------
    * Initialize hypre_AMGData
    *----------------------------------------------------------*/

   global_num_variables = hypre_ParCSRMatrixGlobalNumRows(A);


   hypre_ParAMGDataNumVariables(amg_data) = global_num_variables;

   not_finished_coarsening = 1;
   level = 0;
  
   strong_threshold = hypre_ParAMGDataStrongThreshold(amg_data);

   /*-----------------------------------------------------
    *  Enter Coarsening Loop
    *-----------------------------------------------------*/

   while (not_finished_coarsening)
   {
         fine_size = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);

         /*-------------------------------------------------------------
          * Select coarse-grid points on 'level' : returns CF_marker
          * for the level.  Returns strength matrix, S  
          *--------------------------------------------------------------*/

         hypre_ParAMGCoarsen(A_array[level],strong_threshold,&CF_marker,&S); 
         CF_marker_array[level] = CF_marker;

         /*-------------------------------------------------------------
          * Build prolongation matrix, P, and place in P_array[level] 
          *--------------------------------------------------------------*/


         hypre_ParAMGBuildInterp(A_array[level],CF_marker_array[level],
                                 S,&P,&coarse_partitioning);
         P_array[level] = P; 
      
         /*-------------------------------------------------------------
          * Build coarse-grid operator, A_array[level+1] by R*A*P
          *--------------------------------------------------------------*/

         hypre_ParAMGBuildCoarseOperator(P_array[level], A_array[level] , 
                                         P_array[level], &A_H, 
                                         coarse_partitioning);

         ++level;
         A_array[level] = A_H;
         coarse_size = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);

         if (coarse_size <= 0)
         {
            --level;
            break;
         }

         if (level+1 >= max_levels || 
                     coarse_size == fine_size || 
                              coarse_size <= coarse_threshold)
                                     not_finished_coarsening = 0;
   } 
   
   /*-----------------------------------------------------------------------
    * enter all the stuff created, A[level], P[level], CF_marker[level],
    * for levels 1 through coarsest, into amg_data data structure
    *-----------------------------------------------------------------------*/

   hypre_ParAMGDataNumLevels(amg_data) = level+1;
   hypre_ParAMGDataCFMarkerArray(amg_data) = CF_marker_array;
   hypre_ParAMGDataAArray(amg_data) = A_array;
   hypre_ParAMGDataPArray(amg_data) = P_array;

/*   if (amg_ioutdat == 1 || amg_ioutdat == 3)
                     hypre_ParAMGSetupStats(amg_data);  */
                     
   Setup_err_flag = 0;
   return(Setup_err_flag);
}  


