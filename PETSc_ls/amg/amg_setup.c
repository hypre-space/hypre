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
 * Routine for driving the setup phase of AMG
 *
 ******************************************************************************/


int hypre_AMGSetup(hypre_AMGData *amg_data,
                   hypre_CSRMatrix *A)

{

   /* Data Structure variables */

   hypre_CSRMatrix **A_array;
   hypre_CSRMatrix **P_array;
   int             **CF_marker_array;   
   double            strong_threshold;

   int      num_variables;
   int      max_levels; 
   int      amg_ioutdat;
 
   /* Local variables */
   int              *CF_marker;
   hypre_CSRMatrix  *S;
   hypre_CSRMatrix  *P;
   hypre_CSRMatrix  *A_H;


   int       level;
   int       coarse_size;
   int       fine_size;
   int       not_finished_coarsening = 1;

   int       Setup_err_flag;

   int       coarse_threshold = 9;

   max_levels = hypre_AMGDataMaxLevels(amg_data);
   amg_ioutdat = hypre_AMGDataIOutDat(amg_data);
   
   A_array = hypre_CTAlloc(hypre_CSRMatrix*, max_levels);
   P_array = hypre_CTAlloc(hypre_CSRMatrix*, max_levels-1);
   CF_marker_array = hypre_CTAlloc(int*, max_levels-1);

   A_array[0] = A;

   /*----------------------------------------------------------
    * Initialize hypre_AMGData
    *----------------------------------------------------------*/

   num_variables = hypre_CSRMatrixNumRows(A);


   hypre_AMGDataNumVariables(amg_data) = num_variables;

   not_finished_coarsening = 1;
   level = 0;
  
   strong_threshold = hypre_AMGDataStrongThreshold(amg_data);

   /*-----------------------------------------------------
    *  Enter Coarsening Loop
    *-----------------------------------------------------*/

   while (not_finished_coarsening)
   {
         fine_size = hypre_CSRMatrixNumRows(A_array[level]);

         /*-------------------------------------------------------------
          * Select coarse-grid points on 'level' : returns CF_marker
          * for the level.  Returns strength matrix, S  
          *--------------------------------------------------------------*/

         hypre_AMGCoarsen(A_array[level], strong_threshold, &CF_marker, &S); 
         CF_marker_array[level] = CF_marker;
      
         /*-------------------------------------------------------------
          * Build prolongation matrix, P, and place in P_array[level] 
          *--------------------------------------------------------------*/


         hypre_AMGBuildInterp(A_array[level], CF_marker_array[level], S, &P);
         P_array[level] = P; 
   
         /*-------------------------------------------------------------
          * Build coarse-grid operator, A_array[level+1] by R*A*P
          *--------------------------------------------------------------*/

         hypre_AMGBuildCoarseOperator(P_array[level], A_array[level] , 
                                     P_array[level], &A_H);

         ++level;
         A_array[level] = A_H;
         coarse_size = hypre_CSRMatrixNumRows(A_array[level]);

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

   hypre_AMGDataNumLevels(amg_data) = level+1;
   hypre_AMGDataCFMarkerArray(amg_data) = CF_marker_array;
   hypre_AMGDataAArray(amg_data) = A_array;
   hypre_AMGDataPArray(amg_data) = P_array;

   if (amg_ioutdat == 1 || amg_ioutdat == 3)
                     hypre_AMGSetupStats(amg_data);

   Setup_err_flag = 0;
   return(Setup_err_flag);
}  


