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
hypre_ParAMGSetup( void               *amg_vdata,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector    *f,
                   hypre_ParVector    *u         )
{
   hypre_ParAMGData   *amg_data = amg_vdata;

   /* Data Structure variables */

   hypre_ParCSRMatrix **A_array;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;
   hypre_ParCSRMatrix **P_array;
   int                **CF_marker_array;   
   double               strong_threshold;

   int      num_variables;
   int      max_levels; 
   int      amg_ioutdat;
 
   /* Local variables */
   int                 *CF_marker;
   hypre_ParCSRMatrix  *S;
   hypre_ParCSRMatrix  *P;
   hypre_ParCSRMatrix  *A_H;

   int       num_levels;
   int       level;
   int       coarse_size;
   int       fine_size;
   int       not_finished_coarsening = 1;
   int       Setup_err_flag;
   int       coarse_threshold = 9;
   int       j;

   max_levels = hypre_ParAMGDataMaxLevels(amg_data);
   amg_ioutdat = hypre_ParAMGDataIOutDat(amg_data);
   
   A_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_levels);
   P_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_levels-1);
   CF_marker_array = hypre_CTAlloc(int*, max_levels-1);

   hypre_SetParCSRMatrixNumNonzeros(A);
   A_array[0] = A;
   /*----------------------------------------------------------
    * Initialize hypre_ParAMGData
    *----------------------------------------------------------*/

   num_variables = hypre_ParCSRMatrixNumRows(A);

   hypre_ParAMGDataNumVariables(amg_data) = num_variables;

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

      hypre_ParAMGCoarsen(A_array[level], strong_threshold,
                          &S, &CF_marker, &coarse_size); 

      /* if no coarse-grid, stop coarsening */
      if (coarse_size == 0)
         break;

      CF_marker_array[level] = CF_marker;
      
      /*-------------------------------------------------------------
       * Build prolongation matrix, P, and place in P_array[level] 
       *--------------------------------------------------------------*/

      hypre_ParAMGBuildInterp(A_array[level], CF_marker_array[level], S, &P);
      P_array[level] = P; 
      hypre_DestroyParCSRMatrix(S);
      coarse_size = hypre_ParCSRMatrixGlobalNumCols(P);
      /*-------------------------------------------------------------
       * Build coarse-grid operator, A_array[level+1] by R*A*P
       *--------------------------------------------------------------*/

      hypre_ParAMGBuildCoarseOperator(P_array[level], A_array[level] , 
                                      P_array[level], &A_H);

      ++level;
      hypre_SetParCSRMatrixNumNonzeros(A_H);
      A_array[level] = A_H;

      if ( (level+1 >= max_levels) || 
           (coarse_size == fine_size) || 
           (coarse_size <= coarse_threshold) )
      {
         not_finished_coarsening = 0;
      }
   } 
   
   /*-----------------------------------------------------------------------
    * enter all the stuff created, A[level], P[level], CF_marker[level],
    * for levels 1 through coarsest, into amg_data data structure
    *-----------------------------------------------------------------------*/

   num_levels = level+1;
   hypre_ParAMGDataNumLevels(amg_data) = num_levels;
   hypre_ParAMGDataCFMarkerArray(amg_data) = CF_marker_array;
   hypre_ParAMGDataAArray(amg_data) = A_array;
   hypre_ParAMGDataPArray(amg_data) = P_array;

   /*-----------------------------------------------------------------------
    * Setup F and U arrays
    *-----------------------------------------------------------------------*/

   F_array = hypre_CTAlloc(hypre_ParVector*, num_levels);
   U_array = hypre_CTAlloc(hypre_ParVector*, num_levels);

   F_array[0] = f;
   U_array[0] = u;

   for (j = 1; j < num_levels; j++)
   {
      F_array[j] =
         hypre_CreateParVector(hypre_ParCSRMatrixComm(A_array[j]),
                               hypre_ParCSRMatrixGlobalNumRows(A_array[j]),
                               hypre_ParCSRMatrixRowStarts(A_array[j]));
      hypre_InitializeParVector(F_array[j]);
      hypre_SetParVectorPartitioningOwner(F_array[j],0);

      U_array[j] =
         hypre_CreateParVector(hypre_ParCSRMatrixComm(A_array[j]),
                               hypre_ParCSRMatrixGlobalNumRows(A_array[j]),
                               hypre_ParCSRMatrixRowStarts(A_array[j]));
      hypre_InitializeParVector(U_array[j]);
      hypre_SetParVectorPartitioningOwner(U_array[j],0);
   }

   hypre_ParAMGDataFArray(amg_data) = F_array;
   hypre_ParAMGDataUArray(amg_data) = U_array;

#if 0 /* add later */
   /*-----------------------------------------------------------------------
    * Print some stuff
    *-----------------------------------------------------------------------*/

   if (amg_ioutdat == 1 || amg_ioutdat == 3)
      hypre_ParAMGSetupStats(amg_data);

   if (amg_ioutdat == -3)
   {  
      char     fnam[255];

      int j;

      for (j = 1; j < level+1; j++)
      {
         sprintf(fnam,"SP_A_%d.ysmp",j);
         hypre_PrintParCSRMatrix(A_array[j],fnam);
      }                         

      for (j = 0; j < level; j++)
      { 
         sprintf(fnam,"SP_P_%d.ysmp",j);
         hypre_PrintParCSRMatrix(P_array[j],fnam);
      }   
   } 
#endif

   Setup_err_flag = 0;
   return(Setup_err_flag);
}  

