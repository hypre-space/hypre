/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "headers.h"
#include "amg.h"

/*****************************************************************************
 *
 * Routine for driving the setup phase of AMG
 *
 *****************************************************************************/

int
hypre_AMGSetup( void            *amg_vdata,
                hypre_CSRMatrix *A,
                hypre_Vector    *f,
                hypre_Vector    *u         )
{
   hypre_AMGData   *amg_data = amg_vdata;

   /* Data Structure variables */

   hypre_CSRMatrix **A_array;
   hypre_Vector    **F_array;
   hypre_Vector    **U_array;
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

   int       num_levels;
   int       level;
   int       coarse_size;
   int       fine_size;
   int       not_finished_coarsening = 1;
   int       Setup_err_flag;
   int       coarse_threshold = 9;
   int       j;
   int	     coarsen_type;

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

   coarsen_type = hypre_AMGDataCoarsenType(amg_data);

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

      if (coarsen_type == 1)
      {
	 hypre_AMGCoarsenRuge(A_array[level], strong_threshold,
                       &S, &CF_marker, &coarse_size); 
      }
      else
      {
         hypre_AMGCoarsen(A_array[level], strong_threshold,
                       &S, &CF_marker, &coarse_size); 
      }
      /* if no coarse-grid, stop coarsening */
      if (coarse_size == 0)
         break;

      CF_marker_array[level] = CF_marker;
      
      /*-------------------------------------------------------------
       * Build prolongation matrix, P, and place in P_array[level] 
       *--------------------------------------------------------------*/

      hypre_AMGBuildInterp(A_array[level], CF_marker_array[level], S, &P);
      P_array[level] = P; 
      hypre_DestroyCSRMatrix(S);
 
      /*-------------------------------------------------------------
       * Build coarse-grid operator, A_array[level+1] by R*A*P
       *--------------------------------------------------------------*/

      hypre_AMGBuildCoarseOperator(P_array[level], A_array[level] , 
                                   P_array[level], &A_H);

      ++level;
      A_array[level] = A_H;

      if (level+1 >= max_levels || 
          coarse_size == fine_size || 
          coarse_size <= coarse_threshold)
         not_finished_coarsening = 0;
   } 
   
   /*-----------------------------------------------------------------------
    * enter all the stuff created, A[level], P[level], CF_marker[level],
    * for levels 1 through coarsest, into amg_data data structure
    *-----------------------------------------------------------------------*/

   num_levels = level+1;
   hypre_AMGDataNumLevels(amg_data) = num_levels;
   hypre_AMGDataCFMarkerArray(amg_data) = CF_marker_array;
   hypre_AMGDataAArray(amg_data) = A_array;
   hypre_AMGDataPArray(amg_data) = P_array;

   /*-----------------------------------------------------------------------
    * Setup F and U arrays
    *-----------------------------------------------------------------------*/

   F_array = hypre_CTAlloc(hypre_Vector*, num_levels);
   U_array = hypre_CTAlloc(hypre_Vector*, num_levels);

   F_array[0] = f;
   U_array[0] = u;

   for (j = 1; j < num_levels; j++)
   {
      F_array[j] = hypre_CreateVector(hypre_CSRMatrixNumRows(A_array[j]));
      hypre_InitializeVector(F_array[j]);

      U_array[j] = hypre_CreateVector(hypre_CSRMatrixNumRows(A_array[j]));
      hypre_InitializeVector(U_array[j]);
   }

   hypre_AMGDataFArray(amg_data) = F_array;
   hypre_AMGDataUArray(amg_data) = U_array;

   /*-----------------------------------------------------------------------
    * Print some stuff
    *-----------------------------------------------------------------------*/

   if (amg_ioutdat == 1 || amg_ioutdat == 3)
      hypre_AMGSetupStats(amg_data);

   if (amg_ioutdat == -3)
   {  
      char     fnam[255];

      int j;

      for (j = 1; j < level+1; j++)
      {
         sprintf(fnam,"SP_A_%d.ysmp",j);
         hypre_PrintCSRMatrix(A_array[j],fnam);

      }                         

      for (j = 0; j < level; j++)
      { 
         sprintf(fnam,"SP_P_%d.ysmp",j);
         hypre_PrintCSRMatrix(P_array[j],fnam);
      }   
   } 
                     
   Setup_err_flag = 0;
   return(Setup_err_flag);
}  

