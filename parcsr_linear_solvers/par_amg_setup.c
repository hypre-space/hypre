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
   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(A); 

   hypre_ParAMGData   *amg_data = amg_vdata;

   /* Data Structure variables */

   hypre_ParCSRMatrix **A_array;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;
   hypre_ParVector     *Vtemp;
   hypre_ParCSRMatrix **P_array;
   int                **CF_marker_array;   
   double              *relax_weight;
   double               strong_threshold;
   double               trunc_factor;

   int      num_variables;
   int      max_levels; 
   int      amg_ioutdat;
   int      debug_flag;
 
   /* Local variables */
   int                 *CF_marker;
   hypre_ParCSRMatrix  *S;
   hypre_ParCSRMatrix  *P;
   hypre_ParCSRMatrix  *A_H;

   int       old_num_levels, num_levels;
   int       level;
   int       coarse_size;
   int       coarsen_type;
   int       measure_type;
   int       fine_size;
   double    size;
   int       not_finished_coarsening = 1;
   int       Setup_err_flag;
   int       coarse_threshold = 9;
   int       j;
   int       num_procs,my_id;

   double    wall_time;   /* for debugging instrumentation */

   MPI_Comm_size(comm, &num_procs);   
   MPI_Comm_rank(comm,&my_id);

   old_num_levels = hypre_ParAMGDataNumLevels(amg_data);
   max_levels = hypre_ParAMGDataMaxLevels(amg_data);
   amg_ioutdat = hypre_ParAMGDataIOutDat(amg_data);
   coarsen_type = hypre_ParAMGDataCoarsenType(amg_data);
   measure_type = hypre_ParAMGDataMeasureType(amg_data);
   debug_flag = hypre_ParAMGDataDebugFlag(amg_data);
   relax_weight = hypre_ParAMGDataRelaxWeight(amg_data);
   
   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   CF_marker_array = hypre_ParAMGDataCFMarkerArray(amg_data);

   if (A_array != NULL || P_array != NULL || CF_marker_array != NULL)
   {
      for (j = 1; j < old_num_levels; j++)
      {
        if (A_array[j] != NULL)
        {
           hypre_ParCSRMatrixDestroy(A_array[j]);
           A_array[j] = NULL;
        }
      }

      for (j = 0; j < old_num_levels-1; j++)
      {
         if (P_array[j] != NULL)
         {
            hypre_ParCSRMatrixDestroy(P_array[j]);
            P_array[j] = NULL;
         }

         if (CF_marker_array[j] != NULL)
            hypre_TFree(CF_marker_array[j]);
      }
   }

   if (A_array == NULL)
      A_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_levels);
   if (P_array == NULL)
      P_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_levels-1);
   if (CF_marker_array == NULL)
      CF_marker_array = hypre_CTAlloc(int*, max_levels-1);

   hypre_ParCSRMatrixSetNumNonzeros(A);
   A_array[0] = A;
   /*----------------------------------------------------------
    * Initialize hypre_ParAMGData
    *----------------------------------------------------------*/

   num_variables = hypre_ParCSRMatrixNumRows(A);

   hypre_ParAMGDataNumVariables(amg_data) = num_variables;

   not_finished_coarsening = 1;
   level = 0;
  
   strong_threshold = hypre_ParAMGDataStrongThreshold(amg_data);
   trunc_factor = hypre_ParAMGDataTruncFactor(amg_data);

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
     
      if (debug_flag==1) wall_time = time_getWallclockSeconds();
      if (debug_flag==3)
      {
          printf("\n ===== Proc = %d     Level = %d  =====\n",
                        my_id, level);
          fflush(NULL);
      }
      if (relax_weight[level] == 0.0)
      {
	 hypre_ParCSRMatrixScaledNorm(A_array[level], &relax_weight[level]);
	 if (relax_weight[level] != 0.0)
	    relax_weight[level] = (4.0/3.0)/relax_weight[level];
	 else
	   printf (" Warning ! Matrix norm is zero !!!");
      }
      if (coarsen_type == 6)
      {
	 hypre_ParAMGCoarsenFalgout(A_array[level], strong_threshold,
                                    debug_flag, &S, &CF_marker, &coarse_size); 
      }
      else if (coarsen_type)
      {
	 hypre_ParAMGCoarsenRuge(A_array[level], strong_threshold,
                                 measure_type, coarsen_type, debug_flag,
                                 &S, &CF_marker, &coarse_size); 
      }
      else
      {
	 hypre_ParAMGCoarsen(A_array[level], strong_threshold,
                             debug_flag, &S, &CF_marker, &coarse_size); 
      }
 
      if (debug_flag==1)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d    Level = %d    Coarsen Time = %f\n",
                       my_id,level, wall_time); 
	 fflush(NULL);
      }

      CF_marker_array[level] = CF_marker;
      
      /*-------------------------------------------------------------
       * Build prolongation matrix, P, and place in P_array[level] 
       *--------------------------------------------------------------*/
      if (debug_flag==1) wall_time = time_getWallclockSeconds();

      hypre_ParAMGBuildInterp(A_array[level], CF_marker_array[level], S,
                              debug_flag, trunc_factor, &P);

      if (debug_flag==1)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d    Level = %d    Build Interp Time = %f\n",
                       my_id,level, wall_time);
	 fflush(NULL);
      }

      P_array[level] = P; 
      hypre_ParCSRMatrixDestroy(S);
      S = NULL;
      coarse_size = hypre_ParCSRMatrixGlobalNumCols(P);

      /* if no coarse-grid, stop coarsening */
      if (coarse_size == 0)
         break; 

      /*-------------------------------------------------------------
       * Build coarse-grid operator, A_array[level+1] by R*A*P
       *--------------------------------------------------------------*/

      if (debug_flag==1) wall_time = time_getWallclockSeconds();

      hypre_ParAMGBuildCoarseOperator(P_array[level], A_array[level] , 
                                      P_array[level], &A_H);

      if (debug_flag==1)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d    Level = %d    Build Coarse Operator Time = %f\n",
                       my_id,level, wall_time);
	 fflush(NULL);
      }

      ++level;
      hypre_ParCSRMatrixSetNumNonzeros(A_H);
      A_array[level] = A_H;

      size = ((double) fine_size )*.75;
      if (coarsen_type > 0 && coarse_size >= (int) size)
      {
	coarsen_type = 0;      
      }

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
   if (coarse_size == fine_size) num_levels = level;
   hypre_ParAMGDataNumLevels(amg_data) = num_levels;
   hypre_ParAMGDataCFMarkerArray(amg_data) = CF_marker_array;
   hypre_ParAMGDataAArray(amg_data) = A_array;
   hypre_ParAMGDataPArray(amg_data) = P_array;
   hypre_ParAMGDataRArray(amg_data) = P_array;

   /*-----------------------------------------------------------------------
    * Setup Vtemp, F and U arrays
    *-----------------------------------------------------------------------*/

   Vtemp = hypre_ParAMGDataVtemp(amg_data);

   if (Vtemp != NULL)
   {
      hypre_ParVectorDestroy(Vtemp);
      Vtemp = NULL;
   }

   Vtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                 hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                 hypre_ParCSRMatrixRowStarts(A_array[0]));
   hypre_ParVectorInitialize(Vtemp);
   hypre_ParVectorSetPartitioningOwner(Vtemp,0);
   hypre_ParAMGDataVtemp(amg_data) = Vtemp;

   F_array = hypre_ParAMGDataFArray(amg_data);
   U_array = hypre_ParAMGDataUArray(amg_data);

   if (F_array != NULL || U_array != NULL)
   {
      for (j = 1; j < old_num_levels; j++)
      {
         if (F_array[j] != NULL)
         {
            hypre_ParVectorDestroy(F_array[j]);
            F_array[j] = NULL;
         }
         if (U_array[j] != NULL)
         {
            hypre_ParVectorDestroy(U_array[j]);
            U_array[j] = NULL;
         }
      }
   }

   if (F_array == NULL)
      F_array = hypre_CTAlloc(hypre_ParVector*, max_levels);
   if (U_array == NULL)
      U_array = hypre_CTAlloc(hypre_ParVector*, max_levels);

   F_array[0] = f;
   U_array[0] = u;

   for (j = 1; j < num_levels; j++)
   {
      F_array[j] =
         hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[j]),
                               hypre_ParCSRMatrixGlobalNumRows(A_array[j]),
                               hypre_ParCSRMatrixRowStarts(A_array[j]));
      hypre_ParVectorInitialize(F_array[j]);
      hypre_ParVectorSetPartitioningOwner(F_array[j],0);

      U_array[j] =
         hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[j]),
                               hypre_ParCSRMatrixGlobalNumRows(A_array[j]),
                               hypre_ParCSRMatrixRowStarts(A_array[j]));
      hypre_ParVectorInitialize(U_array[j]);
      hypre_ParVectorSetPartitioningOwner(U_array[j],0);
   }

   hypre_ParAMGDataFArray(amg_data) = F_array;
   hypre_ParAMGDataUArray(amg_data) = U_array;

   /*-----------------------------------------------------------------------
    * Print some stuff
    *-----------------------------------------------------------------------*/

   if (amg_ioutdat == 1 || amg_ioutdat == 3)
      hypre_ParAMGSetupStats(amg_data,A);

#if 0 /* add later */
   if (amg_ioutdat == -3)
   {  
      char     fnam[255];

      int j;

      for (j = 1; j < level+1; j++)
      {
         sprintf(fnam,"SP_A_%d.ysmp",j);
         hypre_ParCSRMatrixPrint(A_array[j],fnam);
      }                         

      for (j = 0; j < level; j++)
      { 
         sprintf(fnam,"SP_P_%d.ysmp",j);
         hypre_ParCSRMatrixPrint(P_array[j],fnam);
      }   
   } 
#endif

   Setup_err_flag = 0;
   return(Setup_err_flag);
}  

