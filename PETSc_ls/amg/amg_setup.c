/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Routine for driving the setup phase of AMG
 *
 ******************************************************************************/

int hypre_AMGSetup(hypre_AMGData *amg_data)

{

   /*--------------------------------------------------------------------------
    * do some callocs, declare some variables, get some stuff out of the 
    * amg_data data structure, etc.
    *--------------------------------------------------------------------------/

   return(0);
#if 0

   level = 0;
   A[level] = hypre_AMGDataMatrixA(amg_data);
   strong_threshold = hypre_AMGDataThreshold(amg_data);

   while (not_finished_coarsening)
   {
         CF_marker = hypre_CTAlloc(int, hypre_CSRMatrixSize(A[level]));

         hypre_AMGCoarsen(A[level], CF_marker[level], strong_threshold);
      
              /*-------------------------------------------------------------
               *  here we feed in the matrix A of the current level, probably
               *  A[level] where *A is a pointer to an array of hypre_CSRMatrx,
               *  and we get back CF_pointer.  This routine calls
               *  
               *  hypre_AMGIndepSet(ST_data, ST_i, ST_j, num_variables,
               *                    measure_array, IS_array, IS_size) 
               *          
               *    I am working on hypre_AMGCoarsen and the routines it calls
               *--------------------------------------------------------------*/
           
         hypre_AMGBuildInterp(amg_data, A[level], CF_marker[level], P[level]);
            
              /*-------------------------------------------------------------
               * this routine builds the prolongation operator.  It will
               * return a hypre_CSRMatrix, P[level].
               *
               * Jim has volunteered to construct this next week
               *-------------------------------------------------------------*/
      

         hypre_AMGRap(amg_data, A[level], P[level], A[level+1]);

              /*-------------------------------------------------------------
               * this routine builds the  coarse_grid operator, A[level+1]
               * Jim has this essentially coded
               *-------------------------------------------------------------*/
      
         not_finished_coarsening = hypre_AMGShould_We_Stop_Coarsening(amg_data);

              /*-------------------------------------------------------------
               * the purpose of this routine (or test) is obvious, but we 
               * haven't yet discussed what the test should be
               *-------------------------------------------------------------*/
   } 
   
   /*-----------------------------------------------------------------------
    * enter all the stuff created, A[level], P[level], CF_marker[level],
    * for levels 1 through coarsest, into amg_data data structure
    *-----------------------------------------------------------------------*/

#endif

}  


