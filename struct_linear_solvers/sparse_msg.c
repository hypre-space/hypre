/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"
#include "sparse_msg.h"

/*--------------------------------------------------------------------------
 * hypre_SparseMSGCreate
 *--------------------------------------------------------------------------*/

void *
hypre_SparseMSGCreate( MPI_Comm  comm )
{
   hypre_SparseMSGData *SparseMSG_data;

   SparseMSG_data = hypre_CTAlloc(hypre_SparseMSGData, 1);

   (SparseMSG_data -> comm)       = comm;
   (SparseMSG_data -> time_index) = hypre_InitializeTiming("SparseMSG");

   /* set defaults */
   (SparseMSG_data -> tol)            = 1.0e-06;
   (SparseMSG_data -> max_iter)       = 200;
   (SparseMSG_data -> rel_change)     = 0;
   (SparseMSG_data -> zero_guess)     = 0;
   (SparseMSG_data -> relax_type)     = 1;       /* weighted Jacobi */
   (SparseMSG_data -> num_pre_relax)  = 1;
   (SparseMSG_data -> num_post_relax) = 1;
   (SparseMSG_data -> logging)        = 0;

   /* initialize */
   (SparseMSG_data -> num_levels[0])    = 1;
   (SparseMSG_data -> num_levels[1])    = 1;
   (SparseMSG_data -> num_levels[2])    = 1;

   return (void *) SparseMSG_data;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGDestroy
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGDestroy( void *SparseMSG_vdata )
{
   hypre_SparseMSGData *SparseMSG_data = SparseMSG_vdata;

   int l;
   int ierr = 0;

   if (SparseMSG_data)
   {
      if ((SparseMSG_data -> logging) > 0)
      {
         hypre_TFree(SparseMSG_data -> norms);
         hypre_TFree(SparseMSG_data -> rel_norms);
      }

      if ((SparseMSG_data -> total_num_levels) > 1)
      {
         for (l = 0; l < SparseMSG_data -> num_grids; l++)
         {
            hypre_PFMGRelaxDestroy(SparseMSG_data -> relax_data_array[l]);
            hypre_StructMatvecDestroy(SparseMSG_data -> matvec_data_array[l]);
         }

         for (l = 0; l < (3*(SparseMSG_data -> num_grids)); l++)
         {
            hypre_PFMGRestrictDestroy(SparseMSG_data ->
                                      restrict_data_array[l]);
            hypre_PFMGInterpDestroy(SparseMSG_data -> interp_data_array[l]);
         }

         hypre_TFree(SparseMSG_data -> relax_data_array);
         hypre_TFree(SparseMSG_data -> matvec_data_array);
         hypre_TFree(SparseMSG_data -> restrict_data_array);
         hypre_TFree(SparseMSG_data -> interp_data_array);
         hypre_TFree(SparseMSG_data -> restrict_weights);
         hypre_TFree(SparseMSG_data -> interp_weights);
         
         for (l = 0; l < SparseMSG_data -> num_grids; l++)
         {
            hypre_DestroyStructGrid(SparseMSG_data -> grid_array[l]);
            hypre_DestroyStructMatrix(SparseMSG_data -> A_array[l]);
            hypre_DestroyStructVector(SparseMSG_data -> b_array[l]);
            hypre_DestroyStructVector(SparseMSG_data -> x_array[l]);
            hypre_DestroyStructVector(SparseMSG_data -> tx_array[l]);
            hypre_DestroyStructVector(SparseMSG_data -> r_array[l]);
         }

         for (l = 0; l < (3*(SparseMSG_data -> num_grids)); l++)
         {
            hypre_DestroyStructMatrix(SparseMSG_data -> P_array[l]);
            hypre_DestroyStructGrid(SparseMSG_data -> P_grid_array[l]); 
         }

         hypre_TFree(SparseMSG_data -> grid_array);
         hypre_TFree(SparseMSG_data -> P_grid_array);
         hypre_TFree(SparseMSG_data -> A_array);
         hypre_TFree(SparseMSG_data -> P_array);
         hypre_TFree(SparseMSG_data -> RT_array);
         hypre_TFree(SparseMSG_data -> b_array);
         hypre_TFree(SparseMSG_data -> x_array);
         hypre_TFree(SparseMSG_data -> tx_array);
         hypre_TFree(SparseMSG_data -> r_array);
      }
 
      hypre_FinalizeTiming(SparseMSG_data -> time_index);
      hypre_TFree(SparseMSG_data);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetTol
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetTol( void   *SparseMSG_vdata,
                       double  tol       )
{
   hypre_SparseMSGData *SparseMSG_data = SparseMSG_vdata;
   int             ierr = 0;
 
   (SparseMSG_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetMaxIter
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetMaxIter( void *SparseMSG_vdata,
                           int   max_iter  )
{
   hypre_SparseMSGData *SparseMSG_data = SparseMSG_vdata;
   int             ierr = 0;
 
   (SparseMSG_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetJump
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetJump(  void *SparseMSG_vdata,
                         int   jump   )

{
   hypre_SparseMSGData *SparseMSG_data = SparseMSG_vdata;
   int                  ierr = 0;

   (SparseMSG_data -> jump) = jump;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetRelChange
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetRelChange( void *SparseMSG_vdata,
                             int   rel_change  )
{
   hypre_SparseMSGData *SparseMSG_data = SparseMSG_vdata;
   int             ierr = 0;
 
   (SparseMSG_data -> rel_change) = rel_change;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
hypre_SparseMSGSetZeroGuess( void *SparseMSG_vdata,
                             int   zero_guess )
{
   hypre_SparseMSGData *SparseMSG_data = SparseMSG_vdata;
   int             ierr = 0;
 
   (SparseMSG_data -> zero_guess) = zero_guess;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetRelaxType
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetRelaxType( void *SparseMSG_vdata,
                             int   relax_type )
{
   hypre_SparseMSGData *SparseMSG_data = SparseMSG_vdata;
   int             ierr = 0;
 
   (SparseMSG_data -> relax_type) = relax_type;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetNumPreRelax
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetNumPreRelax( void *SparseMSG_vdata,
                               int   num_pre_relax )
{
   hypre_SparseMSGData *SparseMSG_data = SparseMSG_vdata;
   int             ierr = 0;
 
   (SparseMSG_data -> num_pre_relax) = num_pre_relax;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetNumPostRelax
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetNumPostRelax( void *SparseMSG_vdata,
                                int   num_post_relax )
{
   hypre_SparseMSGData *SparseMSG_data = SparseMSG_vdata;
   int             ierr = 0;
 
   (SparseMSG_data -> num_post_relax) = num_post_relax;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetLogging
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetLogging( void *SparseMSG_vdata,
                           int   logging)
{
   hypre_SparseMSGData *SparseMSG_data = SparseMSG_vdata;
   int             ierr = 0;
 
   (SparseMSG_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGGetNumIterations
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGGetNumIterations( void *SparseMSG_vdata,
                                 int  *num_iterations )
{
   hypre_SparseMSGData *SparseMSG_data = SparseMSG_vdata;
   int             ierr = 0;

   *num_iterations = (SparseMSG_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGPrintLogging
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGPrintLogging( void *SparseMSG_vdata,
                             int   myid)
{
   hypre_SparseMSGData *SparseMSG_data = SparseMSG_vdata;
   int             ierr = 0;
   int             i;
   int             num_iterations  = (SparseMSG_data -> num_iterations);
   int             logging   = (SparseMSG_data -> logging);
   double         *norms     = (SparseMSG_data -> norms);
   double         *rel_norms = (SparseMSG_data -> rel_norms);

   if (myid == 0)
   {
      if (logging > 0)
      {
         for (i = 0; i < num_iterations; i++)
         {
            printf("Residual norm[%d] = %e   ",i,norms[i]);
            printf("Relative residual norm[%d] = %e\n",i,rel_norms[i]);
         }
      }
   }
  
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGGetFinalRelativeResidualNorm( void   *SparseMSG_vdata,
                                             double *relative_residual_norm )
{
   hypre_SparseMSGData *SparseMSG_data = SparseMSG_vdata;

   int             max_iter        = (SparseMSG_data -> max_iter);
   int             num_iterations  = (SparseMSG_data -> num_iterations);
   int             logging         = (SparseMSG_data -> logging);
   double         *rel_norms       = (SparseMSG_data -> rel_norms);
            
   int             ierr = 0;

   
   if (logging > 0)
   {
      if (max_iter == 0)
      {
         ierr = 1;
      }
      else if (num_iterations == max_iter)
      {
         *relative_residual_norm = rel_norms[num_iterations-1];
      }
      else
      {
         *relative_residual_norm = rel_norms[num_iterations];
      }
   }
   
   return ierr;
}


