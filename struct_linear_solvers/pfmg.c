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
#include "pfmg.h"

/*--------------------------------------------------------------------------
 * hypre_PFMGInitialize
 *--------------------------------------------------------------------------*/

void *
hypre_PFMGInitialize( MPI_Comm  comm )
{
   hypre_PFMGData *pfmg_data;

   pfmg_data = hypre_CTAlloc(hypre_PFMGData, 1);

   (pfmg_data -> comm)       = comm;
   (pfmg_data -> time_index) = hypre_InitializeTiming("PFMG");

   /* set defaults */
   (pfmg_data -> tol)            = 1.0e-06;
   (pfmg_data -> max_iter)       = 200;
   (pfmg_data -> rel_change)     = 0;
   (pfmg_data -> zero_guess)     = 0;
   (pfmg_data -> max_levels)     = 0;
   (pfmg_data -> dxyz)[0]        = 0.0;
   (pfmg_data -> dxyz)[1]        = 0.0;
   (pfmg_data -> dxyz)[2]        = 0.0;
   (pfmg_data -> relax_type)     = 1;       /* weighted Jacobi */
   (pfmg_data -> num_pre_relax)  = 1;
   (pfmg_data -> num_post_relax) = 1;
   (pfmg_data -> skip_relax)     = 1;
   (pfmg_data -> logging)        = 0;

   /* initialize */
   (pfmg_data -> num_levels) = -1;

   return (void *) pfmg_data;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGFinalize
 *--------------------------------------------------------------------------*/

int
hypre_PFMGFinalize( void *pfmg_vdata )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;

   int l;
   int ierr = 0;

   if (pfmg_data)
   {
      if ((pfmg_data -> logging) > 0)
      {
         hypre_TFree(pfmg_data -> norms);
         hypre_TFree(pfmg_data -> rel_norms);
      }

      if ((pfmg_data -> num_levels) > -1)
      {
         for (l = 0; l < (pfmg_data -> num_levels); l++)
         {
            hypre_PFMGRelaxFinalize(pfmg_data -> relax_data_l[l]);
            hypre_StructMatvecFinalize(pfmg_data -> matvec_data_l[l]);
         }
         for (l = 0; l < ((pfmg_data -> num_levels) - 1); l++)
         {
            hypre_PFMGRestrictFinalize(pfmg_data -> restrict_data_l[l]);
            hypre_PFMGInterpFinalize(pfmg_data -> interp_data_l[l]);
         }
         hypre_TFree(pfmg_data -> relax_data_l);
         hypre_TFree(pfmg_data -> matvec_data_l);
         hypre_TFree(pfmg_data -> restrict_data_l);
         hypre_TFree(pfmg_data -> interp_data_l);
 
         hypre_FreeStructVector(pfmg_data -> tx_l[0]);
         hypre_FreeStructGrid(pfmg_data -> grid_l[0]);
         hypre_FreeStructMatrix(pfmg_data -> A_l[0]);
         hypre_FreeStructVector(pfmg_data -> b_l[0]);
         hypre_FreeStructVector(pfmg_data -> x_l[0]);
         for (l = 0; l < ((pfmg_data -> num_levels) - 1); l++)
         {
            hypre_FreeStructGrid(pfmg_data -> grid_l[l+1]);
            hypre_FreeStructGrid(pfmg_data -> P_grid_l[l+1]);
            hypre_FreeStructMatrix(pfmg_data -> A_l[l+1]);
            hypre_FreeStructMatrix(pfmg_data -> P_l[l]);
            hypre_FreeStructVector(pfmg_data -> b_l[l+1]);
            hypre_FreeStructVector(pfmg_data -> x_l[l+1]);
            hypre_FreeStructVector(pfmg_data -> tx_l[l+1]);
         }
         hypre_SharedTFree(pfmg_data -> data);
         hypre_TFree(pfmg_data -> cdir_l);
         hypre_TFree(pfmg_data -> active_l);
         hypre_TFree(pfmg_data -> grid_l);
         hypre_TFree(pfmg_data -> P_grid_l);
         hypre_TFree(pfmg_data -> A_l);
         hypre_TFree(pfmg_data -> P_l);
         hypre_TFree(pfmg_data -> RT_l);
         hypre_TFree(pfmg_data -> b_l);
         hypre_TFree(pfmg_data -> x_l);
         hypre_TFree(pfmg_data -> tx_l);
      }
 
      hypre_FinalizeTiming(pfmg_data -> time_index);
      hypre_TFree(pfmg_data);
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetTol
 *--------------------------------------------------------------------------*/

int
hypre_PFMGSetTol( void   *pfmg_vdata,
                  double  tol       )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   int             ierr = 0;
 
   (pfmg_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetMaxIter
 *--------------------------------------------------------------------------*/

int
hypre_PFMGSetMaxIter( void *pfmg_vdata,
                      int   max_iter  )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   int             ierr = 0;
 
   (pfmg_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetRelChange
 *--------------------------------------------------------------------------*/

int
hypre_PFMGSetRelChange( void *pfmg_vdata,
                        int   rel_change  )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   int             ierr = 0;
 
   (pfmg_data -> rel_change) = rel_change;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
hypre_PFMGSetZeroGuess( void *pfmg_vdata,
                        int   zero_guess )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   int             ierr = 0;
 
   (pfmg_data -> zero_guess) = zero_guess;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetRelaxType
 *--------------------------------------------------------------------------*/

int
hypre_PFMGSetRelaxType( void *pfmg_vdata,
                        int   relax_type )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   int             ierr = 0;
 
   (pfmg_data -> relax_type) = relax_type;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetNumPreRelax
 *--------------------------------------------------------------------------*/

int
hypre_PFMGSetNumPreRelax( void *pfmg_vdata,
                          int   num_pre_relax )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   int             ierr = 0;
 
   (pfmg_data -> num_pre_relax) = num_pre_relax;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

int
hypre_PFMGSetNumPostRelax( void *pfmg_vdata,
                           int   num_post_relax )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   int             ierr = 0;
 
   (pfmg_data -> num_post_relax) = num_post_relax;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetNumSkipRelax
 *--------------------------------------------------------------------------*/

int
hypre_PFMGSetSkipRelax( void *pfmg_vdata,
                        int  skip_relax )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   int             ierr = 0;
 
   (pfmg_data -> skip_relax) = skip_relax;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetDxyz
 *--------------------------------------------------------------------------*/

int
hypre_PFMGSetDxyz( void   *pfmg_vdata,
                   double *dxyz       )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   int             ierr = 0;

   (pfmg_data -> dxyz[0]) = dxyz[0];
   (pfmg_data -> dxyz[1]) = dxyz[1];
   (pfmg_data -> dxyz[2]) = dxyz[2];
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetLogging
 *--------------------------------------------------------------------------*/

int
hypre_PFMGSetLogging( void *pfmg_vdata,
                      int   logging)
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   int             ierr = 0;
 
   (pfmg_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGGetNumIterations
 *--------------------------------------------------------------------------*/

int
hypre_PFMGGetNumIterations( void *pfmg_vdata,
                            int  *num_iterations )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   int             ierr = 0;

   *num_iterations = (pfmg_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGPrintLogging
 *--------------------------------------------------------------------------*/

int
hypre_PFMGPrintLogging( void *pfmg_vdata,
                        int   myid)
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   int             ierr = 0;
   int             i;
   int             num_iterations  = (pfmg_data -> num_iterations);
   int             logging   = (pfmg_data -> logging);
   double         *norms     = (pfmg_data -> norms);
   double         *rel_norms = (pfmg_data -> rel_norms);

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
 * hypre_PFMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
hypre_PFMGGetFinalRelativeResidualNorm( void   *pfmg_vdata,
                                        double *relative_residual_norm )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;

   int             max_iter        = (pfmg_data -> max_iter);
   int             num_iterations  = (pfmg_data -> num_iterations);
   int             logging         = (pfmg_data -> logging);
   double         *rel_norms       = (pfmg_data -> rel_norms);
            
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


