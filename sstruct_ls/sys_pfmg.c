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
#include "sys_pfmg.h"

/*--------------------------------------------------------------------------
 * hypre_SysPFMGCreate
 *--------------------------------------------------------------------------*/

void *
hypre_SysPFMGCreate( MPI_Comm  comm )
{
   hypre_SysPFMGData *sys_pfmg_data;

   sys_pfmg_data = hypre_CTAlloc(hypre_SysPFMGData, 1);

   (sys_pfmg_data -> comm)       = comm;
   (sys_pfmg_data -> time_index) = hypre_InitializeTiming("SYS_PFMG");

   /* set defaults */
   (sys_pfmg_data -> tol)            = 1.0e-06;
   (sys_pfmg_data -> max_iter)       = 200;
   (sys_pfmg_data -> rel_change)     = 0;
   (sys_pfmg_data -> zero_guess)     = 0;
   (sys_pfmg_data -> max_levels)     = 0;
   (sys_pfmg_data -> dxyz)[0]        = 0.0;
   (sys_pfmg_data -> dxyz)[1]        = 0.0;
   (sys_pfmg_data -> dxyz)[2]        = 0.0;
   (sys_pfmg_data -> relax_type)     = 1;       /* weighted Jacobi */
   (sys_pfmg_data -> num_pre_relax)  = 1;
   (sys_pfmg_data -> num_post_relax) = 1;
   (sys_pfmg_data -> skip_relax)     = 1;
   (sys_pfmg_data -> logging)        = 0;

   /* initialize */
   (sys_pfmg_data -> num_levels) = -1;

   return (void *) sys_pfmg_data;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGDestroy
 *--------------------------------------------------------------------------*/

int
hypre_SysPFMGDestroy( void *sys_pfmg_vdata )
{
   hypre_SysPFMGData *sys_pfmg_data = sys_pfmg_vdata;

   int l;
   int ierr = 0;

   if (sys_pfmg_data)
   {
      if ((sys_pfmg_data -> logging) > 0)
      {
         hypre_TFree(sys_pfmg_data -> norms);
         hypre_TFree(sys_pfmg_data -> rel_norms);
      }

      if ((sys_pfmg_data -> num_levels) > -1)
      {
         for (l = 0; l < (sys_pfmg_data -> num_levels); l++)
         {
            hypre_SysPFMGRelaxDestroy(sys_pfmg_data -> relax_data_l[l]);
            hypre_SStructPMatvecDestroy(sys_pfmg_data -> matvec_data_l[l]);
         }
         for (l = 0; l < ((sys_pfmg_data -> num_levels) - 1); l++)
         {
            hypre_SysSemiRestrictDestroy(sys_pfmg_data -> restrict_data_l[l]);
            hypre_SysSemiInterpDestroy(sys_pfmg_data -> interp_data_l[l]);
         }
         hypre_TFree(sys_pfmg_data -> relax_data_l);
         hypre_TFree(sys_pfmg_data -> matvec_data_l);
         hypre_TFree(sys_pfmg_data -> restrict_data_l);
         hypre_TFree(sys_pfmg_data -> interp_data_l);
 
         hypre_SStructPVectorDestroy(sys_pfmg_data -> tx_l[0]);
         /*hypre_SStructPGridDestroy(sys_pfmg_data -> grid_l[0]);*/
         hypre_SStructPMatrixDestroy(sys_pfmg_data -> A_l[0]);
         hypre_SStructPVectorDestroy(sys_pfmg_data -> b_l[0]);
         hypre_SStructPVectorDestroy(sys_pfmg_data -> x_l[0]);
         for (l = 0; l < ((sys_pfmg_data -> num_levels) - 1); l++)
         {
            hypre_SStructPGridDestroy(sys_pfmg_data -> grid_l[l+1]);
            hypre_SStructPGridDestroy(sys_pfmg_data -> P_grid_l[l+1]);
            hypre_SStructPMatrixDestroy(sys_pfmg_data -> A_l[l+1]);
            hypre_SStructPMatrixDestroy(sys_pfmg_data -> P_l[l]);
            hypre_SStructPVectorDestroy(sys_pfmg_data -> b_l[l+1]);
            hypre_SStructPVectorDestroy(sys_pfmg_data -> x_l[l+1]);
            hypre_SStructPVectorDestroy(sys_pfmg_data -> tx_l[l+1]);
         }
         hypre_SharedTFree(sys_pfmg_data -> data);
         hypre_TFree(sys_pfmg_data -> cdir_l);
         hypre_TFree(sys_pfmg_data -> active_l);
         hypre_TFree(sys_pfmg_data -> grid_l);
         hypre_TFree(sys_pfmg_data -> P_grid_l);
         hypre_TFree(sys_pfmg_data -> A_l);
         hypre_TFree(sys_pfmg_data -> P_l);
         hypre_TFree(sys_pfmg_data -> RT_l);
         hypre_TFree(sys_pfmg_data -> b_l);
         hypre_TFree(sys_pfmg_data -> x_l);
         hypre_TFree(sys_pfmg_data -> tx_l);
      }
 
      hypre_FinalizeTiming(sys_pfmg_data -> time_index);
      hypre_TFree(sys_pfmg_data);
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGSetTol
 *--------------------------------------------------------------------------*/

int
hypre_SysPFMGSetTol( void   *sys_pfmg_vdata,
                     double  tol       )
{
   hypre_SysPFMGData *sys_pfmg_data = sys_pfmg_vdata;
   int                ierr = 0;
 
   (sys_pfmg_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGSetMaxIter
 *--------------------------------------------------------------------------*/

int
hypre_SysPFMGSetMaxIter( void *sys_pfmg_vdata,
                         int   max_iter  )
{
   hypre_SysPFMGData *sys_pfmg_data = sys_pfmg_vdata;
   int                ierr = 0;
 
   (sys_pfmg_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGSetRelChange
 *--------------------------------------------------------------------------*/

int
hypre_SysPFMGSetRelChange( void *sys_pfmg_vdata,
                           int   rel_change  )
{
   hypre_SysPFMGData *sys_pfmg_data = sys_pfmg_vdata;
   int                ierr = 0;
 
   (sys_pfmg_data -> rel_change) = rel_change;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
hypre_SysPFMGSetZeroGuess( void *sys_pfmg_vdata,
                           int   zero_guess )
{
   hypre_SysPFMGData *sys_pfmg_data = sys_pfmg_vdata;
   int                ierr = 0;
 
   (sys_pfmg_data -> zero_guess) = zero_guess;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGSetRelaxType
 *--------------------------------------------------------------------------*/

int
hypre_SysPFMGSetRelaxType( void *sys_pfmg_vdata,
                           int   relax_type )
{
   hypre_SysPFMGData *sys_pfmg_data = sys_pfmg_vdata;
   int                ierr = 0;
 
   (sys_pfmg_data -> relax_type) = relax_type;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGSetNumPreRelax
 *--------------------------------------------------------------------------*/

int
hypre_SysPFMGSetNumPreRelax( void *sys_pfmg_vdata,
                             int   num_pre_relax )
{
   hypre_SysPFMGData *sys_pfmg_data = sys_pfmg_vdata;
   int                ierr = 0;
 
   (sys_pfmg_data -> num_pre_relax) = num_pre_relax;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

int
hypre_SysPFMGSetNumPostRelax( void *sys_pfmg_vdata,
                              int   num_post_relax )
{
   hypre_SysPFMGData *sys_pfmg_data = sys_pfmg_vdata;
   int                ierr = 0;
 
   (sys_pfmg_data -> num_post_relax) = num_post_relax;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGSetNumSkipRelax
 *--------------------------------------------------------------------------*/

int
hypre_SysPFMGSetSkipRelax( void *sys_pfmg_vdata,
                           int  skip_relax )
{
   hypre_SysPFMGData *sys_pfmg_data = sys_pfmg_vdata;
   int                ierr = 0;
 
   (sys_pfmg_data -> skip_relax) = skip_relax;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGSetDxyz
 *--------------------------------------------------------------------------*/

int
hypre_SysPFMGSetDxyz( void   *sys_pfmg_vdata,
                      double *dxyz       )
{
   hypre_SysPFMGData *sys_pfmg_data = sys_pfmg_vdata;
   int                ierr = 0;

   (sys_pfmg_data -> dxyz[0]) = dxyz[0];
   (sys_pfmg_data -> dxyz[1]) = dxyz[1];
   (sys_pfmg_data -> dxyz[2]) = dxyz[2];
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGSetLogging
 *--------------------------------------------------------------------------*/

int
hypre_SysPFMGSetLogging( void *sys_pfmg_vdata,
                         int   logging)
{
   hypre_SysPFMGData *sys_pfmg_data = sys_pfmg_vdata;
   int                ierr = 0;
 
   (sys_pfmg_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGGetNumIterations
 *--------------------------------------------------------------------------*/

int
hypre_SysPFMGGetNumIterations( void *sys_pfmg_vdata,
                               int  *num_iterations )
{
   hypre_SysPFMGData *sys_pfmg_data = sys_pfmg_vdata;
   int                ierr = 0;

   *num_iterations = (sys_pfmg_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGPrintLogging
 *--------------------------------------------------------------------------*/

int
hypre_SysPFMGPrintLogging( void *sys_pfmg_vdata,
                           int   myid)
{
   hypre_SysPFMGData *sys_pfmg_data = sys_pfmg_vdata;
   int                ierr = 0;
   int                i;
   int                num_iterations  = (sys_pfmg_data -> num_iterations);
   int                logging   = (sys_pfmg_data -> logging);
   double            *norms     = (sys_pfmg_data -> norms);
   double            *rel_norms = (sys_pfmg_data -> rel_norms);

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
 * hypre_SysPFMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
hypre_SysPFMGGetFinalRelativeResidualNorm( void   *sys_pfmg_vdata,
                                           double *relative_residual_norm )
{
   hypre_SysPFMGData *sys_pfmg_data = sys_pfmg_vdata;

   int                max_iter        = (sys_pfmg_data -> max_iter);
   int                num_iterations  = (sys_pfmg_data -> num_iterations);
   int                logging         = (sys_pfmg_data -> logging);
   double            *rel_norms       = (sys_pfmg_data -> rel_norms);
            
   int                ierr = 0;

   
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


