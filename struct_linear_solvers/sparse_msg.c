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
   hypre_SparseMSGData *smsg_data;

   smsg_data = hypre_CTAlloc(hypre_SparseMSGData, 1);

   (smsg_data -> comm)       = comm;
   (smsg_data -> time_index) = hypre_InitializeTiming("SparseMSG");

   /* set defaults */
   (smsg_data -> tol)            = 1.0e-06;
   (smsg_data -> max_iter)       = 200;
   (smsg_data -> rel_change)     = 0;
   (smsg_data -> zero_guess)     = 0;
   (smsg_data -> jump)           = 0;
   (smsg_data -> relax_type)     = 1;       /* weighted Jacobi */
   (smsg_data -> num_pre_relax)  = 1;
   (smsg_data -> num_post_relax) = 1;
   (smsg_data -> num_fine_relax) = 1;
   (smsg_data -> logging)        = 0;

   /* initialize */
   (smsg_data -> num_grids[0])    = 1;
   (smsg_data -> num_grids[1])    = 1;
   (smsg_data -> num_grids[2])    = 1;

   return (void *) smsg_data;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGDestroy
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGDestroy( void *smsg_vdata )
{
   hypre_SparseMSGData *smsg_data = smsg_vdata;

   int fi, l;
   int ierr = 0;

/* RDF */
#if 0
   if (smsg_data)
   {
      if ((smsg_data -> logging) > 0)
      {
         hypre_TFree(smsg_data -> norms);
         hypre_TFree(smsg_data -> rel_norms);
      }

      if ((smsg_data -> num_levels) > 1)
      {
         for (fi = 0; fi < (smsg_data -> num_all_grids); fi++)
         {
            hypre_PFMGRelaxDestroy(smsg_data -> relax_array[fi]);
            hypre_StructMatvecDestroy(smsg_data -> matvec_array[fi]);
            hypre_SemiRestrictDestroy(smsg_data -> restrictx_array[fi]);
            hypre_SemiRestrictDestroy(smsg_data -> restricty_array[fi]);
            hypre_SemiRestrictDestroy(smsg_data -> restrictz_array[fi]);
            hypre_SemiInterpDestroy(smsg_data -> interpx_array[fi]);
            hypre_SemiInterpDestroy(smsg_data -> interpy_array[fi]);
            hypre_SemiInterpDestroy(smsg_data -> interpz_array[fi]);
            hypre_StructMatrixDestroy(smsg_data -> A_array[fi]);
            hypre_StructVectorDestroy(smsg_data -> b_array[fi]);
            hypre_StructVectorDestroy(smsg_data -> x_array[fi]);
            hypre_StructVectorDestroy(smsg_data -> t_array[fi]);
            hypre_StructVectorDestroy(smsg_data -> r_array[fi]);
            hypre_StructVectorDestroy(smsg_data -> visitx_array[fi]);
            hypre_StructVectorDestroy(smsg_data -> visity_array[fi]);
            hypre_StructVectorDestroy(smsg_data -> visitz_array[fi]);
            hypre_StructGridDestroy(smsg_data -> grid_array[fi]);
         }

         for (l = 0; l < (smsg_data -> num_grids[0]) - 1; l++)
         {
            hypre_StructMatrixDestroy(smsg_data -> Px_array[l]);
            hypre_StructGridDestroy(smsg_data -> Px_grid_array[l]); 
         }
         for (l = 0; l < (smsg_data -> num_grids[1]) - 1; l++)
         {
            hypre_StructMatrixDestroy(smsg_data -> Py_array[l]);
            hypre_StructGridDestroy(smsg_data -> Py_grid_array[l]); 
         }
         for (l = 0; l < (smsg_data -> num_grids[2]) - 1; l++)
         {
            hypre_StructMatrixDestroy(smsg_data -> Pz_array[l]);
            hypre_StructGridDestroy(smsg_data -> Pz_grid_array[l]); 
         }

         hypre_SharedTFree(smsg_data -> data);

         hypre_TFree(smsg_data -> relax_array);
         hypre_TFree(smsg_data -> matvec_array);
         hypre_TFree(smsg_data -> restrictx_array);
         hypre_TFree(smsg_data -> restricty_array);
         hypre_TFree(smsg_data -> restrictz_array);
         hypre_TFree(smsg_data -> interpx_array);
         hypre_TFree(smsg_data -> interpy_array);
         hypre_TFree(smsg_data -> interpz_array);
         hypre_TFree(smsg_data -> A_array);
         hypre_TFree(smsg_data -> Px_array);
         hypre_TFree(smsg_data -> Py_array);
         hypre_TFree(smsg_data -> Pz_array);
         hypre_TFree(smsg_data -> RTx_array);
         hypre_TFree(smsg_data -> RTy_array);
         hypre_TFree(smsg_data -> RTz_array);
         hypre_TFree(smsg_data -> b_array);
         hypre_TFree(smsg_data -> x_array);
         hypre_TFree(smsg_data -> t_array);
         hypre_TFree(smsg_data -> r_array);
         hypre_TFree(smsg_data -> grid_array);
         hypre_TFree(smsg_data -> Px_grid_array);
         hypre_TFree(smsg_data -> Py_grid_array);
         hypre_TFree(smsg_data -> Pz_grid_array);
      }
 
      hypre_FinalizeTiming(smsg_data -> time_index);
      hypre_TFree(smsg_data);
   }
#endif
/* RDF */

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetTol
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetTol( void   *smsg_vdata,
                       double  tol        )
{
   hypre_SparseMSGData *smsg_data = smsg_vdata;
   int             ierr = 0;
 
   (smsg_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetMaxIter
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetMaxIter( void *smsg_vdata,
                           int   max_iter   )
{
   hypre_SparseMSGData *smsg_data = smsg_vdata;
   int             ierr = 0;
 
   (smsg_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetJump
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetJump(  void *smsg_vdata,
                         int   jump       )

{
   hypre_SparseMSGData *smsg_data = smsg_vdata;
   int                  ierr = 0;

   (smsg_data -> jump) = jump;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetRelChange
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetRelChange( void *smsg_vdata,
                             int   rel_change )
{
   hypre_SparseMSGData *smsg_data = smsg_vdata;
   int             ierr = 0;
 
   (smsg_data -> rel_change) = rel_change;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
hypre_SparseMSGSetZeroGuess( void *smsg_vdata,
                             int   zero_guess )
{
   hypre_SparseMSGData *smsg_data = smsg_vdata;
   int             ierr = 0;
 
   (smsg_data -> zero_guess) = zero_guess;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetRelaxType
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetRelaxType( void *smsg_vdata,
                             int   relax_type )
{
   hypre_SparseMSGData *smsg_data = smsg_vdata;
   int             ierr = 0;
 
   (smsg_data -> relax_type) = relax_type;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetNumPreRelax
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetNumPreRelax( void *smsg_vdata,
                               int   num_pre_relax )
{
   hypre_SparseMSGData *smsg_data = smsg_vdata;
   int             ierr = 0;
 
   (smsg_data -> num_pre_relax) = num_pre_relax;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetNumPostRelax
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetNumPostRelax( void *smsg_vdata,
                                int   num_post_relax )
{
   hypre_SparseMSGData *smsg_data = smsg_vdata;
   int             ierr = 0;
 
   (smsg_data -> num_post_relax) = num_post_relax;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetNumFineRelax
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetNumFineRelax( void *smsg_vdata,
                                int   num_fine_relax )
{
   hypre_SparseMSGData *smsg_data = smsg_vdata;
   int             ierr = 0;
 
   (smsg_data -> num_fine_relax) = num_fine_relax;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetLogging
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetLogging( void *smsg_vdata,
                           int   logging    )
{
   hypre_SparseMSGData *smsg_data = smsg_vdata;
   int             ierr = 0;
 
   (smsg_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGGetNumIterations
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGGetNumIterations( void *smsg_vdata,
                                 int  *num_iterations )
{
   hypre_SparseMSGData *smsg_data = smsg_vdata;
   int             ierr = 0;

   *num_iterations = (smsg_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGPrintLogging
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGPrintLogging( void *smsg_vdata,
                             int   myid       )
{
   hypre_SparseMSGData *smsg_data = smsg_vdata;
   int             ierr = 0;
   int             i;
   int             num_iterations  = (smsg_data -> num_iterations);
   int             logging   = (smsg_data -> logging);
   double         *norms     = (smsg_data -> norms);
   double         *rel_norms = (smsg_data -> rel_norms);

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
hypre_SparseMSGGetFinalRelativeResidualNorm( void   *smsg_vdata,
                                             double *relative_residual_norm )
{
   hypre_SparseMSGData *smsg_data = smsg_vdata;

   int             max_iter        = (smsg_data -> max_iter);
   int             num_iterations  = (smsg_data -> num_iterations);
   int             logging         = (smsg_data -> logging);
   double         *rel_norms       = (smsg_data -> rel_norms);
            
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


