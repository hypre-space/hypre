/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
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
#include "smg.h"

/*--------------------------------------------------------------------------
 * hypre_SMGCreate
 *--------------------------------------------------------------------------*/

void *
hypre_SMGCreate( MPI_Comm  comm )
{
   hypre_SMGData *smg_data;

   smg_data = hypre_CTAlloc(hypre_SMGData, 1);

   (smg_data -> comm)        = comm;
   (smg_data -> time_index)  = hypre_InitializeTiming("SMG");

   /* set defaults */
   (smg_data -> memory_use) = 0;
   (smg_data -> tol)        = 1.0e-06;
   (smg_data -> max_iter)   = 200;
   (smg_data -> rel_change) = 0;
   (smg_data -> zero_guess) = 0;
   (smg_data -> max_levels) = 0;
   (smg_data -> num_pre_relax)  = 1;
   (smg_data -> num_post_relax) = 1;
   (smg_data -> cdir) = 2;
   hypre_SetIndex((smg_data -> base_index), 0, 0, 0);
   hypre_SetIndex((smg_data -> base_stride), 1, 1, 1);
   (smg_data -> logging) = 0;

   /* initialize */
   (smg_data -> num_levels) = -1;

   return (void *) smg_data;
}

/*--------------------------------------------------------------------------
 * hypre_SMGDestroy
 *--------------------------------------------------------------------------*/

int
hypre_SMGDestroy( void *smg_vdata )
{
   hypre_SMGData *smg_data = smg_vdata;

   int l;
   int ierr = 0;

   if (smg_data)
   {
      if ((smg_data -> logging) > 0)
      {
         hypre_TFree(smg_data -> norms);
         hypre_TFree(smg_data -> rel_norms);
      }

      if ((smg_data -> num_levels) > -1)
      {
         for (l = 0; l < ((smg_data -> num_levels) - 1); l++)
         {
            hypre_SMGRelaxDestroy(smg_data -> relax_data_l[l]);
            hypre_SMGResidualDestroy(smg_data -> residual_data_l[l]);
            hypre_SemiRestrictDestroy(smg_data -> restrict_data_l[l]);
            hypre_SemiInterpDestroy(smg_data -> interp_data_l[l]);
         }
         hypre_SMGRelaxDestroy(smg_data -> relax_data_l[l]);
         if (l == 0)
         {
            hypre_SMGResidualDestroy(smg_data -> residual_data_l[l]);
         }
         hypre_TFree(smg_data -> relax_data_l);
         hypre_TFree(smg_data -> residual_data_l);
         hypre_TFree(smg_data -> restrict_data_l);
         hypre_TFree(smg_data -> interp_data_l);
 
         hypre_StructVectorDestroy(smg_data -> tb_l[0]);
         hypre_StructVectorDestroy(smg_data -> tx_l[0]);
         hypre_StructGridDestroy(smg_data -> grid_l[0]);
         hypre_StructMatrixDestroy(smg_data -> A_l[0]);
         hypre_StructVectorDestroy(smg_data -> b_l[0]);
         hypre_StructVectorDestroy(smg_data -> x_l[0]);
         for (l = 0; l < ((smg_data -> num_levels) - 1); l++)
         {
            hypre_StructGridDestroy(smg_data -> grid_l[l+1]);
            hypre_StructGridDestroy(smg_data -> PT_grid_l[l+1]);
            hypre_StructMatrixDestroy(smg_data -> A_l[l+1]);
            if (smg_data -> PT_l[l] == smg_data -> R_l[l])
            {
               hypre_StructMatrixDestroy(smg_data -> PT_l[l]);
            }
            else
            {
               hypre_StructMatrixDestroy(smg_data -> PT_l[l]);
               hypre_StructMatrixDestroy(smg_data -> R_l[l]);
            }
            hypre_StructVectorDestroy(smg_data -> b_l[l+1]);
            hypre_StructVectorDestroy(smg_data -> x_l[l+1]);
            hypre_StructVectorDestroy(smg_data -> tb_l[l+1]);
            hypre_StructVectorDestroy(smg_data -> tx_l[l+1]);
         }
         hypre_SharedTFree(smg_data -> data);
         hypre_TFree(smg_data -> grid_l);
         hypre_TFree(smg_data -> PT_grid_l);
         hypre_TFree(smg_data -> A_l);
         hypre_TFree(smg_data -> PT_l);
         hypre_TFree(smg_data -> R_l);
         hypre_TFree(smg_data -> b_l);
         hypre_TFree(smg_data -> x_l);
         hypre_TFree(smg_data -> tb_l);
         hypre_TFree(smg_data -> tx_l);
      }
 
      hypre_FinalizeTiming(smg_data -> time_index);
      hypre_TFree(smg_data);
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetMemoryUse
 *--------------------------------------------------------------------------*/

int
hypre_SMGSetMemoryUse( void *smg_vdata,
                       int   memory_use )
{
   hypre_SMGData *smg_data = smg_vdata;
   int            ierr = 0;
 
   (smg_data -> memory_use) = memory_use;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetTol
 *--------------------------------------------------------------------------*/

int
hypre_SMGSetTol( void   *smg_vdata,
                 double  tol       )
{
   hypre_SMGData *smg_data = smg_vdata;
   int            ierr = 0;
 
   (smg_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetMaxIter
 *--------------------------------------------------------------------------*/

int
hypre_SMGSetMaxIter( void *smg_vdata,
                     int   max_iter  )
{
   hypre_SMGData *smg_data = smg_vdata;
   int            ierr = 0;
 
   (smg_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetRelChange
 *--------------------------------------------------------------------------*/

int
hypre_SMGSetRelChange( void *smg_vdata,
                       int   rel_change  )
{
   hypre_SMGData *smg_data = smg_vdata;
   int            ierr = 0;
 
   (smg_data -> rel_change) = rel_change;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
hypre_SMGSetZeroGuess( void *smg_vdata,
                       int   zero_guess )
{
   hypre_SMGData *smg_data = smg_vdata;
   int            ierr = 0;
 
   (smg_data -> zero_guess) = zero_guess;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetNumPreRelax
 * Note that we require at least 1 pre-relax sweep. 
 *--------------------------------------------------------------------------*/

int
hypre_SMGSetNumPreRelax( void *smg_vdata,
                         int   num_pre_relax )
{
   hypre_SMGData *smg_data = smg_vdata;
   int            ierr = 0;
 
   (smg_data -> num_pre_relax) = hypre_max(num_pre_relax,1);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

int
hypre_SMGSetNumPostRelax( void *smg_vdata,
                          int   num_post_relax )
{
   hypre_SMGData *smg_data = smg_vdata;
   int            ierr = 0;
 
   (smg_data -> num_post_relax) = num_post_relax;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetBase
 *--------------------------------------------------------------------------*/
 
int
hypre_SMGSetBase( void        *smg_vdata,
                  hypre_Index  base_index,
                  hypre_Index  base_stride )
{
   hypre_SMGData *smg_data = smg_vdata;
   int            d;
   int            ierr = 0;
 
   for (d = 0; d < 3; d++)
   {
      hypre_IndexD((smg_data -> base_index),  d) =
         hypre_IndexD(base_index,  d);
      hypre_IndexD((smg_data -> base_stride), d) =
         hypre_IndexD(base_stride, d);
   }
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetLogging
 *--------------------------------------------------------------------------*/

int
hypre_SMGSetLogging( void *smg_vdata,
                     int   logging)
{
   hypre_SMGData *smg_data = smg_vdata;
   int            ierr = 0;
 
   (smg_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGGetNumIterations
 *--------------------------------------------------------------------------*/

int
hypre_SMGGetNumIterations( void *smg_vdata,
                           int  *num_iterations )
{
   hypre_SMGData *smg_data = smg_vdata;
   int            ierr = 0;

   *num_iterations = (smg_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGPrintLogging
 *--------------------------------------------------------------------------*/

int
hypre_SMGPrintLogging( void *smg_vdata,
                       int   myid)
{
   hypre_SMGData *smg_data = smg_vdata;
   int          ierr = 0;
   int          i;
   int          num_iterations  = (smg_data -> num_iterations);
   int          logging   = (smg_data -> logging);
   double      *norms     = (smg_data -> norms);
   double      *rel_norms = (smg_data -> rel_norms);

   
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
 * hypre_SMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
hypre_SMGGetFinalRelativeResidualNorm( void   *smg_vdata,
                                       double *relative_residual_norm )
{
   hypre_SMGData *smg_data = smg_vdata;

   int            max_iter        = (smg_data -> max_iter);
   int            num_iterations  = (smg_data -> num_iterations);
   int            logging         = (smg_data -> logging);
   double        *rel_norms       = (smg_data -> rel_norms);

   int            ierr = -1;

   
   if (logging > 0)
   {
      if (num_iterations == max_iter)
      {
         *relative_residual_norm = rel_norms[num_iterations-1];
      }
      else
      {
         *relative_residual_norm = rel_norms[num_iterations];
      }

      ierr = 0;
   }
   
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetStructVectorConstantValues
 *--------------------------------------------------------------------------*/

int
hypre_SMGSetStructVectorConstantValues( hypre_StructVector *vector,
                                        double              values,
                                        hypre_BoxArray     *box_array,
                                        hypre_Index         stride    )
{
   int    ierr = 0;

   hypre_Box          *v_data_box;

   int                 vi;
   double             *vp;

   hypre_Box          *box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;

   int                 loopi, loopj, loopk;
   int                 i;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_ForBoxI(i, box_array)
      {
         box   = hypre_BoxArrayBox(box_array, i);
         start = hypre_BoxIMin(box);

         v_data_box =
            hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);
         vp = hypre_StructVectorBoxData(vector, i);

         hypre_BoxGetStrideSize(box, stride, loop_size);

         hypre_BoxLoop1Begin(loop_size,
                             v_data_box, start, stride, vi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,vi
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop1For(loopi, loopj, loopk, vi)
            {
               vp[vi] = values;
            }
         hypre_BoxLoop1End(vi);
      }

   return ierr;
}


