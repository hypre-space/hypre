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
 * zzz_SMGInitialize
 *--------------------------------------------------------------------------*/

void *
zzz_SMGInitialize( MPI_Comm *comm )
{
   zzz_SMGData *smg_data;

   smg_data = zzz_CTAlloc(zzz_SMGData, 1);

   (smg_data -> comm)        = comm;
   (smg_data -> base_index)  = zzz_NewIndex();
   (smg_data -> base_stride) = zzz_NewIndex();

   /* set defaults */
   (smg_data -> tol)        = 1.0e-06;
   (smg_data -> max_iter)   = 200;
   (smg_data -> zero_guess) = 0;
   (smg_data -> max_levels) = 0;
   (smg_data -> cdir) = 2;
   (smg_data -> ci) = 0;
   (smg_data -> fi) = 1;
   (smg_data -> cs) = 2;
   (smg_data -> fs) = 2;
   zzz_SetIndex((smg_data -> base_index), 0, 0, 0);
   zzz_SetIndex((smg_data -> base_stride), 1, 1, 1);

   return (void *) smg_data;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetTol
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetTol( void   *smg_vdata,
               double  tol       )
{
   zzz_SMGData *smg_data = smg_vdata;
   int          ierr = 0;
 
   (smg_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetMaxIter
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetMaxIter( void *smg_vdata,
                   int   max_iter  )
{
   zzz_SMGData *smg_data = smg_vdata;
   int          ierr = 0;
 
   (smg_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
zzz_SMGSetZeroGuess( void *smg_vdata )
{
   zzz_SMGData *smg_data = smg_vdata;
   int          ierr = 0;
 
   (smg_data -> zero_guess) = 1;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetBase
 *--------------------------------------------------------------------------*/
 
int
zzz_SMGSetBase( void      *smg_vdata,
                zzz_Index *base_index,
                zzz_Index *base_stride )
{
   zzz_SMGData *smg_data = smg_vdata;
   int          d;
   int          ierr = 0;
 
   for (d = 0; d < 3; d++)
   {
      zzz_IndexD((smg_data -> base_index),  d) = zzz_IndexD(base_index,  d);
      zzz_IndexD((smg_data -> base_stride), d) = zzz_IndexD(base_stride, d);
   }
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGGetNumIterations
 *--------------------------------------------------------------------------*/

int
zzz_SMGGetNumIterations( void *smg_vdata,
                         int  *num_iterations )
{
   zzz_SMGData *smg_data = smg_vdata;
   int ierr;

   *num_iterations = (smg_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGFinalize
 *--------------------------------------------------------------------------*/

int
zzz_SMGFinalize( void *smg_vdata )
{
   zzz_SMGData *smg_data = smg_vdata;

   int l;
   int ierr;

   if (smg_data)
   {
      if ((smg_data -> logging) > 0)
      {
         zzz_TFree(smg_data -> norms);
         zzz_TFree(smg_data -> rel_norms);
      }

      zzz_SMGRelaxFinalize(smg_data -> pre_relax_data_initial);
      zzz_SMGRelaxFinalize(smg_data -> coarse_relax_data);
      for (l = 0; l < ((smg_data -> num_levels) - 1); l++)
      {
         zzz_SMGRelaxFinalize(smg_data -> pre_relax_data_l[l]);
         zzz_SMGRelaxFinalize(smg_data -> post_relax_data_l[l]);
         zzz_SMGResidualFinalize(smg_data -> residual_data_l[l]);
         zzz_SMGRestrictFinalize(smg_data -> restrict_data_l[l]);
         zzz_SMGIntAddFinalize(smg_data -> intadd_data_l[l]);
      }
      zzz_TFree(smg_data -> pre_relax_data_l);
      zzz_TFree(smg_data -> post_relax_data_l);
      zzz_TFree(smg_data -> residual_data_l);
      zzz_TFree(smg_data -> restrict_data_l);
      zzz_TFree(smg_data -> intadd_data_l);
 
      zzz_FreeStructVector(smg_data -> r_l[0]);
      for (l = 0; l < ((smg_data -> num_levels) - 1); l++)
      {
         zzz_FreeStructGrid(smg_data -> grid_l[l+1]);
         zzz_FreeStructMatrix(smg_data -> A_l[l+1]);
         zzz_FreeStructMatrix(smg_data -> PT_l[l]);
         if (!zzz_StructMatrixSymmetric(smg_data -> A_l[0]))
            zzz_FreeStructMatrix(smg_data -> R_l[l]);
         zzz_FreeStructVector(smg_data -> b_l[l+1]);
         zzz_FreeStructVector(smg_data -> x_l[l+1]);
         zzz_FreeStructVectorShell(smg_data -> r_l[l+1]);
      }
      zzz_TFree(smg_data -> grid_l);
      zzz_TFree(smg_data -> A_l);
      zzz_TFree(smg_data -> PT_l);
      zzz_TFree(smg_data -> R_l);
      zzz_TFree(smg_data -> b_l);
      zzz_TFree(smg_data -> x_l);
      zzz_TFree(smg_data -> r_l);
      zzz_TFree(smg_data -> e_l);
 
      for (l = 0; l < 2; l++)
      {
         zzz_FreeIndex(smg_data -> base_index_l[l]);
         zzz_FreeIndex(smg_data -> cindex_l[l]);
         zzz_FreeIndex(smg_data -> findex_l[l]);
         zzz_FreeIndex(smg_data -> base_stride_l[l]);
         zzz_FreeIndex(smg_data -> cstride_l[l]);
         zzz_FreeIndex(smg_data -> fstride_l[l]);
      }
      zzz_TFree(smg_data -> base_index_l);
      zzz_TFree(smg_data -> cindex_l);
      zzz_TFree(smg_data -> findex_l);
      zzz_TFree(smg_data -> base_stride_l);
      zzz_TFree(smg_data -> cstride_l);
      zzz_TFree(smg_data -> fstride_l);
 
      zzz_TFree(smg_data);
   }

   return(ierr);
}

