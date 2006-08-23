/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_FACCreate
 *--------------------------------------------------------------------------*/
void *
hypre_FACCreate( MPI_Comm  comm )
{
    hypre_FACData *fac_data;

    fac_data = hypre_CTAlloc(hypre_FACData, 1);

   (fac_data -> comm)       = comm;
   (fac_data -> time_index) = hypre_InitializeTiming("FAC");

   /* set defaults */
   (fac_data -> tol)            = 1.0e-06;
   (fac_data -> max_cycles)     = 200;
   (fac_data -> zero_guess)     = 0;
   (fac_data -> max_levels)     = 0;
   (fac_data -> relax_type)     = 2; /*  1 Jacobi; 2 Gauss-Seidel */
   (fac_data -> num_pre_smooth) = 1;
   (fac_data -> num_post_smooth)= 1;
   (fac_data -> csolver_type)   = 1;
   (fac_data -> logging)        = 0;

   return (void *) fac_data;
}

/*--------------------------------------------------------------------------
 * hypre_FACDestroy
 *--------------------------------------------------------------------------*/
int
hypre_FACDestroy2(void *fac_vdata)
{
   hypre_FACData *fac_data = fac_vdata;

   int level;
   int ierr = 0;

   if (fac_data)
   {
      hypre_TFree((fac_data ->plevels) );
      hypre_TFree((fac_data ->prefinements) );

      HYPRE_SStructGraphDestroy(hypre_SStructMatrixGraph((fac_data -> A_rap)));
      HYPRE_SStructMatrixDestroy((fac_data -> A_rap));
      for (level= 0; level<= (fac_data -> max_levels); level++)
      {
         HYPRE_SStructMatrixDestroy( (fac_data -> A_level[level]) );
         HYPRE_SStructVectorDestroy( (fac_data -> x_level[level]) );
         HYPRE_SStructVectorDestroy( (fac_data -> b_level[level]) );
         HYPRE_SStructVectorDestroy( (fac_data -> r_level[level]) );
         HYPRE_SStructVectorDestroy( (fac_data -> e_level[level]) );
         hypre_SStructPVectorDestroy( (fac_data -> tx_level[level]) );

         HYPRE_SStructGraphDestroy( (fac_data -> graph_level[level]) );
         HYPRE_SStructGridDestroy(  (fac_data -> grid_level[level]) );

         hypre_SStructMatvecDestroy( (fac_data   -> matvec_data_level[level]) );
         hypre_SStructPMatvecDestroy((fac_data  -> pmatvec_data_level[level]) );

         hypre_SysPFMGRelaxDestroy( (fac_data -> relax_data_level[level]) );

         if (level > 0)
         {
            hypre_FacSemiRestrictDestroy2( (fac_data -> restrict_data_level[level]) );
         }

         if (level < (fac_data -> max_levels))
         {
            hypre_FacSemiInterpDestroy2( (fac_data -> interp_data_level[level]) );
         }
      }
      hypre_SStructMatvecDestroy( (fac_data -> matvec_data) );

      hypre_TFree(fac_data -> A_level);
      hypre_TFree(fac_data -> x_level); 
      hypre_TFree(fac_data -> b_level); 
      hypre_TFree(fac_data -> r_level); 
      hypre_TFree(fac_data -> e_level); 
      hypre_TFree(fac_data -> tx_level); 
      hypre_TFree(fac_data -> relax_data_level); 
      hypre_TFree(fac_data -> restrict_data_level); 
      hypre_TFree(fac_data -> matvec_data_level); 
      hypre_TFree(fac_data -> pmatvec_data_level); 
      hypre_TFree(fac_data -> interp_data_level); 

      hypre_TFree(fac_data -> grid_level); 
      hypre_TFree(fac_data -> graph_level); 

      HYPRE_SStructVectorDestroy(fac_data -> tx);

      hypre_TFree(fac_data -> level_to_part);
      hypre_TFree(fac_data -> part_to_level);
      hypre_TFree(fac_data -> refine_factors);

      if ( (fac_data -> csolver_type) == 1)
      {
          HYPRE_SStructPCGDestroy(fac_data -> csolver);
          HYPRE_SStructSysPFMGDestroy(fac_data -> cprecond);
      }
      else if ((fac_data -> csolver_type) == 2)
      {
          HYPRE_SStructSysPFMGDestroy(fac_data -> csolver);
      }

      if ((fac_data -> logging) > 0)
      {
         hypre_TFree(fac_data -> norms);
         hypre_TFree(fac_data -> rel_norms);
      }

      hypre_FinalizeTiming(fac_data -> time_index);

      hypre_TFree(fac_data);
   }

   return(ierr);
}

int
hypre_FACSetTol( void   *fac_vdata,
                 double  tol       )
{
   hypre_FACData *fac_data = fac_vdata;
   int                ierr = 0;

   (fac_data -> tol) = tol;

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_FACSetPLevels
 *--------------------------------------------------------------------------*/
int
hypre_FACSetPLevels( void *fac_vdata,
                     int   nparts,
                     int  *plevels)
{
   hypre_FACData *fac_data   = fac_vdata;
   int           *fac_plevels;
   int            ierr       = 0;
   int            i;

   fac_plevels= hypre_CTAlloc(int, nparts);

   for (i= 0; i< nparts; i++)
   {
      fac_plevels[i]= plevels[i];
   }

   (fac_data -> plevels)=  fac_plevels;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetPRefinements
 *--------------------------------------------------------------------------*/
int
hypre_FACSetPRefinements( void         *fac_vdata,
                          int           nparts,
                          int         (*prefinements)[3] )
{
   hypre_FACData *fac_data   = fac_vdata;
   hypre_Index   *fac_prefinements;
   int            ierr       = 0;
   int            i;

   fac_prefinements= hypre_TAlloc(hypre_Index, nparts);

   for (i= 0; i< nparts; i++)
   {
      hypre_CopyIndex( prefinements[i], fac_prefinements[i] );
   }

   (fac_data -> prefinements)=  fac_prefinements;

   return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_FACSetMaxLevels
 *--------------------------------------------------------------------------*/

int
hypre_FACSetMaxLevels( void *fac_vdata,
                       int   nparts )
{
   hypre_FACData *fac_data = fac_vdata;
   int                ierr = 0;

   (fac_data -> max_levels) = nparts-1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetMaxIter
 *--------------------------------------------------------------------------*/

int
hypre_FACSetMaxIter( void *fac_vdata,
                     int   max_iter  )
{
   hypre_FACData *fac_data = fac_vdata;
   int                ierr = 0;

   (fac_data -> max_cycles) = max_iter;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetRelChange
 *--------------------------------------------------------------------------*/

int
hypre_FACSetRelChange( void *fac_vdata,
                       int   rel_change  )
{
   hypre_FACData *fac_data = fac_vdata;
   int                ierr = 0;

   (fac_data -> rel_change) = rel_change;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetZeroGuess
 *--------------------------------------------------------------------------*/

int
hypre_FACSetZeroGuess( void *fac_vdata,
                       int   zero_guess )
{
   hypre_FACData *fac_data = fac_vdata;
   int                ierr = 0;

   (fac_data -> zero_guess) = zero_guess;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetRelaxType
 *--------------------------------------------------------------------------*/

int
hypre_FACSetRelaxType( void *fac_vdata,
                       int   relax_type )
{
   hypre_FACData *fac_data = fac_vdata;
   int                ierr = 0;

   (fac_data -> relax_type) = relax_type;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetNumPreRelax
 *--------------------------------------------------------------------------*/

int
hypre_FACSetNumPreSmooth( void *fac_vdata,
                          int   num_pre_smooth )
{
   hypre_FACData *fac_data = fac_vdata;
   int                ierr = 0;

   (fac_data -> num_pre_smooth) = num_pre_smooth;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetNumPostRelax
 *--------------------------------------------------------------------------*/

int
hypre_FACSetNumPostSmooth( void *fac_vdata,
                           int   num_post_smooth )
{
   hypre_FACData *fac_data = fac_vdata;
   int                ierr = 0;

   (fac_data -> num_post_smooth) = num_post_smooth;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetCoarseSolverType
 *--------------------------------------------------------------------------*/

int
hypre_FACSetCoarseSolverType( void *fac_vdata,
                              int   csolver_type)
{
   hypre_FACData *fac_data = fac_vdata;
   int                ierr = 0;

   (fac_data -> csolver_type) = csolver_type;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetLogging
 *--------------------------------------------------------------------------*/

int
hypre_FACSetLogging( void *fac_vdata,
                     int   logging)
{
   hypre_FACData *fac_data = fac_vdata;
   int                ierr = 0;

   (fac_data -> logging) = logging;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysFACGetNumIterations
 *--------------------------------------------------------------------------*/

int
hypre_FACGetNumIterations( void *fac_vdata,
                           int  *num_iterations )
{
   hypre_FACData *fac_data = fac_vdata;
   int                ierr = 0;

   *num_iterations = (fac_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACPrintLogging
 *--------------------------------------------------------------------------*/

int
hypre_FACPrintLogging( void *fac_vdata,
                       int   myid)
{
   hypre_FACData *fac_data = fac_vdata;
   int                ierr = 0;
   int                i;
   int                num_iterations  = (fac_data -> num_iterations);
   int                logging   = (fac_data -> logging);
   double            *norms     = (fac_data -> norms);
   double            *rel_norms = (fac_data -> rel_norms);

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
 * hypre_FACGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
hypre_FACGetFinalRelativeResidualNorm( void   *fac_vdata,
                                       double *relative_residual_norm )
{
   hypre_FACData *fac_data = fac_vdata;

   int                max_iter        = (fac_data -> max_cycles);
   int                num_iterations  = (fac_data -> num_iterations);
   int                logging         = (fac_data -> logging);
   double            *rel_norms       = (fac_data -> rel_norms);

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

