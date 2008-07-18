/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/




#include "headers.h"
#include "maxwell_TV.h"

/*--------------------------------------------------------------------------
 * hypre_MaxwellTVCreate
 *--------------------------------------------------------------------------*/

void *
hypre_MaxwellTVCreate( MPI_Comm  comm )
{
   hypre_MaxwellData *maxwell_data;
   hypre_Index       *maxwell_rfactor;

   maxwell_data = hypre_CTAlloc(hypre_MaxwellData, 1);

   (maxwell_data -> comm)       = comm;
   (maxwell_data -> time_index) = hypre_InitializeTiming("Maxwell_Solver");

   /* set defaults */
   (maxwell_data -> tol)            = 1.0e-06;
   (maxwell_data -> max_iter)       = 200;
   (maxwell_data -> rel_change)     = 0;
   (maxwell_data -> zero_guess)     = 0;
   (maxwell_data -> num_pre_relax)  = 1;
   (maxwell_data -> num_post_relax) = 1;
   (maxwell_data -> constant_coef)  = 0;
   (maxwell_data -> print_level)    = 0;
   (maxwell_data -> logging)        = 0;

   maxwell_rfactor= hypre_TAlloc(hypre_Index, 1);
   hypre_SetIndex(maxwell_rfactor[0], 2, 2, 2);
   (maxwell_data -> rfactor)= maxwell_rfactor;
                                         

   return (void *) maxwell_data;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellTVDestroy
 *--------------------------------------------------------------------------*/

int
hypre_MaxwellTVDestroy( void *maxwell_vdata )
{
   hypre_MaxwellData *maxwell_data= maxwell_vdata;

   int l;
   int ierr = 0;

   if (maxwell_data)
   {
      hypre_TFree(maxwell_data-> rfactor);

      if ((maxwell_data -> logging) > 0)
      {
         hypre_TFree(maxwell_data -> norms);
         hypre_TFree(maxwell_data -> rel_norms);
      }

      if ((maxwell_data -> edge_numlevels) > 0)
      {
         for (l = 0; l < (maxwell_data-> edge_numlevels); l++)
         {
             HYPRE_SStructGridDestroy(maxwell_data-> egrid_l[l]);
             hypre_ParVectorDestroy(maxwell_data-> rese_l[l]);
             hypre_ParVectorDestroy(maxwell_data-> ee_l[l]);
             hypre_ParVectorDestroy(maxwell_data-> eVtemp_l[l]);
             hypre_ParVectorDestroy(maxwell_data-> eVtemp2_l[l]);
             hypre_TFree(maxwell_data -> eCF_marker_l[l]);

            /* Cannot destroy Aee_l[0] since it points to the user
               Aee_in. */
             if (l) 
             {
                hypre_ParCSRMatrixDestroy(maxwell_data-> Aee_l[l]);
                hypre_ParVectorDestroy(maxwell_data-> be_l[l]);
                hypre_ParVectorDestroy(maxwell_data-> xe_l[l]);
             }

             if (l < (maxwell_data-> edge_numlevels)-1) 
             {
                HYPRE_IJMatrixDestroy( 
                            (HYPRE_IJMatrix)  (maxwell_data-> Pe_l[l]));
             }

             hypre_TFree(maxwell_data-> BdryRanks_l[l]);
         }
         hypre_TFree(maxwell_data-> egrid_l);
         hypre_TFree(maxwell_data-> Aee_l);
         hypre_TFree(maxwell_data-> be_l);
         hypre_TFree(maxwell_data-> xe_l);
         hypre_TFree(maxwell_data-> rese_l);
         hypre_TFree(maxwell_data-> ee_l);
         hypre_TFree(maxwell_data-> eVtemp_l);
         hypre_TFree(maxwell_data-> eVtemp2_l);
         hypre_TFree(maxwell_data-> Pe_l);
         hypre_TFree(maxwell_data-> ReT_l);
         hypre_TFree(maxwell_data-> eCF_marker_l);
         hypre_TFree(maxwell_data-> erelax_weight);
         hypre_TFree(maxwell_data-> eomega);
         
         hypre_TFree(maxwell_data-> BdryRanks_l);
         hypre_TFree(maxwell_data-> BdryRanksCnts_l);
      }

      if ((maxwell_data -> node_numlevels) > 0)
      {
         for (l = 0; l < (maxwell_data-> node_numlevels); l++)
         {
             hypre_ParVectorDestroy(maxwell_data-> resn_l[l]);
             hypre_ParVectorDestroy(maxwell_data-> en_l[l]);
             hypre_ParVectorDestroy(maxwell_data-> nVtemp_l[l]);
             hypre_ParVectorDestroy(maxwell_data-> nVtemp2_l[l]);
         }
         hypre_BoomerAMGDestroy(maxwell_data-> amg_vdata);

         hypre_TFree(maxwell_data-> Ann_l);
         hypre_TFree(maxwell_data-> Pn_l);
         hypre_TFree(maxwell_data-> RnT_l);
         hypre_TFree(maxwell_data-> bn_l);
         hypre_TFree(maxwell_data-> xn_l);
         hypre_TFree(maxwell_data-> resn_l);
         hypre_TFree(maxwell_data-> en_l);
         hypre_TFree(maxwell_data-> nVtemp_l);
         hypre_TFree(maxwell_data-> nVtemp2_l);
         hypre_TFree(maxwell_data-> nCF_marker_l);
         hypre_TFree(maxwell_data-> nrelax_weight);
         hypre_TFree(maxwell_data-> nomega);
      }

      HYPRE_SStructStencilDestroy(maxwell_data-> Ann_stencils[0]);
      hypre_TFree(maxwell_data-> Ann_stencils);

      if ((maxwell_data -> en_numlevels) > 0)
      {
         for (l= 1; l< (maxwell_data-> en_numlevels); l++)
         {
             hypre_ParCSRMatrixDestroy(maxwell_data-> Aen_l[l]);
         }
      }
      hypre_TFree(maxwell_data-> Aen_l);

      HYPRE_SStructVectorDestroy(
           (HYPRE_SStructVector) maxwell_data-> bn);
      HYPRE_SStructVectorDestroy(
           (HYPRE_SStructVector) maxwell_data-> xn);
      HYPRE_SStructMatrixDestroy(
           (HYPRE_SStructMatrix) maxwell_data-> Ann);
      HYPRE_IJMatrixDestroy(maxwell_data-> Aen);

      hypre_ParCSRMatrixDestroy(maxwell_data-> T_transpose);

      hypre_FinalizeTiming(maxwell_data -> time_index);
      hypre_TFree(maxwell_data);
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellSetRfactors
 *--------------------------------------------------------------------------*/
int
hypre_MaxwellSetRfactors(void         *maxwell_vdata,
                          int           rfactor[3] )
{
   hypre_MaxwellData *maxwell_data   = maxwell_vdata;
   hypre_Index       *maxwell_rfactor=(maxwell_data -> rfactor);
   int                ierr       = 0;
                                                                                                              
   hypre_CopyIndex(rfactor, maxwell_rfactor[0]);
                                                                                                              
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellSetGrad
 *--------------------------------------------------------------------------*/
int
hypre_MaxwellSetGrad(void               *maxwell_vdata,
                     hypre_ParCSRMatrix *T )
{
   hypre_MaxwellData *maxwell_data= maxwell_vdata;
   int                ierr       = 0;
                                                                                                              
   (maxwell_data -> Tgrad)=  T;
                                                                                                              
   return ierr;
}
                                                                                                              
/*--------------------------------------------------------------------------
 * hypre_MaxwellSetConstantCoef
 *--------------------------------------------------------------------------*/
int
hypre_MaxwellSetConstantCoef( void   *maxwell_vdata,
                              int     constant_coef)
{
   hypre_MaxwellData *maxwell_data= maxwell_vdata;
   int                ierr        = 0;
                                                                                                                            
  (maxwell_data -> constant_coef) = constant_coef;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellSetTol
 *--------------------------------------------------------------------------*/
int
hypre_MaxwellSetTol( void   *maxwell_vdata,
                     double  tol       )
{
   hypre_MaxwellData *maxwell_data= maxwell_vdata;
   int                ierr        = 0;
                                                                                                                            
  (maxwell_data -> tol) = tol;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellSetMaxIter
 *--------------------------------------------------------------------------*/
int
hypre_MaxwellSetMaxIter( void *maxwell_vdata,
                         int   max_iter  )
{
   hypre_MaxwellData *maxwell_data = maxwell_vdata;
   int                ierr = 0;

  (maxwell_data -> max_iter) = max_iter;

   return ierr;
}
                                                                                                                            
/*--------------------------------------------------------------------------
 * hypre_MaxwellSetRelChange
 *--------------------------------------------------------------------------*/
int
hypre_MaxwellSetRelChange( void *maxwell_vdata,
                           int   rel_change  )
{
   hypre_MaxwellData *maxwell_data = maxwell_vdata;
   int                ierr = 0;
                                                                                                                            
  (maxwell_data -> rel_change) = rel_change;

   return ierr;
}
                                                                                                                            
/*--------------------------------------------------------------------------
 * hypre_MaxwellNumPreRelax
 *--------------------------------------------------------------------------*/
                                                                                                                            
int
hypre_MaxwellSetNumPreRelax( void *maxwell_vdata,
                             int   num_pre_relax )
{
   hypre_MaxwellData *maxwell_data= maxwell_vdata;
   int                ierr = 0;

  (maxwell_data -> num_pre_relax) = num_pre_relax;

   return ierr;
}
                                                                                                                            
/*--------------------------------------------------------------------------
 * hypre_MaxwellSetNumPostRelax
 *--------------------------------------------------------------------------*/
int
hypre_MaxwellSetNumPostRelax( void *maxwell_vdata,
                              int   num_post_relax )
{
   hypre_MaxwellData *maxwell_data= maxwell_vdata;
   int                ierr = 0;
                                                                                                                            
  (maxwell_data -> num_post_relax)= num_post_relax;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellGetNumIterations
 *--------------------------------------------------------------------------*/
int
hypre_MaxwellGetNumIterations( void *maxwell_vdata,
                               int  *num_iterations )
{
   hypre_MaxwellData *maxwell_data= maxwell_vdata;
   int                ierr = 0;
                                                                                                                            
  *num_iterations = (maxwell_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellSetPrintLevel
 *--------------------------------------------------------------------------*/
int
hypre_MaxwellSetPrintLevel( void *maxwell_vdata,
                            int   print_level)
{
   hypre_MaxwellData *maxwell_data = maxwell_vdata;
   int                ierr = 0;
                                                                                                             
   (maxwell_data -> print_level) = print_level;
                                                                                                             
   return ierr;
}
                                                                                                             
/*--------------------------------------------------------------------------
 * hypre_MaxwellSetLogging
 *--------------------------------------------------------------------------*/
int
hypre_MaxwellSetLogging( void *maxwell_vdata,
                         int   logging)
{
   hypre_MaxwellData *maxwell_data = maxwell_vdata;
   int                ierr = 0;
                                                                                                                       
   (maxwell_data -> logging) = logging;
                                                                                                                       
   return ierr;
}
                                                                                                                       
/*--------------------------------------------------------------------------
 * hypre_MaxwellPrintLogging
 *--------------------------------------------------------------------------*/
int
hypre_MaxwellPrintLogging( void *maxwell_vdata,
                           int   myid)
{
   hypre_MaxwellData *maxwell_data = maxwell_vdata;
   int                ierr = 0;
   int                i;
   int                num_iterations= (maxwell_data -> num_iterations);
   int                logging       = (maxwell_data -> logging);
   int                print_level   = (maxwell_data -> print_level);
   double            *norms         = (maxwell_data -> norms);
   double            *rel_norms     = (maxwell_data -> rel_norms);
                                                                                                                            
   if (myid == 0)
   {
     if (print_level > 0 )
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
   }

   return ierr;
}

int
hypre_MaxwellGetFinalRelativeResidualNorm( void   *maxwell_vdata,
                                           double *relative_residual_norm )
{
   hypre_MaxwellData *maxwell_data = maxwell_vdata;
                                                                                                                            
   int                max_iter        = (maxwell_data -> max_iter);
   int                num_iterations  = (maxwell_data -> num_iterations);
   int                logging         = (maxwell_data -> logging);
   double            *rel_norms       = (maxwell_data -> rel_norms);

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
