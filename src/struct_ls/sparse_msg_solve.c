/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"
#include "sparse_msg.h"

#define DEBUG 0

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGSolve( void               *smsg_vdata,
                      hypre_StructMatrix *A,
                      hypre_StructVector *b,
                      hypre_StructVector *x          )
{
   hypre_SparseMSGData  *smsg_data = smsg_vdata;

   double                tol                 = (smsg_data -> tol);
   HYPRE_Int             max_iter            = (smsg_data -> max_iter);
   HYPRE_Int             rel_change          = (smsg_data -> rel_change);
   HYPRE_Int             zero_guess          = (smsg_data -> zero_guess);
   HYPRE_Int             jump                = (smsg_data -> jump);
   HYPRE_Int             num_pre_relax       = (smsg_data -> num_pre_relax);
   HYPRE_Int             num_post_relax      = (smsg_data -> num_post_relax);
   HYPRE_Int             num_fine_relax      = (smsg_data -> num_fine_relax);
   HYPRE_Int            *num_grids           = (smsg_data -> num_grids);
   HYPRE_Int             num_all_grids       = (smsg_data -> num_all_grids);
   HYPRE_Int             num_levels          = (smsg_data -> num_levels);
   hypre_StructMatrix  **A_array             = (smsg_data -> A_array);
   hypre_StructMatrix  **Px_array            = (smsg_data -> Px_array);
   hypre_StructMatrix  **Py_array            = (smsg_data -> Py_array);
   hypre_StructMatrix  **Pz_array            = (smsg_data -> Pz_array);
   hypre_StructMatrix  **RTx_array           = (smsg_data -> RTx_array);
   hypre_StructMatrix  **RTy_array           = (smsg_data -> RTy_array);
   hypre_StructMatrix  **RTz_array           = (smsg_data -> RTz_array);
   hypre_StructVector  **b_array             = (smsg_data -> b_array);
   hypre_StructVector  **x_array             = (smsg_data -> x_array);
   hypre_StructVector  **t_array             = (smsg_data -> t_array);
   hypre_StructVector  **r_array             = (smsg_data -> r_array);
   hypre_StructVector  **e_array             = (smsg_data -> e_array);
   hypre_StructVector  **visitx_array        = (smsg_data -> visitx_array);
   hypre_StructVector  **visity_array        = (smsg_data -> visity_array);
   hypre_StructVector  **visitz_array        = (smsg_data -> visitz_array);
   HYPRE_Int            *grid_on             = (smsg_data -> grid_on);
   void                **relax_array         = (smsg_data -> relax_array);
   void                **matvec_array        = (smsg_data -> matvec_array);
   void                **restrictx_array     = (smsg_data -> restrictx_array);
   void                **restricty_array     = (smsg_data -> restricty_array);
   void                **restrictz_array     = (smsg_data -> restrictz_array);
   void                **interpx_array       = (smsg_data -> interpx_array);
   void                **interpy_array       = (smsg_data -> interpy_array);
   void                **interpz_array       = (smsg_data -> interpz_array);
   HYPRE_Int             logging             = (smsg_data -> logging);
   double               *norms               = (smsg_data -> norms);
   double               *rel_norms           = (smsg_data -> rel_norms);

   HYPRE_Int            *restrict_count;

   double                b_dot_b, r_dot_r, eps;
   double                e_dot_e, x_dot_x;
                    
   HYPRE_Int             i, l, lx, ly, lz;
   HYPRE_Int             lymin, lymax, lzmin, lzmax;
   HYPRE_Int             fi, ci;                              
   HYPRE_Int             ierr = 0;

#if DEBUG
   char                  filename[255];
#endif

   /*-----------------------------------------------------
    * Initialize some things and deal with special cases
    *-----------------------------------------------------*/

   hypre_BeginTiming(smsg_data -> time_index);

   hypre_StructMatrixDestroy(A_array[0]);
   hypre_StructVectorDestroy(b_array[0]);
   hypre_StructVectorDestroy(x_array[0]);
   A_array[0] = hypre_StructMatrixRef(A);
   b_array[0] = hypre_StructVectorRef(b);
   x_array[0] = hypre_StructVectorRef(x);

   (smsg_data -> num_iterations) = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         hypre_StructVectorSetConstantValues(x, 0.0);
      }

      hypre_EndTiming(smsg_data -> time_index);
      return ierr;
   }

   /* part of convergence check */
   if (tol > 0.0)
   {
      /* eps = (tol^2) */
      b_dot_b = hypre_StructInnerProd(b_array[0], b_array[0]);
      eps = tol*tol;

      /* if rhs is zero, return a zero solution */
      if (b_dot_b == 0.0)
      {
         hypre_StructVectorSetConstantValues(x, 0.0);
         if (logging > 0)
         {
            norms[0]     = 0.0;
            rel_norms[0] = 0.0;
         }

         hypre_EndTiming(smsg_data -> time_index);
         return ierr;
      }
   }

   restrict_count = hypre_TAlloc(HYPRE_Int, num_all_grids);

   /*-----------------------------------------------------
    * Do V-cycles:
    *   For each index l, "fine" = l, "coarse" = (l+1)
    *-----------------------------------------------------*/

   for (i = 0; i < max_iter; i++)
   {
      /*--------------------------------------------------
       * Down cycle:
       *   Note that r = b = x through the jump region
       *--------------------------------------------------*/

      /* fine grid pre-relaxation */
      hypre_PFMGRelaxSetPreRelax(relax_array[0]);
      hypre_PFMGRelaxSetMaxIter(relax_array[0], num_fine_relax);
      hypre_PFMGRelaxSetZeroGuess(relax_array[0], zero_guess);
      hypre_PFMGRelax(relax_array[0], A_array[0], b_array[0], x_array[0]);
      zero_guess = 0;

      /* compute fine grid residual (b - Ax) */
      hypre_StructCopy(b_array[0], r_array[0]);
      hypre_StructMatvecCompute(matvec_array[0],
                                -1.0, A_array[0], x_array[0], 1.0, r_array[0]);

      /* convergence check */
      if (tol > 0.0)
      {
         r_dot_r = hypre_StructInnerProd(r_array[0], r_array[0]);

         if (logging > 0)
         {
            norms[i] = sqrt(r_dot_r);
            if (b_dot_b > 0)
               rel_norms[i] = sqrt(r_dot_r/b_dot_b);
            else
               rel_norms[i] = 0.0;
         }
/* RDF */
#if 0

hypre_printf("iter = %d, rel_norm = %e\n", i, rel_norms[i]);

#endif

         /* always do at least 1 V-cycle */
         if ((r_dot_r/b_dot_b < eps) && (i > 0))
         {
            if (rel_change)
            {
               if ((e_dot_e/x_dot_x) < eps)
                  break;
            }
            else
            {
               break;
            }
         }
      }

      if (num_levels > 1)
      {
         /* initialize restrict_count */
         for (fi = 0; fi < num_all_grids; fi++)
         {
            restrict_count[fi] = 0;
         }

         for (l = 0; l <= (num_levels - 2); l++)
         {
            lzmin = hypre_max((l - num_grids[1] - num_grids[0] + 2), 0);
            lzmax = hypre_min((l), (num_grids[2] - 1));
            for (lz = lzmin; lz <= lzmax; lz++)
            {
               lymin = hypre_max((l - lz - num_grids[0] + 1), 0);
               lymax = hypre_min((l - lz), (num_grids[1] - 1));
               for (ly = lymin; ly <= lymax; ly++)
               {
                  lx = l - lz - ly;

                  hypre_SparseMSGMapIndex(lx, ly, lz, num_grids, fi);

                  if (!grid_on[fi])
                  {
                     break;
                  }

                  if (restrict_count[fi] > 1)
                  {
                     hypre_StructScale((1.0/restrict_count[fi]), b_array[fi]);
                  }

                  if (l > jump)
                  {
                     /* pre-relaxation */
                     hypre_PFMGRelaxSetPreRelax(relax_array[fi]);
                     hypre_PFMGRelaxSetMaxIter(relax_array[fi], num_pre_relax);
                     hypre_PFMGRelaxSetZeroGuess(relax_array[fi], 1);
                     hypre_PFMGRelax(relax_array[fi], A_array[fi], b_array[fi],
                                     x_array[fi]);

                     /* compute residual (b - Ax) */
                     hypre_StructCopy(b_array[fi], r_array[fi]);
                     hypre_StructMatvecCompute(matvec_array[fi],
                                               -1.0, A_array[fi], x_array[fi],
                                               1.0, r_array[fi]);
                  }
                        
                  if ((lx+1) < num_grids[0])
                  {
                     /* restrict to ((lx+1), ly, lz) */
                     hypre_SparseMSGMapIndex((lx+1), ly, lz, num_grids, ci);
                     if (grid_on[ci])
                     {
                        if (restrict_count[ci])
                        {
                           hypre_SparseMSGRestrict(restrictx_array[fi],
                                                   RTx_array[lx], r_array[fi],
                                                   t_array[ci]);
                           hypre_StructAxpy(1.0, t_array[ci], b_array[ci]);
                        }
                        else
                        {
                           hypre_SparseMSGRestrict(restrictx_array[fi],
                                                   RTx_array[lx], r_array[fi],
                                                   b_array[ci]);
                        }
                        restrict_count[ci]++;
                     }
                  }
                  if ((ly+1) < num_grids[1])
                  {
                     /* restrict to (lx, (ly+1), lz) */
                     hypre_SparseMSGMapIndex(lx, (ly+1), lz, num_grids, ci);
                     if (grid_on[ci])
                     {
                        if (restrict_count[ci])
                        {
                           hypre_SparseMSGRestrict(restricty_array[fi],
                                                   RTy_array[ly], r_array[fi],
                                                   t_array[ci]);
                           hypre_StructAxpy(1.0, t_array[ci], b_array[ci]);
                        }
                        else
                        {
                           hypre_SparseMSGRestrict(restricty_array[fi],
                                                   RTy_array[ly], r_array[fi],
                                                   b_array[ci]);
                        }
                        restrict_count[ci]++;
                     }
                  }
                  if ((lz+1) < num_grids[2])
                  {
                     /* restrict to (lx, ly, (lz+1)) */
                     hypre_SparseMSGMapIndex(lx, ly, (lz+1), num_grids, ci);
                     if (grid_on[ci])
                     {
                        if (restrict_count[ci])
                        {
                           hypre_SparseMSGRestrict(restrictz_array[fi],
                                                   RTz_array[lz], r_array[fi],
                                                   t_array[ci]);
                           hypre_StructAxpy(1.0, t_array[ci], b_array[ci]);
                        }
                        else
                        {
                           hypre_SparseMSGRestrict(restrictz_array[fi],
                                                   RTz_array[lz], r_array[fi],
                                                   b_array[ci]);
                        }
                        restrict_count[ci]++;
                     }
                  }
#if DEBUG
                  hypre_sprintf(filename, "zoutSMSG_bdown.%d.%d.%d", lx, ly, lz);
                  hypre_StructVectorPrint(filename, b_array[fi], 0);
                  hypre_sprintf(filename, "zoutSMSG_xdown.%d.%d.%d", lx, ly, lz);
                  hypre_StructVectorPrint(filename, x_array[fi], 0);
                  hypre_sprintf(filename, "zoutSMSG_rdown.%d.%d.%d", lx, ly, lz);
                  hypre_StructVectorPrint(filename, r_array[fi], 0);
#endif
               }
            }
         }

         /*--------------------------------------------------
          * Bottom
          *--------------------------------------------------*/
      
         fi = num_all_grids - 1;

         if (restrict_count[fi] > 1)
         {
            hypre_StructScale((1.0/restrict_count[fi]), b_array[fi]);
         }

         hypre_PFMGRelaxSetZeroGuess(relax_array[fi], 1);
         hypre_PFMGRelax(relax_array[fi], A_array[fi], b_array[fi],
                         x_array[fi]);

#if DEBUG
         hypre_sprintf(filename, "zoutSMSG_bbottom.%d.%d.%d", lx, ly, lz);
         hypre_StructVectorPrint(filename, b_array[fi], 0);
         hypre_sprintf(filename, "zoutSMSG_xbottom.%d.%d.%d", lx, ly, lz);
         hypre_StructVectorPrint(filename, x_array[fi], 0);
#endif

         /*--------------------------------------------------
          * Up cycle
          *   Note that r = b = x through the jump region
          *--------------------------------------------------*/

         for (l = (num_levels - 2); l >= 0; l--)
         {
            lzmin = hypre_max((l - num_grids[1] - num_grids[0] + 2), 0);
            lzmax = hypre_min((l), (num_grids[2] - 1));
            for (lz = lzmax; lz >= lzmin; lz--)
            {
               lymin = hypre_max((l - lz - num_grids[0] + 1), 0);
               lymax = hypre_min((l - lz), (num_grids[1] - 1));
               for (ly = lymax; ly >= lymin; ly--)
               {
                  lx = l - lz - ly;

                  hypre_SparseMSGMapIndex(lx, ly, lz, num_grids, fi);
                     
                  if (!grid_on[fi])
                  {
                     break;
                  }

                  if ((l >= 1) && (l <= jump))
                  {
                     hypre_StructVectorSetConstantValues(x_array[fi], 0.0);
                  }
                  if ((lx+1) < num_grids[0])
                  {
                     /* interpolate from ((lx+1), ly, lz) */
                     hypre_SparseMSGMapIndex((lx+1), ly, lz, num_grids, ci);
                     if (grid_on[ci])
                     {
                        hypre_SparseMSGInterp(interpx_array[fi],
                                              Px_array[lx], x_array[ci],
                                              e_array[fi]);
                        hypre_SparseMSGFilter(visitx_array[fi], e_array[fi],
                                              lx, ly, lz, jump);
                        hypre_StructAxpy(1.0, e_array[fi], x_array[fi]);
                     }
                  }
                  if ((ly+1) < num_grids[1])
                  {
                     /* interpolate from (lx, (ly+1), lz) */
                     hypre_SparseMSGMapIndex(lx, (ly+1), lz, num_grids, ci);
                     if (grid_on[ci])
                     {
                        hypre_SparseMSGInterp(interpy_array[fi],
                                              Py_array[ly], x_array[ci],
                                              e_array[fi]);
                        hypre_SparseMSGFilter(visity_array[fi], e_array[fi],
                                              lx, ly, lz, jump);
                        hypre_StructAxpy(1.0, e_array[fi], x_array[fi]);
                     }
                  }
                  if ((lz+1) < num_grids[2])
                  {
                     /* interpolate from (lx, ly, (lz+1)) */
                     hypre_SparseMSGMapIndex(lx, ly, (lz+1), num_grids, ci);
                     if (grid_on[ci])
                     {
                        hypre_SparseMSGInterp(interpz_array[fi],
                                              Pz_array[lz], x_array[ci],
                                              e_array[fi]);
                        hypre_SparseMSGFilter(visitz_array[fi], e_array[fi],
                                              lx, ly, lz, jump);
                        hypre_StructAxpy(1.0, e_array[fi], x_array[fi]);
                     }
                  }               
#if DEBUG
                  hypre_sprintf(filename, "zoutSMSG_xup.%d.%d.%d", lx, ly, lz);
                  hypre_StructVectorPrint(filename, x_array[fi], 0);
#endif
                  if (l > jump)
                  {
                     /* post-relaxation */
                     hypre_PFMGRelaxSetPostRelax(relax_array[fi]);
                     hypre_PFMGRelaxSetMaxIter(relax_array[fi],
                                               num_post_relax);
                     hypre_PFMGRelaxSetZeroGuess(relax_array[fi], 0);
                     hypre_PFMGRelax(relax_array[fi], A_array[fi], b_array[fi],
                                     x_array[fi]);
                  }
               }
            }
         }
      }

      /* part of convergence check */
      if ((tol > 0.0) && (rel_change))
      {
         if (num_levels > 1)
         {
            e_dot_e = hypre_StructInnerProd(e_array[0], e_array[0]);
            x_dot_x = hypre_StructInnerProd(x_array[0], x_array[0]);
         }
         else
         {
            e_dot_e = 0.0;
            x_dot_x = 1.0;
         }
      }

      /* fine grid post-relaxation */
      hypre_PFMGRelaxSetPostRelax(relax_array[0]);
      hypre_PFMGRelaxSetMaxIter(relax_array[0], num_fine_relax);
      hypre_PFMGRelaxSetZeroGuess(relax_array[0], 0);
      hypre_PFMGRelax(relax_array[0], A_array[0], b_array[0], x_array[0]);

      (smsg_data -> num_iterations) = (i + 1);
   }

   hypre_EndTiming(smsg_data -> time_index);

   return ierr;
}

