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

#define DEBUG 0

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSolve
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSolve( void               *SparseMSG_vdata,
                 hypre_StructMatrix *A,
                 hypre_StructVector *b,
                 hypre_StructVector *x         )
{
   hypre_SparseMSGData       *SparseMSG_data = SparseMSG_vdata;

   double                tol                  	= (SparseMSG_data -> tol);
   int                   max_iter             	= (SparseMSG_data -> max_iter);
   int                   rel_change           	= (SparseMSG_data -> rel_change);
   int                   zero_guess           	= (SparseMSG_data -> zero_guess);
   int                   jump                   = (SparseMSG_data -> jump);
   int                   num_pre_relax     	= (SparseMSG_data -> num_pre_relax);
   int                   num_post_relax 	= (SparseMSG_data -> num_post_relax);
   int                   total_num_levels       = (SparseMSG_data -> total_num_levels);
   int                   num_grids              = (SparseMSG_data -> num_grids);
   int                  *num_levels             = (SparseMSG_data -> num_levels);
   hypre_StructMatrix  **A_array              	= (SparseMSG_data -> A_array);
   hypre_StructMatrix  **P_array              	= (SparseMSG_data -> P_array);
   hypre_StructMatrix  **RT_array             	= (SparseMSG_data -> RT_array);
   hypre_StructVector  **b_array              	= (SparseMSG_data -> b_array);
   hypre_StructVector  **x_array              	= (SparseMSG_data -> x_array);
   hypre_StructVector  **r_array              	= (SparseMSG_data -> r_array);
   hypre_StructVector  **e_array              	= (SparseMSG_data -> e_array);
   hypre_StructVector  **tx_array               = (SparseMSG_data -> tx_array);
   void                **relax_data_array     	= (SparseMSG_data -> relax_data_array);
   void                **matvec_data_array    	= (SparseMSG_data -> matvec_data_array);
   void                **restrict_data_array  	= (SparseMSG_data -> restrict_data_array);
   void                **interp_data_array    	= (SparseMSG_data -> interp_data_array);
   int                   logging              	= (SparseMSG_data -> logging);
   double               *restrict_weights       = (SparseMSG_data -> restrict_weights);
   double               *interp_weights         = (SparseMSG_data -> interp_weights);
   double               *norms                	= (SparseMSG_data -> norms);
   double               *rel_norms            	= (SparseMSG_data -> rel_norms);

   double                b_dot_b, r_dot_r, eps;
   double                e_dot_e, x_dot_x;
                    
   int                   i,l,lx,ly,lz;
   int			 index,index2;		                    
   int                   ierr = 0;

#if DEBUG
   char                  filename[255];
#endif

   /*-----------------------------------------------------
    * Initialize some things and deal with special cases
    *-----------------------------------------------------*/

   hypre_BeginTiming(SparseMSG_data -> time_index);

   hypre_FreeStructMatrix(A_array[0]);
   hypre_FreeStructVector(b_array[0]);
   hypre_FreeStructVector(x_array[0]);
   A_array[0] = hypre_RefStructMatrix(A);
   b_array[0] = hypre_RefStructVector(b);
   x_array[0] = hypre_RefStructVector(x);

   (SparseMSG_data -> num_iterations) = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         hypre_SetStructVectorConstantValues(x, 0.0);
      }

      hypre_EndTiming(SparseMSG_data -> time_index);
      return ierr;
   }

   /* part of convergence check */
   if (tol > 0.0)
   {
      /* eps = (tol^2)*<b,b> */
      b_dot_b = hypre_StructInnerProd(b_array[0], b_array[0]);
      eps = (tol*tol)*b_dot_b;

      /* if rhs is zero, return a zero solution */
      if (b_dot_b == 0.0)
      {
         hypre_SetStructVectorConstantValues(x, 0.0);
         if (logging > 0)
         {
            norms[0]     = 0.0;
            rel_norms[0] = 0.0;
         }

         hypre_EndTiming(SparseMSG_data -> time_index);
         return ierr;
      }
   }

   /*-----------------------------------------------------
    * Do V-cycles:
    *   For each index l, "fine" = l, "coarse" = (l+1)
    *-----------------------------------------------------*/

   for (i = 0; i < max_iter; i++)
   {
      /*--------------------------------------------------
       * Down cycle
       *--------------------------------------------------*/

      /* fine grid pre-relaxation */
      hypre_PFMGRelaxSetPreRelax(relax_data_array[0]);
      hypre_PFMGRelaxSetMaxIter(relax_data_array[0], num_pre_relax);
      hypre_PFMGRelaxSetZeroGuess(relax_data_array[0], zero_guess);
      hypre_PFMGRelax(relax_data_array[0], A_array[0], b_array[0], x_array[0]);
      zero_guess = 0;

      /* compute fine grid residual (b - Ax) */
      hypre_StructCopy(b_array[0], r_array[0]);
      hypre_StructMatvecCompute(matvec_data_array[0],
				-1.0, A_array[0], x_array[0], 1.0, r_array[0]);

      /* convergence check */
      if (tol > 0.0)
      {
         r_dot_r = hypre_StructInnerProd(r_array[0], r_array[0]);

         if (logging > 0)
         {
            norms[i] = sqrt(r_dot_r);
/*	    if (i)
	    {
               printf("Convergence rate : %f\n",(norms[i]/norms[i-1]));
	    }
*/
            if (b_dot_b > 0)
               rel_norms[i] = sqrt(r_dot_r/b_dot_b);
            else
               rel_norms[i] = 0.0;
         }

         /* always do at least 1 V-cycle */
         if ((r_dot_r < eps) && (i > 0))
         {
            if (rel_change)
            {
               if ((e_dot_e/x_dot_x) < (eps/b_dot_b))
                  break;
            }
            else
            {
               break;
            }
         }
      }

      if (total_num_levels > 1)
      {
         /* compute the restricted residuals through the jump */
         for (l = 1; l <= (jump+1); l++)
         { 
            for (lz = 0; lz <= hypre_min(l,(num_levels[2]-1)); lz++)
            {
               for (ly = 0; ly <= hypre_min(l,(num_levels[1]-1)); ly++)
               {
                  lx = l - (lz + ly);
                  
                  /* if lx is a viable index then restriction can proceed */
                  if (lx >= 0 && lx <= (num_levels[0]-1))
                     {
                     /* compute the array index */
                     hypre_SparseMSGComputeArrayIndex(lx,ly,lz,num_levels[0],num_levels[1],index);

                     /* initialize b_array */
                     hypre_SetStructVectorConstantValues(b_array[index], 0.0);
                        
                     if (lx > 0)
                     {
                        /* restrict residual from ((lx-1),ly,lz) to (lx,ly,lz) */
                        hypre_SparseMSGComputeArrayIndex((lx-1),ly,lz,
                                                         num_levels[0],num_levels[1],index2);
                        hypre_PFMGRestrict(restrict_data_array[3*index2],
                                           RT_array[3*index2],r_array[index2],tx_array[index]);
                        hypre_StructAxpy(restrict_weights[3*index], tx_array[index], b_array[index]);
#if DEBUG
                        sprintf(filename, "zoutSMSG_xdown.%02d_%02d", index2, 0);
                        hypre_PrintStructVector(filename, x_array[index2], 0);
                        sprintf(filename, "zoutSMSG_rdown.%02d_%02d", index2, 0);
                        hypre_PrintStructVector(filename, r_array[index2], 0);
                        sprintf(filename, "zoutSMSG_txdown.%02d_%02d", index, 0);
                        hypre_PrintStructVector(filename, tx_array[index], 0);
#endif
                     }
                     if (ly > 0)
                     {
                        /* restrict residual from (lx,(ly-1),lz) to (lx,ly,lz) */
                        hypre_SparseMSGComputeArrayIndex(lx,(ly-1),lz,
                                                         num_levels[0],num_levels[1],index2);
                        hypre_PFMGRestrict(restrict_data_array[(3*index2)+1],
                                           RT_array[(3*index2)+1],r_array[index2],
                                           tx_array[index]);
                        hypre_StructAxpy(restrict_weights[(3*index)+1], tx_array[index], 
                                         b_array[index]);
#if DEBUG
                        sprintf(filename, "zoutSMSG_xdown.%02d_%02d", index2, 1);
                        hypre_PrintStructVector(filename, x_array[index2], 0);
                        sprintf(filename, "zoutSMSG_rdown.%02d_%02d", index2, 1);
                        hypre_PrintStructVector(filename, r_array[index2], 0);
                        sprintf(filename, "zoutSMSG_txdown.%02d_%02d", index, 1);
                        hypre_PrintStructVector(filename, tx_array[index], 0);
#endif
                     }
                     if (lz > 0)
                     {
                        /* restrict residual from (lx,ly,(lz-1)) to (lx,ly,lz) */
                        hypre_SparseMSGComputeArrayIndex(lx,ly,(lz-1),
                                                         num_levels[0],num_levels[1],index2);
                        hypre_PFMGRestrict(restrict_data_array[(3*index2)+2],
                                                RT_array[(3*index2)+2],r_array[index2],
                                                tx_array[index]);
                        hypre_StructAxpy(restrict_weights[(3*index)+2], tx_array[index], 
                                         b_array[index]);
#if DEBUG
                        sprintf(filename, "zoutSMSG_xdown.%02d_%02d", index2, 2);
                        hypre_PrintStructVector(filename, x_array[index2], 0);
                        sprintf(filename, "zoutSMSG_rdown.%02d_%02d", index2, 2);
                        hypre_PrintStructVector(filename, r_array[index2], 0);
                        sprintf(filename, "zoutSMSG_txdown.%02d_%02d", index, 2);
                        hypre_PrintStructVector(filename, tx_array[index], 0);
#endif
                     }
#if DEBUG
                     sprintf(filename, "zoutSMSG_b.%02d", index);
                     hypre_PrintStructVector(filename, b_array[index], 0);
#endif
                        
                     if (l == (jump+1))
                     {
                        /* pre-relaxation */
                        hypre_PFMGRelaxSetPreRelax(relax_data_array[index]);
                        hypre_PFMGRelaxSetMaxIter(relax_data_array[index], num_pre_relax);
                        hypre_PFMGRelaxSetZeroGuess(relax_data_array[index], 1);
                        hypre_PFMGRelax(relax_data_array[index], A_array[index], 
                                             b_array[index], x_array[index]);
                        /* compute residual (b - Ax) */
                        hypre_StructCopy(b_array[index], r_array[index]);
                        hypre_StructMatvecCompute(matvec_data_array[index],-1.0, A_array[index], 
                                                  x_array[index], 1.0, r_array[index]);
                     }
                  }
               }
            }
         }


       	 /* complete down cycle after jump */
         for (l = (jump+2); l <= (total_num_levels-1); l++)
         { 
            for (lz = 0; lz <= hypre_min(l,(num_levels[2]-1)); lz++)
            {
               for (ly = 0; ly <= hypre_min(l,(num_levels[1]-1)); ly++)
               {
                  lx = l - (lz + ly);
                  
                  /* if lz is a viable index then restriction can proceed */
                  if (lx >= 0 && lx <= (num_levels[0]-1))
                  {
                     /* compute the array index */
                     hypre_SparseMSGComputeArrayIndex(lx,ly,lz,num_levels[0],num_levels[1],index);   

                     /* initialize b_array */
                     hypre_SetStructVectorConstantValues(b_array[index], 0.0);
                     
                     if (lx > 0)
                     {
                        /* restrict residual from ((lx-1),ly,lz) to (lx,ly,lz) */
                        hypre_SparseMSGComputeArrayIndex((lx-1),ly,lz,
                                                         num_levels[0],num_levels[1],index2);
                        hypre_PFMGRestrict(restrict_data_array[3*index2],
                                                RT_array[3*index2],r_array[index2],tx_array[index]);
                        hypre_StructAxpy(restrict_weights[3*index], tx_array[index], b_array[index]);
#if DEBUG
                        sprintf(filename, "zoutSMSG_xdown.%02d_%02d", index2, 0);
                        hypre_PrintStructVector(filename, x_array[index2], 0);
                        sprintf(filename, "zoutSMSG_rdown.%02d_%02d", index2, 0);
                        hypre_PrintStructVector(filename, r_array[index2], 0);
                        sprintf(filename, "zoutSMSG_txdown.%02d_%02d", index, 0);
                        hypre_PrintStructVector(filename, tx_array[index], 0);
#endif
                     }
                     if (ly > 0)
                     {
                        /* restrict residual from (lx,(ly-1),lz) to (lx,ly,lz) */
                        hypre_SparseMSGComputeArrayIndex(lx,(ly-1),
                                                         lz,num_levels[0],num_levels[1],index2);
                        hypre_PFMGRestrict(restrict_data_array[(3*index2)+1],
                                                RT_array[(3*index2)+1],r_array[index2],
                                                tx_array[index]);
                        hypre_StructAxpy(restrict_weights[(3*index)+1], tx_array[index], 
                                         b_array[index]);
#if DEBUG
                        sprintf(filename, "zoutSMSG_xdown.%02d_%02d", index2, 1);
                        hypre_PrintStructVector(filename, x_array[index2], 0);
                        sprintf(filename, "zoutSMSG_rdown.%02d_%02d", index2, 1);
                        hypre_PrintStructVector(filename, r_array[index2], 0);
                        sprintf(filename, "zoutSMSG_txdown.%02d_%02d", index, 1);
                        hypre_PrintStructVector(filename, tx_array[index], 0);
#endif
                     }
                     if (lz > 0)
                     {
                        /* restrict residual from (lx,ly,(lz-1)) to (lx,ly,lz) */
                        hypre_SparseMSGComputeArrayIndex(lx,ly,(lz-1),
                                                         num_levels[0],num_levels[1],index2);
                        hypre_PFMGRestrict(restrict_data_array[(3*index2)+2],
                                                RT_array[(3*index2)+2],r_array[index2],
                                                tx_array[index]);
                        hypre_StructAxpy(restrict_weights[(3*index)+2], tx_array[index], 
                                         b_array[index]);
#if DEBUG
                        sprintf(filename, "zoutSMSG_xdown.%02d_%02d", index2, 2);
                        hypre_PrintStructVector(filename, x_array[index2], 0);
                        sprintf(filename, "zoutSMSG_rdown.%02d_%02d", index2, 2);
                        hypre_PrintStructVector(filename, r_array[index2], 0);
                        sprintf(filename, "zoutSMSG_txdown.%02d_%02d", index, 2);
                        hypre_PrintStructVector(filename, tx_array[index], 0);
#endif
                     }

#if DEBUG
                        sprintf(filename, "zoutSMSG_b.%02d", index);
                        hypre_PrintStructVector(filename, b_array[index], 0);
#endif
                     if (l < (total_num_levels-1))
                     {
                        /* pre-relaxation */
                        hypre_PFMGRelaxSetPreRelax(relax_data_array[index]);
                        hypre_PFMGRelaxSetMaxIter(relax_data_array[index], num_pre_relax);
                        hypre_PFMGRelaxSetZeroGuess(relax_data_array[index], 1);
                        hypre_PFMGRelax(relax_data_array[index], A_array[index], 
                                             b_array[index], x_array[index]);
                        /* compute residual (b - Ax) */
                        hypre_StructCopy(b_array[index], r_array[index]);
                        hypre_StructMatvecCompute(matvec_data_array[index],-1.0, A_array[index], 
                                                  x_array[index], 1.0, r_array[index]);
                     }
                  }
               }
            }
         }

         /*--------------------------------------------------
          * Bottom
          *--------------------------------------------------*/
      
            hypre_PFMGRelaxSetZeroGuess(relax_data_array[index], 1);
            hypre_PFMGRelax(relax_data_array[index], A_array[index], 
                                 b_array[index], x_array[index]);

#if DEBUG
         sprintf(filename, "zoutSMSG_xbottom.%02d", index);
         hypre_PrintStructVector(filename, x_array[index], 0);
#endif

         /*--------------------------------------------------
          * Up cycle
          *--------------------------------------------------*/

         for (l = (total_num_levels - 2); l >= (jump + 1); l--)
         {
            for (lz = hypre_min(l,(num_levels[2]-1)); lz >= 0; lz--)
            {
               for (ly = hypre_min(l,(num_levels[1]-1)); ly >= 0; ly--)
               {
                  lx = l - (lz + ly);

                  if (lx >= 0 &&  lx <= (num_levels[0]-1))
                  {
                     /* compute the array index */
                     hypre_SparseMSGComputeArrayIndex(lx,ly,lz,num_levels[0],num_levels[1],index);
                     
                     if (lx < (num_levels[0]-1))
                     {
                        /* interpolate error from ((lx+1),ly,lz) to (lx,ly,lz) */
                        hypre_SparseMSGComputeArrayIndex((lx+1),ly,lz,
                                                         num_levels[0],num_levels[1],index2);
                        hypre_PFMGInterp(interp_data_array[3*index],P_array[3*index],
                                              x_array[index2],e_array[index]);
                        hypre_StructAxpy(interp_weights[3*index], e_array[index], x_array[index]);
#if DEBUG
                        sprintf(filename, "zoutSMSG_eup.%02d_%02d", index, 0);
                        hypre_PrintStructVector(filename, e_array[index], 0);
#endif
                     }
                     if (ly < (num_levels[1]-1))
                     {
                        /* interpolate error from (lx,(ly+1),lz) to (lx,ly,lz) */
                        hypre_SparseMSGComputeArrayIndex(lx,(ly+1),lz,
                                                         num_levels[0],num_levels[1],index2);
                        hypre_PFMGInterp(interp_data_array[(3*index)+1],P_array[(3*index)+1],
                                              x_array[index2],e_array[index]);
                        hypre_StructAxpy(interp_weights[(3*index)+1], e_array[index], x_array[index]);
#if DEBUG
                        sprintf(filename, "zoutSMSG_eup.%02d_%02d", index, 1);
                        hypre_PrintStructVector(filename, e_array[index], 0);
#endif
                     }
                     if (lz < (num_levels[2]-1))
                     {
                        /* interpolate error from (lx,ly,(lz+1)) to (lx,ly,lz) */
                        hypre_SparseMSGComputeArrayIndex(lx,ly,(lz+1),
                                                         num_levels[0],num_levels[1],index2);
                        hypre_PFMGInterp(interp_data_array[(3*index)+2],P_array[(3*index)+2],
                                              x_array[index2],e_array[index]);
                        hypre_StructAxpy(interp_weights[(3*index)+2], e_array[index], x_array[index]);
#if DEBUG
                        sprintf(filename, "zoutSMSG_eup.%02d_%02d", index, 2);
                        hypre_PrintStructVector(filename, e_array[index], 0);
#endif
                     }               
#if DEBUG
                     sprintf(filename, "zoutSMSG_xup.%02d", index);
                     hypre_PrintStructVector(filename, x_array[index], 0);
#endif
                     /* post-relaxation */
                     hypre_PFMGRelaxSetPostRelax(relax_data_array[index]);
                     hypre_PFMGRelaxSetMaxIter(relax_data_array[index], num_post_relax);
                     hypre_PFMGRelaxSetZeroGuess(relax_data_array[index], 0);
                     hypre_PFMGRelax(relax_data_array[index], A_array[index], b_array[index], x_array[index]);
                  }
               }
            }
         }
   
         
         /* compute the interpolated error after the jump */
          for (l = jump; l >= 0; l--)
         {
            for (lz = hypre_min(l,(num_levels[2]-1)); lz >= 0; lz--)
            {
               for (ly = hypre_min(l,(num_levels[1]-1)); ly >= 0; ly--)
               {
                  lx = l - (lz + ly);

                  if (lx >= 0 &&  lx <= (num_levels[0]-1))
                  {
                     /* compute the array index */
                     hypre_SparseMSGComputeArrayIndex(lx,ly,lz,num_levels[0],num_levels[1],index);

                     if (lx < (num_levels[0]-1))
                     {
                        /* interpolate error from ((lx+1),ly,lz) to (lx,ly,lz) */
                        hypre_SparseMSGComputeArrayIndex((lx+1),ly,lz,
                                                         num_levels[0],num_levels[1],index2);
                        hypre_PFMGInterp(interp_data_array[3*index],P_array[3*index],
                                              x_array[index2],e_array[index]);
                        hypre_StructAxpy(interp_weights[3*index], e_array[index], x_array[index]);
#if DEBUG
                        sprintf(filename, "zoutSMSG_eup.%02d_%02d", index, 0);
                        hypre_PrintStructVector(filename, e_array[index], 0);
#endif
                     }
                     if (ly < (num_levels[1]-1))
                     {
                        /* interpolate error from (lx,(ly+1),lz) to (lx,ly,lz) */
                        hypre_SparseMSGComputeArrayIndex(lx,(ly+1),lz,
                                                         num_levels[0],num_levels[1],index2);
                        hypre_PFMGInterp(interp_data_array[(3*index)+1],P_array[(3*index)+1],
                                              x_array[index2],e_array[index]);
                        hypre_StructAxpy(interp_weights[(3*index)+1], e_array[index], x_array[index]);
#if DEBUG
                        sprintf(filename, "zoutSMSG_eup.%02d_%02d", index, 1);
                        hypre_PrintStructVector(filename, e_array[index], 0);
#endif
                     }
                     if (lz < (num_levels[2]-1))
                     {
                        /* interpolate error from (lx,ly,(lz+1)) to (lx,ly,lz) */
                        hypre_SparseMSGComputeArrayIndex(lx,ly,(lz+1),
                                                         num_levels[0],num_levels[1],index2);
                        hypre_PFMGInterp(interp_data_array[(3*index)+2],P_array[(3*index)+2],
                                              x_array[index2],e_array[index]);
                        hypre_StructAxpy(interp_weights[(3*index)+2], e_array[index], x_array[index]);
#if DEBUG
                        sprintf(filename, "zoutSMSG_eup.%02d_%02d", index, 2);
                        hypre_PrintStructVector(filename, e_array[index], 0);
#endif
                     }
#if DEBUG
                     sprintf(filename, "zoutSMSG_xup.%02d", index);
                     hypre_PrintStructVector(filename, x_array[index], 0);
#endif
                  }
               }
            }
         }
      }

      /* part of convergence check */
      if ((tol > 0.0) && (rel_change))
      {
         if (total_num_levels > 1)
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
      hypre_PFMGRelaxSetPostRelax(relax_data_array[0]);
      hypre_PFMGRelaxSetMaxIter(relax_data_array[0], num_post_relax);
      hypre_PFMGRelaxSetZeroGuess(relax_data_array[0], 0);
      hypre_PFMGRelax(relax_data_array[0], A_array[0], b_array[0], x_array[0]);

      (SparseMSG_data -> num_iterations) = (i + 1);
   }

   hypre_EndTiming(SparseMSG_data -> time_index);

   return ierr;
}















