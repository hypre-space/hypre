/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * AMG cycling routine
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * hypre_AMGCycle
 *--------------------------------------------------------------------------*/

int         hypre_AMGCycle(U_array,F_array, tol, data)

hypre_Vector      **U_array;
hypre_Vector      **F_array;
double       tol;
void        *data; 
{

/* Data Structure variables */

   hypre_Matrix    **A_array;
   hypre_Matrix    **P_array;
   hypre_VectorInt **IU_array;
   hypre_VectorInt **IP_array;
   hypre_VectorInt **IV_array;
   hypre_VectorInt **ICG_array;
   hypre_Vector    *Vtemp;

   int       Fcycle_flag;
   int       Vstar_flag;
   int       cycle_op_count;   
   int      *imin;
   int      *imax;
   int      *ipmn;
   int      *ipmx;
   int      *mu;
   int      *ntrlx;
   int      *iprlx;
   int      *num_coeffs;
   int      *numv;
   int       num_levels;
   int       num_unknowns;


/* Local variables  */

   int      *iarr;
   int      *ity, *ipt;
   int      *lev_counter;
   int       Solve_err_flag;
   int       k;
   int       j;
   int       ii;
   int       level;
   int       cycle_param;
   int       coarse_grid;
   int       fine_grid;
   int       Not_Finished;
   int       num_integers;
   int       num_digits;
   int       num_sweep;
   int       base_lev;

   double    alpha;
   double    beta;
   double   *D_mat;
   double   *S_vec;
   
/* Acquire data and allocate storage */

   hypre_AMGData  *amg_data = data; 

   A_array = hypre_AMGDataAArray(amg_data);
   P_array = hypre_AMGDataPArray(amg_data);
   IU_array = hypre_AMGDataIUArray(amg_data);
   IP_array = hypre_AMGDataIPArray(amg_data);
   IV_array = hypre_AMGDataIVArray(amg_data);
   ICG_array = hypre_AMGDataICGArray(amg_data);
   Vtemp = hypre_AMGDataVtemp(amg_data);

   imin = hypre_AMGDataIMin(amg_data);
   imax = hypre_AMGDataIMax(amg_data);
   ipmn = hypre_AMGDataIPMN(amg_data);
   ipmx =  hypre_AMGDataIPMX(amg_data);
   num_levels  = hypre_AMGDataNumLevels(amg_data);
   mu = hypre_AMGDataMU(amg_data);
   ntrlx = hypre_AMGDataNTRLX(amg_data);
   iprlx = hypre_AMGDataIPRLX(amg_data);
   num_coeffs = hypre_AMGDataNumA(amg_data);
   num_unknowns = hypre_AMGDataNumUnknowns(amg_data);
   numv = hypre_AMGDataNumV(amg_data);
   cycle_op_count = hypre_AMGDataCycleOpCount(amg_data);

   Fcycle_flag = hypre_AMGDataFcycleFlag(amg_data);
   Vstar_flag = hypre_AMGDataVstarFlag(amg_data);

   lev_counter = hypre_CTAlloc(int, num_levels);
   iarr = hypre_CTAlloc(int, 10);
   ity  = hypre_CTAlloc(int, 10);
   ipt  = hypre_CTAlloc(int, 10);

   D_mat = hypre_CTAlloc(double, num_unknowns * num_unknowns);
   S_vec = hypre_CTAlloc(double, num_unknowns);

/* Initialize */

   Solve_err_flag = 0;
   base_lev = 0;             /* Will be removed eventually */
   num_integers = 9;         /* Used in all idec calls */

/*------------------------------------------------------------------------
 *    Initialize cycling control counter
 *
 *     Cycling is controlled using a level counter: lev_counter[k]
 *     
 *     Each time relaxation is performed on level k, the
 *     counter is decremented by 1. If the counter is then
 *     negative, we go to the next finer level. If non-
 *     negative, we go to the next coarser level. The
 *     following actions control cycling:
 *     
 *     a. lev_counter[0] is initialized to 1.
 *     b. lev_counter[k] is initialized to mu[k-1]+Fcycle_flag for k>0.
 *     
 *     c. During cycling, when going down to level k,
 *        lev_counter[k] is set to the max of (lev_counter[k],mu[k-1])
*------------------------------------------------------------------------*/

   Not_Finished = 1;

   lev_counter[0] = 1;
   for (k = 1; k < num_levels; ++k) 
   {
       lev_counter[k] = mu[k-1] + Fcycle_flag;
   }

/*------------------------------------------------------------------------
 * Set the initial cycling parameters
 *
 *     1. ntrlx[0] can have several meanings (nr1,nr2)
 *     nr1 defines the first fine grid sweep
 *     nr2 defines any subsequent sweeps
 *     
 *     ntrlx[0] = 0   - (0,0)
 *     ntrlx[0] = 1   - (ntrlx[1],ntrlx[2])
 *     ntrlx[0] > 9   - standard meaning
 *-----------------------------------------------------------------------*/
   
   level = 0;
   cycle_param = 0;

   if (ntrlx[cycle_param] == 1 || ntrlx[cycle_param] == 2) cycle_param = 1;

/*------------------------------------------------------------------------
 * Main loop of cycling
 *-----------------------------------------------------------------------*/
  
   while (Not_Finished)
   {
      if (ntrlx[cycle_param] > 9)
      {

/*------------------------------------------------------------------------
 * Decode relaxation parameters. error flag set to 7 if error occurs
 *-----------------------------------------------------------------------*/
 
 
         idec_(&ntrlx[cycle_param],&num_integers,&num_digits,iarr);
         num_sweep = iarr[0];
         ii = 0;
         for (k = 1; k < num_digits; k++)
         {
             if (iarr[k] == 0) num_sweep = num_sweep * 10;
             else ity[ii++] = iarr[k];
         }

         idec_(&iprlx[cycle_param],&num_integers,&num_digits,ipt);
         if (num_digits < ii)
         {
            Solve_err_flag = 7;
            return(Solve_err_flag);
         }

 
/*------------------------------------------------------------------------
 * Do the relaxation num_sweep times, looping over partial sweeps.
 *-----------------------------------------------------------------------*/

         for (j = 0; j < num_sweep; j++)
         {
             cycle_op_count += num_coeffs[level];

             for (k = 0; k < ii; k++) 
             {
               Solve_err_flag = hypre_AMGRelax(U_array[level],F_array[level],
                                          A_array[level], ICG_array[level],
                                          IV_array[level],ipmn[level],
                                          ipmx[level],ipt[k],ity[k],
                                          D_mat,S_vec);

               if (Solve_err_flag != 0) return(Solve_err_flag);

             }
         }


/*------------------------------------------------------------------------
 * Decrement the control counter and determine which grid to visit next
 *-----------------------------------------------------------------------*/

      }

      --lev_counter[level];
       
      if (lev_counter[level] >= 0 && level != num_levels-1)
      {
                               
/*------------------------------------------------------------------------
 * Visit coarser level next.  Compute residual using hypre_Matvec.
 * Perform restriction using hypre_MatvecT.
 * Reset counters and cycling parameters for coarse level
 *-----------------------------------------------------------------------*/

          fine_grid = level;
          coarse_grid = level + 1;

          hypre_InitVector(U_array[coarse_grid],0.0);
          
          hypre_CopyVector(F_array[fine_grid],Vtemp);
          alpha = -1.0;
          beta = 1.0;
          hypre_Matvec(alpha, A_array[fine_grid], U_array[fine_grid],
                 beta, Vtemp);

          alpha = 1.0;
          beta = 0.0;
          hypre_MatvecT(alpha,P_array[fine_grid],Vtemp,
                  beta,F_array[coarse_grid]);

          ++level;
          lev_counter[level] = max(lev_counter[level],mu[level-1]);
          cycle_param = 1;
          if (level == num_levels-1) cycle_param = 3;
      }

      else if (level != 0)
      {
                            
/*------------------------------------------------------------------------
 * Visit finer level next.  Interpolate and add correction using hypre_Matvec.
 * Reset counters and cycling parameters for finer level.
 *-----------------------------------------------------------------------*/

          fine_grid = level - 1;
          coarse_grid = level;
          alpha = 1.0;
          beta = 1.0;

          hypre_Matvec(alpha, P_array[fine_grid], U_array[coarse_grid],
                 beta, U_array[fine_grid]);            

          --level;
          cycle_param = 2;
          if (level == 0 && ntrlx[0] > 9) cycle_param = 0;
      }
      else
      {
             Not_Finished = 0;
      }
   }

   hypre_AMGDataCycleOpCount(amg_data) = cycle_op_count;

   hypre_TFree(lev_counter);
   hypre_TFree(iarr); 
   hypre_TFree(ity);  
   hypre_TFree(ipt); 

   hypre_TFree(D_mat);
   hypre_TFree(S_vec); 

   return(Solve_err_flag);
}
