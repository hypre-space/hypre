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
 * amg_Cycle
 *--------------------------------------------------------------------------*/

int         amg_Cycle(U_array,F_array, tol, data)

Vector      **U_array;
Vector      **F_array;
double       tol;
void        *data; 
{

/* Data Structure variables */

   Matrix    **A_array;
   Matrix    **P_array;
   VectorInt **IU_array;
   VectorInt **IP_array;
   VectorInt **IV_array;
   VectorInt **ICG_array;

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
   int      *ierlx;
   int      *iurlx;
   int      *num_coeffs;
   int      *numv;
   int      *leva;
   int      *levv;
   int      *levi;
   int      *levp;
   int       num_levels;


/* Local variables  */

   int      *iarr;
   int      *ity, *iun, *ieq, *ipt;
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
   
/* Acquire data and allocate storage */

   AMGData  *amg_data = data; 

   A_array = AMGDataAArray(amg_data);
   P_array = AMGDataPArray(amg_data);
   IU_array = AMGDataIUArray(amg_data);
   IP_array = AMGDataIPArray(amg_data);
   IV_array = AMGDataIVArray(amg_data);
   ICG_array = AMGDataICGArray(amg_data);

   imin = AMGDataIMin(amg_data);
   imax = AMGDataIMax(amg_data);
   ipmn = AMGDataIPMN(amg_data);
   ipmx =  AMGDataIPMX(amg_data);
   num_levels  = AMGDataNumLevels(amg_data);
   mu = AMGDataMU(amg_data);
   ntrlx = AMGDataNTRLX(amg_data);
   iprlx = AMGDataIPRLX(amg_data);
   ierlx = AMGDataIERLX(amg_data);
   iurlx = AMGDataIURLX(amg_data);
   leva = AMGDataLevA(amg_data);
   levv = AMGDataLevV(amg_data);
   levp = AMGDataLevP(amg_data);
   levi = AMGDataLevI(amg_data);
   num_coeffs = AMGDataNumA(amg_data);
   numv = AMGDataNumV(amg_data);
   cycle_op_count = AMGDataCycleOpCount(amg_data);

   Fcycle_flag = AMGDataFcycleFlag(amg_data);
   Vstar_flag = AMGDataVstarFlag(amg_data);

   lev_counter = ctalloc(int, num_levels);
   iarr = ctalloc(int, 10);
   ity  = ctalloc(int, 10);
   ieq  = ctalloc(int, 10);
   iun  = ctalloc(int, 10);
   ipt  = ctalloc(int, 10);

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

         idec_(&iurlx[cycle_param],&num_integers,&num_digits,iun);
         if (num_digits < ii)
         {
            Solve_err_flag = 7;
            return(Solve_err_flag);
         }

         idec_(&ierlx[cycle_param],&num_integers,&num_digits,ieq);
         if (num_digits < ii)
         {
            Solve_err_flag = 7;
            return(Solve_err_flag);
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

               CALL_RELAX(Solve_err_flag, u, f, tol, amg_data);
 
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
 * Visit coarser level next.  Compute and restrict residual (RSCALI).
 * Reset counters and cycling parameters for coarse level
 *-----------------------------------------------------------------------*/

          fine_grid = level;
          coarse_grid = level + 1;

          InitVector(U_array[coarse_grid],0.0);

    
          CALL_RSCALI(coarse_grid,fine_grid,numv,F_array,U_array,amg_data);

          ++level;
          lev_counter[level] = max(lev_counter[level],mu[level-1]);
          cycle_param = 1;
          if (level == num_levels-1) cycle_param = 3;
      }

      else if (level != 0)
      {
                            
/*------------------------------------------------------------------------
 * Visit finer level next.  Interpolate and add correction (INTAD).
 * Reset counters and cycling parameters for finer level.
 *-----------------------------------------------------------------------*/

          fine_grid = level - 1;
          coarse_grid = level;

          CALL_INTAD(coarse_grid,fine_grid,numv,Vstar_flag,
                     F_array,U_array,amg_data);

          --level;
          cycle_param = 2;
          if (level == 0 && ntrlx[0] > 9) cycle_param = 0;
      }
      else
      {
             Not_Finished = 0;
      }
   }

   AMGDataCycleOpCount(amg_data) = cycle_op_count;
   return(Solve_err_flag);
}
