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
 * AMG functions
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * HYPRE_AMGInitialize
 *--------------------------------------------------------------------------*/

void  *HYPRE_AMGInitialize(port_data)
void  *port_data;
{
   hypre_AMGData  *amg_data;
   FILE     *fp;

   /* setup params */
   int      levmax;
   int      ncg;
   double   ecg;
   int      nwt;
   double   ewt;
   int      nstr;

   /* solve params */
   int      ncyc;
   int     *mu;
   int     *ntrlx;
   int     *iprlx;

   /* output params */
   int      ioutdat;
   int      cycle_op_count;

   /* log file name */
   char    *log_file_name;

   int      i;


   /*-----------------------------------------------------------------------
    * Setup default values for parameters
    *-----------------------------------------------------------------------*/

   /* setup params */
   levmax = 25;
   ncg    = 30012;
   ecg    = 0.25;
   nwt    = 200;
   ewt    = 0.35;
   nstr   = 11;

   /* solve params */
   ncyc  = 1020;
   mu = hypre_CTAlloc(int, levmax);
   for (i = 0; i < levmax; i++)
      mu[i] = 1;
   ntrlx = hypre_CTAlloc(int, 4);
   ntrlx[0] = 133;
   ntrlx[1] = 133;
   ntrlx[2] = 133;
   ntrlx[3] = 19;
   iprlx = hypre_CTAlloc(int, 4);
   iprlx[0] = 31;
   iprlx[1] = 31;
   iprlx[2] = 13;
   iprlx[3] = 2;

   /* output params */
   ioutdat = 0;
   cycle_op_count = 0;


   /*-----------------------------------------------------------------------
    * Create the hypre_AMGData structure and return
    *-----------------------------------------------------------------------*/

   amg_data = hypre_AMGNewData(levmax, ncg, ecg, nwt, ewt, nstr,
			  ncyc, mu, ntrlx, iprlx, 
			  ioutdat, cycle_op_count,
			  "amg.out.log"); 

   
   return (void *)amg_data;
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGFinalize
 *--------------------------------------------------------------------------*/

void   HYPRE_AMGFinalize(data)
void  *data;
{
   hypre_AMGData  *amg_data = data;


   hypre_AMGFreeData(amg_data);
}

