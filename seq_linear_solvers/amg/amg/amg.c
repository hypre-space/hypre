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
 * amg_Initialize
 *--------------------------------------------------------------------------*/

void  *amg_Initialize(port_data)
void  *port_data;
{
   AMGData  *amg_data;
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
   int     *ierlx;
   int     *iurlx;

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
   mu = ctalloc(int, levmax);
   for (i = 0; i < levmax; i++)
      mu[i] = 1;
   ntrlx = ctalloc(int, 4);
   ntrlx[0] = 133;
   ntrlx[1] = 133;
   ntrlx[2] = 133;
   ntrlx[3] = 19;
   iprlx = ctalloc(int, 4);
   iprlx[0] = 31;
   iprlx[1] = 31;
   iprlx[2] = 13;
   iprlx[3] = 2;
   ierlx = ctalloc(int, 4);
   ierlx[0] = 99;
   ierlx[1] = 99;
   ierlx[2] = 99;
   ierlx[3] = 9;
   iurlx = ctalloc(int, 4);
   iurlx[0] = 99;
   iurlx[1] = 99;
   iurlx[2] = 99;
   iurlx[3] = 9;

   /* output params */
   ioutdat = 0;
   cycle_op_count = 0;


   /*-----------------------------------------------------------------------
    * Create the AMGData structure and return
    *-----------------------------------------------------------------------*/

   amg_data = amg_NewData(levmax, ncg, ecg, nwt, ewt, nstr,
			  ncyc, mu, ntrlx, iprlx, ierlx, iurlx,
			  ioutdat, cycle_op_count,
			  "amg.out.log"); 

   
   return (void *)amg_data;
}

/*--------------------------------------------------------------------------
 * amg_Finalize
 *--------------------------------------------------------------------------*/

void   amg_Finalize(data)
void  *data;
{
   AMGData  *amg_data = data;


   amg_FreeData(amg_data);
}

