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
   int      max_levels;
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
   max_levels = 25;
   ncg    = 30012;
   ecg    = 0.25;
   nwt    = 200;
   ewt    = 0.35;
   nstr   = 11;

   /* solve params */
   ncyc  = 1020;
   mu = hypre_CTAlloc(int, max_levels);
   for (i = 0; i < max_levels; i++)
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

   amg_data = hypre_AMGNewData(max_levels, ncg, ecg, nwt, ewt, nstr,
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

/*--------------------------------------------------------------------------
 * Routines to set the setup phase parameters
 *--------------------------------------------------------------------------*/

void      HYPRE_AMGSetMaxLevels(max_levels, data)
int       max_levels;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataMaxLevels(amg_data) = max_levels;
}

void      HYPRE_AMGSetNCG(ncg, data)
int       ncg;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataNCG(amg_data) = ncg;
}

void      HYPRE_AMGSetECG(ecg, data)
double    ecg;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataECG(amg_data) = ecg;
}

void      HYPRE_AMGSetNWT(nwt, data)
int       nwt;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataNWT(amg_data) = nwt;
}

void      HYPRE_AMGSetEWT(ewt, data)
double    ewt;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataEWT(amg_data) = ewt;
}

void      HYPRE_AMGSetNSTR(nstr, data)
int       nstr;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataNSTR(amg_data) = nstr;
}
		  		      
/*--------------------------------------------------------------------------
 * Routines to set the solve phase parameters
 *--------------------------------------------------------------------------*/

void      HYPRE_AMGSetNCyc(ncyc, data)
int       ncyc;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataNCyc(amg_data) = ncyc;
}

void      HYPRE_AMGSetMU(mu, data)
int      *mu;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataMU(amg_data));
   hypre_AMGDataMU(amg_data) = mu;
}

void      HYPRE_AMGSetNTRLX(ntrlx, data)
int      *ntrlx;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataNTRLX(amg_data));
   hypre_AMGDataNTRLX(amg_data) = ntrlx;
}

void      HYPRE_AMGSetIPRLX(iprlx, data)
int      *iprlx;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataIPRLX(amg_data));
   hypre_AMGDataIPRLX(amg_data) = iprlx;
}

/*--------------------------------------------------------------------------
 * Routine to set up logging 
 *--------------------------------------------------------------------------*/

void      HYPRE_AMGSetLogging(ioutdat, log_file_name, data)
int       ioutdat;
char     *log_file_name;
void     *data;
{
   FILE *fp;

   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataIOutDat(amg_data) = ioutdat;
   if (ioutdat > 0)
   {
      if (*log_file_name == 0)  
         sprintf(hypre_AMGDataLogFileName(amg_data), "%s", "amg.out.log");
      else
         sprintf(hypre_AMGDataLogFileName(amg_data), "%s", log_file_name); 
       
   fp = fopen(hypre_AMGDataLogFileName(amg_data),"w");
   fclose(fp);
   }
}

/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

void      HYPRE_AMGSetNumUnknowns(num_unknowns, data)
int       num_unknowns;  
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataNumUnknowns(amg_data) = num_unknowns;
}

void      HYPRE_AMGSetNumPoints(num_points, data)
int       num_points;    
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataNumPoints(amg_data) = num_points;
}

void      HYPRE_AMGSetIU(iu, data)
int      *iu;            
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataIU(amg_data));
   hypre_AMGDataIU(amg_data) = iu;
}

void      HYPRE_AMGSetIP(ip, data)
int      *ip;            
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataIP(amg_data));
   hypre_AMGDataIP(amg_data) = ip;
}

void      HYPRE_AMGSetIV(iv, data)
int      *iv;            
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataIV(amg_data));
   hypre_AMGDataIV(amg_data) = iv;
}


