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
 * AMG parameter functions
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * Routines to set the setup phase parameters
 *--------------------------------------------------------------------------*/

void      amg_SetLevMax(levmax, data)
int       levmax;
void     *data;
{
   AMGData  *amg_data = data;
 
   AMGDataLevMax(amg_data) = levmax;
}

void      amg_SetNCG(ncg, data)
int       ncg;
void     *data;
{
   AMGData  *amg_data = data;
 
   AMGDataNCG(amg_data) = ncg;
}

void      amg_SetECG(ecg, data)
double    ecg;
void     *data;
{
   AMGData  *amg_data = data;
 
   AMGDataECG(amg_data) = ecg;
}

void      amg_SetNWT(nwt, data)
int       nwt;
void     *data;
{
   AMGData  *amg_data = data;
 
   AMGDataNWT(amg_data) = nwt;
}

void      amg_SetEWT(ewt, data)
double    ewt;
void     *data;
{
   AMGData  *amg_data = data;
 
   AMGDataEWT(amg_data) = ewt;
}

void      amg_SetNSTR(nstr, data)
int       nstr;
void     *data;
{
   AMGData  *amg_data = data;
 
   AMGDataNSTR(amg_data) = nstr;
}

		  		      
/*--------------------------------------------------------------------------
 * Routines to set the solve phase parameters
 *--------------------------------------------------------------------------*/

void      amg_SetNCyc(ncyc, data)
int       ncyc;
void     *data;
{
   AMGData  *amg_data = data;
 
   AMGDataNCyc(amg_data) = ncyc;
}

void      amg_SetMU(mu, data)
int      *mu;
void     *data;
{
   AMGData  *amg_data = data;
 
   tfree(AMGDataMU(amg_data));
   AMGDataMU(amg_data) = mu;
}

void      amg_SetNTRLX(ntrlx, data)
int      *ntrlx;
void     *data;
{
   AMGData  *amg_data = data;
 
   tfree(AMGDataNTRLX(amg_data));
   AMGDataNTRLX(amg_data) = ntrlx;
}

void      amg_SetIPRLX(iprlx, data)
int      *iprlx;
void     *data;
{
   AMGData  *amg_data = data;
 
   tfree(AMGDataIPRLX(amg_data));
   AMGDataIPRLX(amg_data) = iprlx;
}

void      amg_SetIERLX(ierlx, data)
int      *ierlx;
void     *data;
{
   AMGData  *amg_data = data;
 
   tfree(AMGDataIERLX(amg_data));
   AMGDataIERLX(amg_data) = ierlx;
}

void      amg_SetIURLX(iurlx, data)
int      *iurlx;
void     *data;
{
   AMGData  *amg_data = data;
 
   tfree(AMGDataIURLX(amg_data));
   AMGDataIURLX(amg_data) = iurlx;
}

		  		      
/*--------------------------------------------------------------------------
 * Routines to set the output parameters
 *--------------------------------------------------------------------------*/

void      amg_SetIOutDat(ioutdat, data)
int       ioutdat;
void     *data;
{
   AMGData  *amg_data = data;
 
   AMGDataIOutDat(amg_data) = ioutdat;
}


				      
/*--------------------------------------------------------------------------
 * Routine to set the log file name
 *--------------------------------------------------------------------------*/

void      amg_SetLogFileName(log_file_name, data)
char     *log_file_name;
void     *data;
{
   AMGData  *amg_data = data;
 
   sprintf(AMGDataLogFileName(amg_data), "%s", log_file_name);
}


/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

void      amg_SetNumUnknowns(num_unknowns, data)
int       num_unknowns;  
void     *data;
{
   AMGData  *amg_data = data;
 
   AMGDataNumUnknowns(amg_data) = num_unknowns;
}

void      amg_SetNumPoints(num_points, data)
int       num_points;    
void     *data;
{
   AMGData  *amg_data = data;
 
   AMGDataNumPoints(amg_data) = num_points;
}

void      amg_SetIU(iu, data)
int      *iu;            
void     *data;
{
   AMGData  *amg_data = data;
 
   tfree(AMGDataIU(amg_data));
   AMGDataIU(amg_data) = iu;
}

void      amg_SetIP(ip, data)
int      *ip;            
void     *data;
{
   AMGData  *amg_data = data;
 
   tfree(AMGDataIP(amg_data));
   AMGDataIP(amg_data) = ip;
}

void      amg_SetIV(iv, data)
int      *iv;            
void     *data;
{
   AMGData  *amg_data = data;
 
   tfree(AMGDataIV(amg_data));
   AMGDataIV(amg_data) = iv;
}

void      amg_SetXP(xp, data)
double   *xp;            
void     *data;
{
   AMGData  *amg_data = data;
 
   tfree(AMGDataXP(amg_data));
   AMGDataXP(amg_data) = xp;
}

void      amg_SetYP(yp, data)
double   *yp;            
void     *data;
{
   AMGData  *amg_data = data;
 
   tfree(AMGDataYP(amg_data));
   AMGDataYP(amg_data) = yp;
}

void      amg_SetZP(zp, data)
double   *zp;            
void     *data;
{
   AMGData  *amg_data = data;
 
   tfree(AMGDataZP(amg_data));
   AMGDataZP(amg_data) = zp;
}

