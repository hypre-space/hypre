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
 * AMG parameter functions (Fortran 90 interface)
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * Routines to set the setup phase parameters
 *--------------------------------------------------------------------------*/

void      amg_SetLevMax_(levmax, data)
int      *levmax;
int      *data;
{
   amg_SetLevMax(*levmax, (void *) *data);
}

void      amg_SetNCG_(ncg, data)
int      *ncg;
int      *data;
{
   amg_SetNCG(*ncg, (void *) *data);
}

void      amg_SetECG_(ecg, data)
double   *ecg;
int      *data;
{
   amg_SetECG(*ecg, (void *) *data);
}

void      amg_SetNWT_(nwt, data)
int      *nwt;
int      *data;
{
   amg_SetNWT(*nwt, (void *) *data);
}

void      amg_SetEWT_(ewt, data)
double   *ewt;
int      *data;
{
   amg_SetEWT(*ewt, (void *) *data);
}

void      amg_SetNSTR_(nstr, data)
int      *nstr;
int      *data;
{
   amg_SetNSTR(*nstr, (void *) *data);
}

		  		      
/*--------------------------------------------------------------------------
 * Routines to set the solve phase parameters
 *--------------------------------------------------------------------------*/

void      amg_SetNCyc_(ncyc, data)
int      *ncyc;
int      *data;
{
   amg_SetNCyc(*ncyc, (void *) *data);
}

void      amg_SetMU_(mu, data)
int      *mu;
int      *data;
{
   amg_SetMU(mu, (void *) *data);
}

void      amg_SetNTRLX_(ntrlx, data)
int      *ntrlx;
int      *data;
{
   amg_SetNTRLX(ntrlx, (void *) *data);
}

void      amg_SetIPRLX_(iprlx, data)
int      *iprlx;
int      *data;
{
   amg_SetIPRLX(iprlx, (void *) *data);
}

void      amg_SetIERLX_(ierlx, data)
int      *ierlx;
int      *data;
{
   amg_SetIERLX(ierlx, (void *) *data);
}

void      amg_SetIURLX_(iurlx, data)
int      *iurlx;
int      *data;
{
   amg_SetIURLX(iurlx, (void *) *data);
}

		  		      
/*--------------------------------------------------------------------------
 * Routines to set the output parameters
 *--------------------------------------------------------------------------*/

void      amg_SetIOutDat_(ioutdat, data)
int      *ioutdat;
int      *data;
{
   amg_SetIOutDat(*ioutdat, (void *) *data);
}

void      amg_SetIOutGrd_(ioutgrd, data)
int      *ioutgrd;
int      *data;
{
   amg_SetIOutGrd(*ioutgrd, (void *) *data);
}

void      amg_SetIOutMat_(ioutmat, data)
int      *ioutmat;
int      *data;
{
   amg_SetIOutMat(*ioutmat, (void *) *data);
}

void      amg_SetIOutRes_(ioutres, data)
int      *ioutres;
int      *data;
{
   amg_SetIOutRes(*ioutres, (void *) *data);
}

void      amg_SetIOutSol_(ioutsol, data)
int      *ioutsol;
int      *data;
{
   amg_SetIOutSol(*ioutsol, (void *) *data);
}

				      
/*--------------------------------------------------------------------------
 * Routine to set the log file name
 *--------------------------------------------------------------------------*/

void      amg_SetLogFileName_(log_file_name, data, log_file_name_len)
char     *log_file_name;
int      *data;
int       log_file_name_len;
{
   amg_SetLogFileName(log_file_name, (void *) *data);
}


/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

void      amg_SetNumUnknowns_(num_unknowns, data)
int      *num_unknowns;  
int      *data;
{
   amg_SetNumUnknowns(*num_unknowns, (void *) *data);
}

void      amg_SetNumPoints_(num_points, data)
int      *num_points;    
int      *data;
{
   amg_SetNumPoints(*num_points, (void *) *data);
}

void      amg_SetIU_(iu, data)
int      *iu;            
int      *data;
{
   amg_SetIU(iu, (void *) *data);
}

void      amg_SetIP_(ip, data)
int      *ip;            
int      *data;
{
   amg_SetIP(ip, (void *) *data);
}

void      amg_SetIV_(iv, data)
int      *iv;            
int      *data;
{
   amg_SetIV(iv, (void *) *data);
}

void      amg_SetXP_(xp, data)
double   *xp;            
int      *data;
{
   amg_SetXP(xp, (void *) *data);
}

void      amg_SetYP_(yp, data)
double   *yp;            
int      *data;
{
   amg_SetYP(yp, (void *) *data);
}

void      amg_SetZP_(zp, data)
double   *zp;            
int      *data;
{
   amg_SetZP(zp, (void *) *data);
}

