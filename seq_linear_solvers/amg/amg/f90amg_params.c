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

void      amg_setlevmax_(levmax, data)
int      *levmax;
int      *data;
{
   amg_SetLevMax(*levmax, (void *) *data);
}

void      amg_setncg_(ncg, data)
int      *ncg;
int      *data;
{
   amg_SetNCG(*ncg, (void *) *data);
}

void      amg_setecg_(ecg, data)
double   *ecg;
int      *data;
{
   amg_SetECG(*ecg, (void *) *data);
}

void      amg_setnwt_(nwt, data)
int      *nwt;
int      *data;
{
   amg_SetNWT(*nwt, (void *) *data);
}

void      amg_setewt_(ewt, data)
double   *ewt;
int      *data;
{
   amg_SetEWT(*ewt, (void *) *data);
}

void      amg_setnstr_(nstr, data)
int      *nstr;
int      *data;
{
   amg_SetNSTR(*nstr, (void *) *data);
}

		  		      
/*--------------------------------------------------------------------------
 * Routines to set the solve phase parameters
 *--------------------------------------------------------------------------*/

void      amg_setncyc_(ncyc, data)
int      *ncyc;
int      *data;
{
   amg_SetNCyc(*ncyc, (void *) *data);
}

void      amg_setmu_(mu, data)
int      *mu;
int      *data;
{
   amg_SetMU(mu, (void *) *data);
}

void      amg_setntrlx_(ntrlx, data)
int      *ntrlx;
int      *data;
{
   amg_SetNTRLX(ntrlx, (void *) *data);
}

void      amg_setiprlx_(iprlx, data)
int      *iprlx;
int      *data;
{
   amg_SetIPRLX(iprlx, (void *) *data);
}

void      amg_setierlx_(ierlx, data)
int      *ierlx;
int      *data;
{
   amg_SetIERLX(ierlx, (void *) *data);
}

void      amg_setiurlx_(iurlx, data)
int      *iurlx;
int      *data;
{
   amg_SetIURLX(iurlx, (void *) *data);
}

		  		     

/*--------------------------------------------------------------------------
 * Routine to set up logging
 *--------------------------------------------------------------------------*/

void      amg_setlogging_(ioutdat, log_file_name, data, log_file_name_len)
int      *ioutdat;
char     *log_file_name;
int      *data;
int       log_file_name_len;
{
   amg_SetLogging(*ioutdat,log_file_name, (void *) *data);    
}

/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

void      amg_setnumunknowns_(num_unknowns, data)
int      *num_unknowns;  
int      *data;
{
   amg_SetNumUnknowns(*num_unknowns, (void *) *data);
}

void      amg_setnumpoints_(num_points, data)
int      *num_points;    
int      *data;
{
   amg_SetNumPoints(*num_points, (void *) *data);
}

void      amg_setiu_(iu, data)
int      *iu;            
int      *data;
{
   amg_SetIU(iu, (void *) *data);
}

void      amg_setip_(ip, data)
int      *ip;            
int      *data;
{
   amg_SetIP(ip, (void *) *data);
}

void      amg_setiv_(iv, data)
int      *iv;            
int      *data;
{
   amg_SetIV(iv, (void *) *data);
}

void      amg_setxp_(xp, data)
double   *xp;            
int      *data;
{
   amg_SetXP(xp, (void *) *data);
}

void      amg_setyp_(yp, data)
double   *yp;            
int      *data;
{
   amg_SetYP(yp, (void *) *data);
}

void      amg_setzp_(zp, data)
double   *zp;            
int      *data;
{
   amg_SetZP(zp, (void *) *data);
}

