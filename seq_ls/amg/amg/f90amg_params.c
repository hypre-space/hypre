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

void      NAME_C_FOR_FORTRAN(amg_setlevmax)(levmax, data)
int      *levmax;
int      *data;
{
   amg_SetLevMax(*levmax, (void *) *data);
}

void      NAME_C_FOR_FORTRAN(amg_setncg)(ncg, data)
int      *ncg;
int      *data;
{
   amg_SetNCG(*ncg, (void *) *data);
}

void      NAME_C_FOR_FORTRAN(amg_setecg)(ecg, data)
double   *ecg;
int      *data;
{
   amg_SetECG(*ecg, (void *) *data);
}

void      NAME_C_FOR_FORTRAN(amg_setnwt)(nwt, data)
int      *nwt;
int      *data;
{
   amg_SetNWT(*nwt, (void *) *data);
}

void      NAME_C_FOR_FORTRAN(amg_setewt)(ewt, data)
double   *ewt;
int      *data;
{
   amg_SetEWT(*ewt, (void *) *data);
}

void      NAME_C_FOR_FORTRAN(amg_setnstr)(nstr, data)
int      *nstr;
int      *data;
{
   amg_SetNSTR(*nstr, (void *) *data);
}

		  		      
/*--------------------------------------------------------------------------
 * Routines to set the solve phase parameters
 *--------------------------------------------------------------------------*/

void      NAME_C_FOR_FORTRAN(amg_setncyc)(ncyc, data)
int      *ncyc;
int      *data;
{
   amg_SetNCyc(*ncyc, (void *) *data);
}

void      NAME_C_FOR_FORTRAN(amg_setmu)(mu, data)
int      *mu;
int      *data;
{
   amg_SetMU(mu, (void *) *data);
}

void      NAME_C_FOR_FORTRAN(amg_setntrlx)(ntrlx, data)
int      *ntrlx;
int      *data;
{
   amg_SetNTRLX(ntrlx, (void *) *data);
}

void      NAME_C_FOR_FORTRAN(amg_setiprlx)(iprlx, data)
int      *iprlx;
int      *data;
{
   amg_SetIPRLX(iprlx, (void *) *data);
}
		  		     

/*--------------------------------------------------------------------------
 * Routine to set up logging
 *--------------------------------------------------------------------------*/

void      NAME_C_FOR_FORTRAN(amg_setlogging)(ioutdat, log_file_name, data,
					     log_file_name_len)
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

void      NAME_C_FOR_FORTRAN(amg_setnumunknowns)(num_unknowns, data)
int      *num_unknowns;  
int      *data;
{
   amg_SetNumUnknowns(*num_unknowns, (void *) *data);
}

void      NAME_C_FOR_FORTRAN(amg_setnumpoints)(num_points, data)
int      *num_points;    
int      *data;
{
   amg_SetNumPoints(*num_points, (void *) *data);
}

void      NAME_C_FOR_FORTRAN(amg_setiu)(iu, data)
int      *iu;            
int      *data;
{
   amg_SetIU(iu, (void *) *data);
}

void      NAME_C_FOR_FORTRAN(amg_setip)(ip, data)
int      *ip;            
int      *data;
{
   amg_SetIP(ip, (void *) *data);
}

void      NAME_C_FOR_FORTRAN(amg_setiv)(iv, data)
int      *iv;            
int      *data;
{
   amg_SetIV(iv, (void *) *data);
}

void      NAME_C_FOR_FORTRAN(amg_setxp)(xp, data)
double   *xp;            
int      *data;
{
   amg_SetXP(xp, (void *) *data);
}

void      NAME_C_FOR_FORTRAN(amg_setyp)(yp, data)
double   *yp;            
int      *data;
{
   amg_SetYP(yp, (void *) *data);
}

void      NAME_C_FOR_FORTRAN(amg_setzp)(zp, data)
double   *zp;            
int      *data;
{
   amg_SetZP(zp, (void *) *data);
}

