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

void      hypre_NAME_C_FOR_FORTRAN(amg_setlevmax)(levmax, data)
int      *levmax;
int      *data;
{
   HYPRE_AMGSetLevMax(*levmax, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setncg)(ncg, data)
int      *ncg;
int      *data;
{
   HYPRE_AMGSetNCG(*ncg, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setecg)(ecg, data)
double   *ecg;
int      *data;
{
   HYPRE_AMGSetECG(*ecg, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setnwt)(nwt, data)
int      *nwt;
int      *data;
{
   HYPRE_AMGSetNWT(*nwt, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setewt)(ewt, data)
double   *ewt;
int      *data;
{
   HYPRE_AMGSetEWT(*ewt, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setnstr)(nstr, data)
int      *nstr;
int      *data;
{
   HYPRE_AMGSetNSTR(*nstr, (void *) *data);
}

		  		      
/*--------------------------------------------------------------------------
 * Routines to set the solve phase parameters
 *--------------------------------------------------------------------------*/

void      hypre_NAME_C_FOR_FORTRAN(amg_setncyc)(ncyc, data)
int      *ncyc;
int      *data;
{
   HYPRE_AMGSetNCyc(*ncyc, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setmu)(mu, data)
int      *mu;
int      *data;
{
   HYPRE_AMGSetMU(mu, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setntrlx)(ntrlx, data)
int      *ntrlx;
int      *data;
{
   HYPRE_AMGSetNTRLX(ntrlx, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setiprlx)(iprlx, data)
int      *iprlx;
int      *data;
{
   HYPRE_AMGSetIPRLX(iprlx, (void *) *data);
}
		  		     

/*--------------------------------------------------------------------------
 * Routine to set up logging
 *--------------------------------------------------------------------------*/

void      hypre_NAME_C_FOR_FORTRAN(amg_setlogging)(ioutdat, log_file_name, data,
					     log_file_name_len)
int      *ioutdat;
char     *log_file_name;
int      *data;
int       log_file_name_len;
{
   HYPRE_AMGSetLogging(*ioutdat,log_file_name, (void *) *data);    
}

/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

void      hypre_NAME_C_FOR_FORTRAN(amg_setnumunknowns)(num_unknowns, data)
int      *num_unknowns;  
int      *data;
{
   HYPRE_AMGSetNumUnknowns(*num_unknowns, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setnumpoints)(num_points, data)
int      *num_points;    
int      *data;
{
   HYPRE_AMGSetNumPoints(*num_points, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setiu)(iu, data)
int      *iu;            
int      *data;
{
   HYPRE_AMGSetIU(iu, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setip)(ip, data)
int      *ip;            
int      *data;
{
   HYPRE_AMGSetIP(ip, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setiv)(iv, data)
int      *iv;            
int      *data;
{
   HYPRE_AMGSetIV(iv, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setxp)(xp, data)
double   *xp;            
int      *data;
{
   HYPRE_AMGSetXP(xp, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setyp)(yp, data)
double   *yp;            
int      *data;
{
   HYPRE_AMGSetYP(yp, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setzp)(zp, data)
double   *zp;            
int      *data;
{
   HYPRE_AMGSetZP(zp, (void *) *data);
}

