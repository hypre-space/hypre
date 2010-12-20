/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/





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
HYPRE_Int      *levmax;
HYPRE_Int      *data;
{
   HYPRE_AMGSetLevMax(*levmax, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setncg)(ncg, data)
HYPRE_Int      *ncg;
HYPRE_Int      *data;
{
   HYPRE_AMGSetNCG(*ncg, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setecg)(ecg, data)
double   *ecg;
HYPRE_Int      *data;
{
   HYPRE_AMGSetECG(*ecg, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setnwt)(nwt, data)
HYPRE_Int      *nwt;
HYPRE_Int      *data;
{
   HYPRE_AMGSetNWT(*nwt, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setewt)(ewt, data)
double   *ewt;
HYPRE_Int      *data;
{
   HYPRE_AMGSetEWT(*ewt, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setnstr)(nstr, data)
HYPRE_Int      *nstr;
HYPRE_Int      *data;
{
   HYPRE_AMGSetNSTR(*nstr, (void *) *data);
}

		  		      
/*--------------------------------------------------------------------------
 * Routines to set the solve phase parameters
 *--------------------------------------------------------------------------*/

void      hypre_NAME_C_FOR_FORTRAN(amg_setncyc)(ncyc, data)
HYPRE_Int      *ncyc;
HYPRE_Int      *data;
{
   HYPRE_AMGSetNCyc(*ncyc, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setmu)(mu, data)
HYPRE_Int      *mu;
HYPRE_Int      *data;
{
   HYPRE_AMGSetMU(mu, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setntrlx)(ntrlx, data)
HYPRE_Int      *ntrlx;
HYPRE_Int      *data;
{
   HYPRE_AMGSetNTRLX(ntrlx, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setiprlx)(iprlx, data)
HYPRE_Int      *iprlx;
HYPRE_Int      *data;
{
   HYPRE_AMGSetIPRLX(iprlx, (void *) *data);
}
		  		     

/*--------------------------------------------------------------------------
 * Routine to set up logging
 *--------------------------------------------------------------------------*/

void      hypre_NAME_C_FOR_FORTRAN(amg_setlogging)(ioutdat, log_file_name, data,
					     log_file_name_len)
HYPRE_Int      *ioutdat;
char     *log_file_name;
HYPRE_Int      *data;
HYPRE_Int       log_file_name_len;
{
   HYPRE_AMGSetLogging(*ioutdat,log_file_name, (void *) *data);    
}

/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

void      hypre_NAME_C_FOR_FORTRAN(amg_setnumunknowns)(num_unknowns, data)
HYPRE_Int      *num_unknowns;  
HYPRE_Int      *data;
{
   HYPRE_AMGSetNumUnknowns(*num_unknowns, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setnumpoints)(num_points, data)
HYPRE_Int      *num_points;    
HYPRE_Int      *data;
{
   HYPRE_AMGSetNumPoints(*num_points, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setiu)(iu, data)
HYPRE_Int      *iu;            
HYPRE_Int      *data;
{
   HYPRE_AMGSetIU(iu, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setip)(ip, data)
HYPRE_Int      *ip;            
HYPRE_Int      *data;
{
   HYPRE_AMGSetIP(ip, (void *) *data);
}

void      hypre_NAME_C_FOR_FORTRAN(amg_setiv)(iv, data)
HYPRE_Int      *iv;            
HYPRE_Int      *data;
{
   HYPRE_AMGSetIV(iv, (void *) *data);
}


