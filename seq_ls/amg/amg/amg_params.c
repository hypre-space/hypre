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
 * AMG parameter functions
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * Routines to set the setup phase parameters
 *--------------------------------------------------------------------------*/

void      HYPRE_AMGSetLevMax(levmax, data)
HYPRE_Int       levmax;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataLevMax(amg_data) = levmax;
}

void      HYPRE_AMGSetNCG(ncg, data)
HYPRE_Int       ncg;
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
HYPRE_Int       nwt;
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
HYPRE_Int       nstr;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataNSTR(amg_data) = nstr;
}

		  		      
/*--------------------------------------------------------------------------
 * Routines to set the solve phase parameters
 *--------------------------------------------------------------------------*/

void      HYPRE_AMGSetNCyc(ncyc, data)
HYPRE_Int       ncyc;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataNCyc(amg_data) = ncyc;
}

void      HYPRE_AMGSetMU(mu, data)
HYPRE_Int      *mu;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataMU(amg_data));
   hypre_AMGDataMU(amg_data) = mu;
}

void      HYPRE_AMGSetNTRLX(ntrlx, data)
HYPRE_Int      *ntrlx;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataNTRLX(amg_data));
   hypre_AMGDataNTRLX(amg_data) = ntrlx;
}

void      HYPRE_AMGSetIPRLX(iprlx, data)
HYPRE_Int      *iprlx;
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
HYPRE_Int       ioutdat;
char     *log_file_name;
void     *data;
{
   FILE *fp;

   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataIOutDat(amg_data) = ioutdat;
   if (ioutdat > 0)
   {
      if (*log_file_name == 0)  
         hypre_sprintf(hypre_AMGDataLogFileName(amg_data), "%s", "amg.out.log");
      else
         hypre_sprintf(hypre_AMGDataLogFileName(amg_data), "%s", log_file_name); 
       
   fp = fopen(hypre_AMGDataLogFileName(amg_data),"w");
   fclose(fp);
   }
}


/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

void      HYPRE_AMGSetNumUnknowns(num_unknowns, data)
HYPRE_Int       num_unknowns;  
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataNumUnknowns(amg_data) = num_unknowns;
}

void      HYPRE_AMGSetNumPoints(num_points, data)
HYPRE_Int       num_points;    
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataNumPoints(amg_data) = num_points;
}

void      HYPRE_AMGSetIU(iu, data)
HYPRE_Int      *iu;            
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataIU(amg_data));
   hypre_AMGDataIU(amg_data) = iu;
}

void      HYPRE_AMGSetIP(ip, data)
HYPRE_Int      *ip;            
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataIP(amg_data));
   hypre_AMGDataIP(amg_data) = ip;
}

void      HYPRE_AMGSetIV(iv, data)
HYPRE_Int      *iv;            
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataIV(amg_data));
   hypre_AMGDataIV(amg_data) = iv;
}


