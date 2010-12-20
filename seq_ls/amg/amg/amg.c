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
   HYPRE_Int      levmax;
   HYPRE_Int      ncg;
   double   ecg;
   HYPRE_Int      nwt;
   double   ewt;
   HYPRE_Int      nstr;

   /* solve params */
   HYPRE_Int      ncyc;
   HYPRE_Int     *mu;
   HYPRE_Int     *ntrlx;
   HYPRE_Int     *iprlx;

   /* output params */
   HYPRE_Int      ioutdat;
   HYPRE_Int      cycle_op_count;

   /* log file name */
   char    *log_file_name;

   HYPRE_Int      i;


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
   mu = hypre_CTAlloc(HYPRE_Int, levmax);
   for (i = 0; i < levmax; i++)
      mu[i] = 1;
   ntrlx = hypre_CTAlloc(HYPRE_Int, 4);
   ntrlx[0] = 133;
   ntrlx[1] = 133;
   ntrlx[2] = 133;
   ntrlx[3] = 19;
   iprlx = hypre_CTAlloc(HYPRE_Int, 4);
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

