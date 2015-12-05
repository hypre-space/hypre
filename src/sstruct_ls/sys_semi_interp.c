/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SysSemiInterpData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int           nvars;
   void              **sinterp_data;
} hypre_SysSemiInterpData;

/*--------------------------------------------------------------------------
 * hypre_SysSemiInterpCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysSemiInterpCreate( void **sys_interp_vdata_ptr )
{
   HYPRE_Int                ierr = 0;
   hypre_SysSemiInterpData *sys_interp_data;

   sys_interp_data = hypre_CTAlloc(hypre_SysSemiInterpData, 1);
   *sys_interp_vdata_ptr = (void *) sys_interp_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysSemiInterpSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysSemiInterpSetup( void                 *sys_interp_vdata,
                          hypre_SStructPMatrix *P,
                          HYPRE_Int             P_stored_as_transpose,
                          hypre_SStructPVector *xc,
                          hypre_SStructPVector *e,
                          hypre_Index           cindex,
                          hypre_Index           findex,
                          hypre_Index           stride       )
{
   HYPRE_Int                ierr = 0;

   hypre_SysSemiInterpData  *sys_interp_data = sys_interp_vdata;
   void                    **sinterp_data;

   HYPRE_Int                 nvars;

   hypre_StructMatrix       *P_s;
   hypre_StructVector       *xc_s;
   hypre_StructVector       *e_s;

   HYPRE_Int                 vi;

   nvars = hypre_SStructPMatrixNVars(P);
   sinterp_data = hypre_CTAlloc(void *, nvars);

   for (vi = 0; vi < nvars; vi++)
   {
      P_s  = hypre_SStructPMatrixSMatrix(P, vi, vi);
      xc_s = hypre_SStructPVectorSVector(xc, vi);
      e_s  = hypre_SStructPVectorSVector(e, vi);
      sinterp_data[vi] = hypre_SemiInterpCreate( );
      hypre_SemiInterpSetup( sinterp_data[vi], P_s, P_stored_as_transpose,
                             xc_s, e_s, cindex, findex, stride);
   }

   (sys_interp_data -> nvars)        = nvars;
   (sys_interp_data -> sinterp_data) = sinterp_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysSemiInterp:
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysSemiInterp( void                 *sys_interp_vdata,
                     hypre_SStructPMatrix *P,
                     hypre_SStructPVector *xc,
                     hypre_SStructPVector *e            )
{
   HYPRE_Int                 ierr = 0;
   
   hypre_SysSemiInterpData  *sys_interp_data = sys_interp_vdata;
   void                    **sinterp_data = (sys_interp_data -> sinterp_data);
   HYPRE_Int                 nvars = (sys_interp_data -> nvars);

   void                     *sdata;
   hypre_StructMatrix       *P_s;
   hypre_StructVector       *xc_s;
   hypre_StructVector       *e_s;

   HYPRE_Int                 vi;

   for (vi = 0; vi < nvars; vi++)
   {
      sdata = sinterp_data[vi];
      P_s  = hypre_SStructPMatrixSMatrix(P, vi, vi);
      xc_s = hypre_SStructPVectorSVector(xc, vi);
      e_s  = hypre_SStructPVectorSVector(e, vi);
      hypre_SemiInterp(sdata, P_s, xc_s, e_s);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysSemiInterpDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysSemiInterpDestroy( void *sys_interp_vdata )
{
   HYPRE_Int             ierr = 0;

   hypre_SysSemiInterpData *sys_interp_data = sys_interp_vdata;

   HYPRE_Int             nvars;
   void                **sinterp_data;
   HYPRE_Int             vi;

   if (sys_interp_data)
   {
      nvars        = (sys_interp_data -> nvars);
      sinterp_data = (sys_interp_data -> sinterp_data);
      for (vi = 0; vi < nvars; vi++)
      {
         if (sinterp_data[vi] != NULL)
         {
            hypre_SemiInterpDestroy(sinterp_data[vi]);
         }
      }
      hypre_TFree(sinterp_data);
      hypre_TFree(sys_interp_data);
   }
   return ierr;
}

