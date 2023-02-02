/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int           nvars;
   void              **sinterp_data;

} hypre_SysSemiInterpData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysSemiInterpCreate( void **sys_interp_vdata_ptr )
{
   hypre_SysSemiInterpData *sys_interp_data;

   sys_interp_data = hypre_CTAlloc(hypre_SysSemiInterpData,  1, HYPRE_MEMORY_HOST);
   *sys_interp_vdata_ptr = (void *) sys_interp_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
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
   hypre_SysSemiInterpData  *sys_interp_data = (hypre_SysSemiInterpData  *)sys_interp_vdata;
   void                    **sinterp_data;

   HYPRE_Int                 nvars;

   hypre_StructMatrix       *P_s;
   hypre_StructVector       *xc_s;
   hypre_StructVector       *e_s;

   HYPRE_Int                 vi;

   nvars = hypre_SStructPMatrixNVars(P);
   sinterp_data = hypre_CTAlloc(void *,  nvars, HYPRE_MEMORY_HOST);

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

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysSemiInterp( void                 *sys_interp_vdata,
                     hypre_SStructPMatrix *P,
                     hypre_SStructPVector *xc,
                     hypre_SStructPVector *e            )
{
   hypre_SysSemiInterpData  *sys_interp_data = (hypre_SysSemiInterpData  *)sys_interp_vdata;
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

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysSemiInterpDestroy( void *sys_interp_vdata )
{
   hypre_SysSemiInterpData *sys_interp_data = (hypre_SysSemiInterpData  *)sys_interp_vdata;

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
      hypre_TFree(sinterp_data, HYPRE_MEMORY_HOST);
      hypre_TFree(sys_interp_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

