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
   void              **srestrict_data;
} hypre_SysSemiRestrictData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysSemiRestrictCreate( void **sys_restrict_vdata_ptr)
{
   hypre_SysSemiRestrictData *sys_restrict_data;

   sys_restrict_data = hypre_CTAlloc(hypre_SysSemiRestrictData,  1, HYPRE_MEMORY_HOST);
   *sys_restrict_vdata_ptr = (void *) sys_restrict_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysSemiRestrictSetup( void                 *sys_restrict_vdata,
                            hypre_SStructPMatrix *R,
                            HYPRE_Int             R_stored_as_transpose,
                            hypre_SStructPVector *r,
                            hypre_SStructPVector *rc,
                            hypre_Index           cindex,
                            hypre_Index           findex,
                            hypre_Index           stride                )
{
   hypre_SysSemiRestrictData  *sys_restrict_data = (hypre_SysSemiRestrictData  *)sys_restrict_vdata;
   void                      **srestrict_data;

   HYPRE_Int                   nvars;

   hypre_StructMatrix         *R_s;
   hypre_StructVector         *rc_s;
   hypre_StructVector         *r_s;

   HYPRE_Int                   vi;

   nvars = hypre_SStructPMatrixNVars(R);
   srestrict_data = hypre_CTAlloc(void *,  nvars, HYPRE_MEMORY_HOST);

   for (vi = 0; vi < nvars; vi++)
   {
      R_s  = hypre_SStructPMatrixSMatrix(R, vi, vi);
      rc_s = hypre_SStructPVectorSVector(rc, vi);
      r_s  = hypre_SStructPVectorSVector(r, vi);
      srestrict_data[vi] = hypre_SemiRestrictCreate( );
      hypre_SemiRestrictSetup( srestrict_data[vi], R_s, R_stored_as_transpose,
                               r_s, rc_s, cindex, findex, stride);
   }

   (sys_restrict_data -> nvars)        = nvars;
   (sys_restrict_data -> srestrict_data) = srestrict_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysSemiRestrict( void                 *sys_restrict_vdata,
                       hypre_SStructPMatrix *R,
                       hypre_SStructPVector *r,
                       hypre_SStructPVector *rc             )
{
   hypre_SysSemiRestrictData  *sys_restrict_data = (hypre_SysSemiRestrictData  *)sys_restrict_vdata;
   void                      **srestrict_data
      = (sys_restrict_data -> srestrict_data);
   HYPRE_Int                   nvars = (sys_restrict_data -> nvars);

   void                       *sdata;
   hypre_StructMatrix         *R_s;
   hypre_StructVector         *rc_s;
   hypre_StructVector         *r_s;

   HYPRE_Int                   vi;

   for (vi = 0; vi < nvars; vi++)
   {
      sdata = srestrict_data[vi];
      R_s  = hypre_SStructPMatrixSMatrix(R, vi, vi);
      rc_s = hypre_SStructPVectorSVector(rc, vi);
      r_s  = hypre_SStructPVectorSVector(r, vi);
      hypre_SemiRestrict(sdata, R_s, r_s, rc_s);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysSemiRestrictDestroy( void *sys_restrict_vdata )
{
   hypre_SysSemiRestrictData *sys_restrict_data = (hypre_SysSemiRestrictData  *)sys_restrict_vdata;

   HYPRE_Int               nvars;
   void                  **srestrict_data;
   HYPRE_Int               vi;

   if (sys_restrict_data)
   {
      nvars        = (sys_restrict_data -> nvars);
      srestrict_data = (sys_restrict_data -> srestrict_data);
      for (vi = 0; vi < nvars; vi++)
      {
         if (srestrict_data[vi] != NULL)
         {
            hypre_SemiRestrictDestroy(srestrict_data[vi]);
         }
      }
      hypre_TFree(srestrict_data, HYPRE_MEMORY_HOST);
      hypre_TFree(sys_restrict_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

