/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.3 $
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
   int                 nvars;
   void              **sinterp_data;
} hypre_SysSemiInterpData;

/*--------------------------------------------------------------------------
 * hypre_SysSemiInterpCreate
 *--------------------------------------------------------------------------*/

int
hypre_SysSemiInterpCreate( void **sys_interp_vdata_ptr )
{
   int                      ierr = 0;
   hypre_SysSemiInterpData *sys_interp_data;

   sys_interp_data = hypre_CTAlloc(hypre_SysSemiInterpData, 1);
   *sys_interp_vdata_ptr = (void *) sys_interp_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysSemiInterpSetup
 *--------------------------------------------------------------------------*/

int
hypre_SysSemiInterpSetup( void                 *sys_interp_vdata,
                          hypre_SStructPMatrix *P,
                          int                   P_stored_as_transpose,
                          hypre_SStructPVector *xc,
                          hypre_SStructPVector *e,
                          hypre_Index           cindex,
                          hypre_Index           findex,
                          hypre_Index           stride       )
{
   int                      ierr = 0;

   hypre_SysSemiInterpData  *sys_interp_data = sys_interp_vdata;
   void                    **sinterp_data;

   int                       nvars;

   hypre_StructMatrix       *P_s;
   hypre_StructVector       *xc_s;
   hypre_StructVector       *e_s;

   int                       vi;

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

int
hypre_SysSemiInterp( void                 *sys_interp_vdata,
                     hypre_SStructPMatrix *P,
                     hypre_SStructPVector *xc,
                     hypre_SStructPVector *e            )
{
   int                       ierr = 0;
   
   hypre_SysSemiInterpData  *sys_interp_data = sys_interp_vdata;
   void                    **sinterp_data = (sys_interp_data -> sinterp_data);
   int                       nvars = (sys_interp_data -> nvars);

   void                     *sdata;
   hypre_StructMatrix       *P_s;
   hypre_StructVector       *xc_s;
   hypre_StructVector       *e_s;

   int                       vi;

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

int
hypre_SysSemiInterpDestroy( void *sys_interp_vdata )
{
   int                   ierr = 0;

   hypre_SysSemiInterpData *sys_interp_data = sys_interp_vdata;

   int                   nvars;
   void                **sinterp_data;
   int                   vi;

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

