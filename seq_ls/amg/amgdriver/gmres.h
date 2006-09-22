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
 * $Revision$
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Header for GMRES
 *
 *****************************************************************************/

#ifndef _GMRES_HEADER
#define _GMRES_HEADER


/*--------------------------------------------------------------------------
 * SPGMRPData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int    (*precond)();
   void    *precond_data;

   hypre_Vector  *s;
   hypre_Vector  *r;

} SPGMRPData;

/*--------------------------------------------------------------------------
 * GMRESData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int        max_krylov;
   int        max_restarts;

   void      *A_data;
   void      *P_data;
   SpgmrMem   spgmr_mem;

   char    *log_file_name;

} GMRESData;

/*--------------------------------------------------------------------------
 * Accessor functions for the GMRESData structure
 *--------------------------------------------------------------------------*/

#define GMRESDataMaxKrylov(gmres_data)    ((gmres_data) -> max_krylov)
#define GMRESDataMaxRestarts(gmres_data)  ((gmres_data) -> max_restarts)

#define GMRESDataAData(gmres_data)        ((gmres_data) -> A_data)
#define GMRESDataPData(gmres_data)        ((gmres_data) -> P_data)
#define GMRESDataSpgmrMem(gmres_data)     ((gmres_data) -> spgmr_mem)

#define GMRESDataLogFileName(gmres_data)  ((gmres_data) -> log_file_name)


#endif
