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
 * $Revision: 2.2 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Header for PCG
 *
 *****************************************************************************/

#ifndef _PCG_HEADER
#define _PCG_HEADER


/*--------------------------------------------------------------------------
 * PCGData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      max_iter;
   int      two_norm;

   hypre_CSRMatrix  *A;
   hypre_Vector  *p;
   hypre_Vector  *s;
   hypre_Vector  *r;

   int    (*precond)();
   void    *precond_data;

   char    *log_file_name;

} PCGData;

/*--------------------------------------------------------------------------
 * Accessor functions for the PCGData structure
 *--------------------------------------------------------------------------*/

#define PCGDataMaxIter(pcg_data)      ((pcg_data) -> max_iter)
#define PCGDataTwoNorm(pcg_data)      ((pcg_data) -> two_norm)

#define PCGDataA(pcg_data)            ((pcg_data) -> A)
#define PCGDataP(pcg_data)            ((pcg_data) -> p)
#define PCGDataS(pcg_data)            ((pcg_data) -> s)
#define PCGDataR(pcg_data)            ((pcg_data) -> r)

#define PCGDataPrecond(pcg_data)      ((pcg_data) -> precond)
#define PCGDataPrecondData(pcg_data)  ((pcg_data) -> precond_data)

#define PCGDataLogFileName(pcg_data)  ((pcg_data) -> log_file_name)


#endif
