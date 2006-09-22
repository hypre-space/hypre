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
 * Header info for WJacobi solver
 *
 *****************************************************************************/

#ifndef _WJACOBI_HEADER
#define _WJACOBI_HEADER


/*--------------------------------------------------------------------------
 * WJacobiData
 *--------------------------------------------------------------------------*/

typedef struct
{
   double   weight;
   int      max_iter;

   hypre_Matrix  *A;
   hypre_Vector  *t;

   char    *log_file_name;

} WJacobiData;

/*--------------------------------------------------------------------------
 * Accessor functions for the WJacobiData structure
 *--------------------------------------------------------------------------*/

#define WJacobiDataWeight(wjacobi_data)      ((wjacobi_data) -> weight)
#define WJacobiDataMaxIter(wjacobi_data)     ((wjacobi_data) -> max_iter)

#define WJacobiDataA(wjacobi_data)           ((wjacobi_data) -> A)
#define WJacobiDataT(wjacobi_data)           ((wjacobi_data) -> t)

#define WJacobiDataLogFileName(wjacobi_data) ((wjacobi_data) -> log_file_name)


#endif
