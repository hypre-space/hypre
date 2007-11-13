/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
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




#ifndef __ALE_FEI_INCLUDE_H__
#define __ALE_FEI_INCLUDE_H__

#ifdef EIGHT_BYTE_GLOBAL_ID
   typedef long long GlobalID;
#else
   typedef int GlobalID;
#endif

#define FEI_FATAL_ERROR -1
#define FEI_LOCAL_TIMES 0
#define FEI_MAX_TIMES	1
#define FEI_MIN_TIMES	2

#if HAVE_FEI

#   define FEI_NOT_USING_ESI
#   ifdef AIX
#      define BOOL_NOT_SUPPORTED
#   endif

#   if PARALLEL
#      define FEI_PAR
#   else
#      define FEI_SER
#   endif

#endif

#endif

