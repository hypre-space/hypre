/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

