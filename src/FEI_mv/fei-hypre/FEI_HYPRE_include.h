/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
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

