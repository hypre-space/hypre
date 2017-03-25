/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header file for Caliper instrumentation macros
 *
 *****************************************************************************/

#ifndef CALIPER_INSTRUMENTATION_HEADER
#define CALIPER_INSTRUMENTATION_HEADER

#include "HYPRE_config.h"

#ifdef HYPRE_USING_CALIPER

#include <caliper/cali.h>

#define HYPRE_ANNOTATION_BEGIN( str ) cali_begin_string_byname("hypre.kernel", str)
#define HYPRE_ANNOTATION_END( str ) cali_end_byname("hypre.kernel")

#else

#define HYPRE_ANNOTATION_BEGIN( str ) 
#define HYPRE_ANNOTATION_END( str ) 

#endif

#endif /* CALIPER_INSTRUMENTATION_HEADER */
