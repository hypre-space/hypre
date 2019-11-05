/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

