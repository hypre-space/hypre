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

#define HYPRE_ANNOTATION_BEGIN(str) CALI_MARK_BEGIN(str)
#define HYPRE_ANNOTATION_END(str)   CALI_MARK_END(str)
#define HYPRE_ANNOTATE_FUNC_BEGIN   CALI_MARK_FUNCTION_BEGIN
#define HYPRE_ANNOTATE_FUNC_END     CALI_MARK_FUNCTION_END

#else

#define HYPRE_ANNOTATION_BEGIN(str)
#define HYPRE_ANNOTATION_END(str)
#define HYPRE_ANNOTATE_FUNC_BEGIN
#define HYPRE_ANNOTATE_FUNC_END

#endif

#endif /* CALIPER_INSTRUMENTATION_HEADER */
