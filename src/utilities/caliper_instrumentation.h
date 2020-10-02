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

char hypre__levelname[16];

#define HYPRE_ANNOTATE_FUNC_BEGIN          CALI_MARK_FUNCTION_BEGIN
#define HYPRE_ANNOTATE_FUNC_END            CALI_MARK_FUNCTION_END
#define HYPRE_ANNOTATE_LOOP_BEGIN(id, str) CALI_MARK_LOOP_BEGIN(id, str)
#define HYPRE_ANNOTATE_LOOP_END(id)        CALI_MARK_LOOP_END(id)
#define HYPRE_ANNOTATE_ITER_BEGIN(id, it)  CALI_MARK_ITERATION_BEGIN(id, it)
#define HYPRE_ANNOTATE_ITER_END(id)        CALI_MARK_ITERATION_END(id)
#define HYPRE_ANNOTATE_MGLEVEL_BEGIN(lvl)\
{\
   hypre_sprintf(hypre__levelname, "MG level %d", lvl);\
   CALI_MARK_BEGIN(hypre__levelname);\
}
#define HYPRE_ANNOTATE_MGLEVEL_END(lvl)\
{\
   hypre_sprintf(hypre__levelname, "MG level %d", lvl);\
   CALI_MARK_END(hypre__levelname);\
}

#else

#define HYPRE_ANNOTATE_FUNC_BEGIN
#define HYPRE_ANNOTATE_FUNC_END
#define HYPRE_ANNOTATE_LOOP_BEGIN(id, str)
#define HYPRE_ANNOTATE_LOOP_END(id)
#define HYPRE_ANNOTATE_ITER_BEGIN(id, it)
#define HYPRE_ANNOTATE_ITER_END(id)
#define HYPRE_ANNOTATE_MGLEVEL_BEGIN(lvl)
#define HYPRE_ANNOTATE_MGLEVEL_END(lvl)

#endif

#endif /* CALIPER_INSTRUMENTATION_HEADER */
