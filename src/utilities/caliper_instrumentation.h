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

#ifdef __cplusplus
extern "C++"
{
#endif

#include <caliper/cali.h>

#ifdef __cplusplus
}
#endif

extern HYPRE_Int hypre__caliper_maxdepth;
extern HYPRE_Int hypre__caliper_depth;
static char hypre__caliper_levelname[16];
static char hypre__caliper_markname[1024];

#define HYPRE_ANNOTATE_FUNC_BEGIN \
{\
   hypre__caliper_depth++;\
   if (hypre__caliper_depth < hypre__caliper_maxdepth)\
   {\
      CALI_MARK_FUNCTION_BEGIN;\
   }\
}
#define HYPRE_ANNOTATE_FUNC_END \
{\
   if (hypre__caliper_depth < hypre__caliper_maxdepth)\
   {\
      CALI_MARK_FUNCTION_END;\
   }\
   hypre__caliper_depth--;\
}
#define HYPRE_ANNOTATE_LOOP_BEGIN(id, str)\
{\
   hypre__caliper_depth++;\
   if (hypre__caliper_depth < hypre__caliper_maxdepth)\
   {\
      CALI_MARK_LOOP_BEGIN(id, str);\
   }\
}
#define HYPRE_ANNOTATE_LOOP_END(id)\
{\
   if (hypre__caliper_depth < hypre__caliper_maxdepth)\
   {\
      CALI_MARK_LOOP_END(id);\
   }\
   hypre__caliper_depth--;\
}
#define HYPRE_ANNOTATE_ITER_BEGIN(id, it)\
{\
   hypre__caliper_depth++;\
   if (hypre__caliper_depth < hypre__caliper_maxdepth)\
   {\
      CALI_MARK_ITERATION_BEGIN(id, it);\
   }\
}
#define HYPRE_ANNOTATE_ITER_END(id)\
{\
   if (hypre__caliper_depth < hypre__caliper_maxdepth)\
   {\
      CALI_MARK_ITERATION_END(id);\
   }\
   hypre__caliper_depth--;\
}
#define HYPRE_ANNOTATE_REGION_BEGIN(...)\
{\
   hypre__caliper_depth++;\
   if (hypre__caliper_depth < hypre__caliper_maxdepth)\
   {\
      hypre_sprintf(hypre__caliper_markname, __VA_ARGS__);\
      CALI_MARK_BEGIN(hypre__caliper_markname);\
   }\
}
#define HYPRE_ANNOTATE_REGION_END(...)\
{\
   if (hypre__caliper_depth < hypre__caliper_maxdepth)\
   {\
      hypre_sprintf(hypre__caliper_markname, __VA_ARGS__);\
      CALI_MARK_END(hypre__caliper_markname);\
   }\
   hypre__caliper_depth--;\
}
#define HYPRE_ANNOTATE_MGLEVEL_BEGIN(lvl)\
{\
   hypre__caliper_depth++;\
   if (hypre__caliper_depth < hypre__caliper_maxdepth)\
   {\
      hypre_sprintf(hypre__caliper_levelname, "MG level %d", lvl);\
      CALI_MARK_BEGIN(hypre__caliper_levelname);\
   }\
}
#define HYPRE_ANNOTATE_MGLEVEL_END(lvl)\
{\
   if (hypre__caliper_depth < hypre__caliper_maxdepth)\
   {\
      hypre_sprintf(hypre__caliper_levelname, "MG level %d", lvl);\
      CALI_MARK_END(hypre__caliper_levelname);\
   }\
   hypre__caliper_depth--;\
}

#else

#define HYPRE_ANNOTATE_FUNC_BEGIN
#define HYPRE_ANNOTATE_FUNC_END
#define HYPRE_ANNOTATE_LOOP_BEGIN(id, str)
#define HYPRE_ANNOTATE_LOOP_END(id)
#define HYPRE_ANNOTATE_ITER_BEGIN(id, it)
#define HYPRE_ANNOTATE_ITER_END(id)
#define HYPRE_ANNOTATE_REGION_BEGIN(...)
#define HYPRE_ANNOTATE_REGION_END(...)
#define HYPRE_ANNOTATE_MAX_MGLEVEL(lvl)
#define HYPRE_ANNOTATE_MGLEVEL_BEGIN(lvl)
#define HYPRE_ANNOTATE_MGLEVEL_END(lvl)

#endif

#endif /* CALIPER_INSTRUMENTATION_HEADER */
