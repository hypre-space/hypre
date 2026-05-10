/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Unified instrumentation macros (Caliper + optional GPU NVTX/ROCTX).
 *
 *****************************************************************************/

#ifndef HYPRE_INSTRUMENTATION_HEADER
#define HYPRE_INSTRUMENTATION_HEADER

#include "HYPRE_config.h"

/* --- Caliper backend ---------------------------------------------------- */

#ifdef HYPRE_USING_CALIPER

/* Thin C wrappers — caliper/cali.h is only included in instrumentation.c */
#ifdef __cplusplus
extern "C" {
#endif
void hypre_CaliperMarkFuncBegin(const char *func);
void hypre_CaliperMarkFuncEnd(void);
void hypre_CaliperMarkBegin(const char *name);
void hypre_CaliperMarkEnd(const char *name);
#ifdef __cplusplus
}
#endif

#define HYPRE_CALIPER_FUNC_BEGIN       hypre_CaliperMarkFuncBegin(__func__)
#define HYPRE_CALIPER_FUNC_END         hypre_CaliperMarkFuncEnd()
#define HYPRE_CALIPER_MARK_BEGIN(name) hypre_CaliperMarkBegin(name)
#define HYPRE_CALIPER_MARK_END(name)   hypre_CaliperMarkEnd(name)

#else

#define HYPRE_CALIPER_FUNC_BEGIN
#define HYPRE_CALIPER_FUNC_END
#define HYPRE_CALIPER_MARK_BEGIN(name)
#define HYPRE_CALIPER_MARK_END(name)

#endif /* HYPRE_USING_CALIPER */

/* --- GPU profiling backend (NVTX / ROCTX) ------------------------------- */

#if defined(HYPRE_USING_NVTX) || defined(HYPRE_USING_ROCTX)

/* Forward declarations — implementations in instrumentation.c */
#ifdef __cplusplus
extern "C" {
#endif
void hypre_GpuProfilingPushRange(const char *name);
void hypre_GpuProfilingPopRange(void);
#ifdef __cplusplus
}
#endif

#define HYPRE_GPU_ANNOTATE_PUSH(name) hypre_GpuProfilingPushRange(name)
#define HYPRE_GPU_ANNOTATE_POP()      hypre_GpuProfilingPopRange()

#else

#define HYPRE_GPU_ANNOTATE_PUSH(name)
#define HYPRE_GPU_ANNOTATE_POP()

#endif /* HYPRE_USING_NVTX || HYPRE_USING_ROCTX */

/* --- Unified annotation macros (fire all enabled backends) -------------- */

#define HYPRE_ANNOTATE_FUNC_BEGIN \
   do { HYPRE_CALIPER_FUNC_BEGIN; HYPRE_GPU_ANNOTATE_PUSH(__func__); } while (0)
#define HYPRE_ANNOTATE_FUNC_END \
   do { HYPRE_CALIPER_FUNC_END; HYPRE_GPU_ANNOTATE_POP(); } while (0)
#define HYPRE_ANNOTATE_LOOP_BEGIN(id, str)
#define HYPRE_ANNOTATE_LOOP_END(id)
#define HYPRE_ANNOTATE_ITER_BEGIN(id, it)
#define HYPRE_ANNOTATE_ITER_END(id)
#define HYPRE_ANNOTATE_REGION_BEGIN(...)\
{\
   char hypre__markname[1024];\
   hypre_sprintf(hypre__markname, __VA_ARGS__);\
   HYPRE_CALIPER_MARK_BEGIN(hypre__markname);\
   HYPRE_GPU_ANNOTATE_PUSH(hypre__markname);\
}
#define HYPRE_ANNOTATE_REGION_END(...)\
{\
   char hypre__markname[1024];\
   hypre_sprintf(hypre__markname, __VA_ARGS__);\
   HYPRE_CALIPER_MARK_END(hypre__markname);\
   HYPRE_GPU_ANNOTATE_POP();\
}
#define HYPRE_ANNOTATE_MGLEVEL_BEGIN(lvl)\
{\
   char hypre__levelname[16];\
   hypre_sprintf(hypre__levelname, "MG level %d", lvl);\
   HYPRE_CALIPER_MARK_BEGIN(hypre__levelname);\
   HYPRE_GPU_ANNOTATE_PUSH(hypre__levelname);\
}
#define HYPRE_ANNOTATE_MGLEVEL_END(lvl)\
{\
   char hypre__levelname[16];\
   hypre_sprintf(hypre__levelname, "MG level %d", lvl);\
   HYPRE_CALIPER_MARK_END(hypre__levelname);\
   HYPRE_GPU_ANNOTATE_POP();\
}

#endif /* HYPRE_INSTRUMENTATION_HEADER */
