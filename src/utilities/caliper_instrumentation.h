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

char hypre__markname[512];

#define HYPRE_ANNOTATE_FUNC_BEGIN            CALI_MARK_FUNCTION_BEGIN
#define HYPRE_ANNOTATE_FUNC_END              CALI_MARK_FUNCTION_END
#define HYPRE_ANNOTATE_LOOP_BEGIN(id, str)   CALI_MARK_LOOP_BEGIN(id, str)
#define HYPRE_ANNOTATE_LOOP_END(id)          CALI_MARK_LOOP_END(id)
#define HYPRE_ANNOTATE_ITER_BEGIN(id, it)    CALI_MARK_ITERATION_BEGIN(id, it)
#define HYPRE_ANNOTATE_ITER_END(id)          CALI_MARK_ITERATION_END(id)
#define HYPRE_ANNOTATE_REGION_BEGIN(str, id)\
{\
   hypre_sprintf(hypre__markname, "%s %d", str, id);\
   CALI_MARK_BEGIN(hypre__markname);\
}
#define HYPRE_ANNOTATE_REGION_END(str, id)\
{\
   hypre_sprintf(hypre__markname, "%s %d", str, id);\
   CALI_MARK_END(hypre__markname);\
}
#define HYPRE_ANNOTATE_MGLEVEL_BEGIN(lvl)\
{\
   hypre_sprintf(hypre__markname, "MG level %d", lvl);\
   CALI_MARK_BEGIN(hypre__markname);\
}
#define HYPRE_ANNOTATE_MGLEVEL_END(lvl)\
{\
   hypre_sprintf(hypre__markname, "MG level %d", lvl);\
   CALI_MARK_END(hypre__markname);\
}

#else

#define HYPRE_ANNOTATE_FUNC_BEGIN
#define HYPRE_ANNOTATE_FUNC_END
#define HYPRE_ANNOTATE_LOOP_BEGIN(id, str)
#define HYPRE_ANNOTATE_LOOP_END(id)
#define HYPRE_ANNOTATE_ITER_BEGIN(id, it)
#define HYPRE_ANNOTATE_ITER_END(id)
#define HYPRE_ANNOTATE_REGION_BEGIN(str, id)
#define HYPRE_ANNOTATE_REGION_END(str, id)
#define HYPRE_ANNOTATE_MGLEVEL_BEGIN(lvl)
#define HYPRE_ANNOTATE_MGLEVEL_END(lvl)

#endif

#endif /* CALIPER_INSTRUMENTATION_HEADER */
