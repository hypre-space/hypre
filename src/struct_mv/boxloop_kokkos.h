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
 * Header info for the BoxLoop
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * BoxLoop macros:
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_NEWBOXLOOP_HEADER
#define HYPRE_NEWBOXLOOP_HEADER
extern "C++" {
#include <Kokkos_Core.hpp>
}
#if defined( KOKKOS_HAVE_MPI )
#include <mpi.h>
#endif

#include "_hypre_struct_mv.h"

#define zypre_newBoxLoopInit(ndim)											\
	HYPRE_Int hypre__tot = 1.0;											\
	for (HYPRE_Int i = 0;i < ndim;i ++)									\
		hypre__tot *= loop_size[i];


#define zypre_BoxLoopDeclare(idx)										\
	HYPRE_Int local_idx;												\
	HYPRE_Int d,idx_local = idx;

#define zypre_newBoxLoop0Begin(ndim, loop_size) 	\
{																	\
    zypre_newBoxLoopInit(ndim)											\
	Kokkos::parallel_for (hypre__tot, [=] (HYPRE_Int idx) \
    {																		\
	    zypre_newBoxLoopDeclare(idx)										\
	    HYPRE_Int hypre_boxD1 = 1.0;										\
	    HYPRE_Int i1 = idx;


#define zypre_newBoxLoop0End(i1)				\
	});											\
}


#define zypre_newBoxLoop1Begin(ndim, loop_size,				\
							   dbox1, start1, stride1, i1) 	\
{																	\
    zypre_newBoxLoopInit(ndim)											\
	Kokkos::parallel_for (hypre__tot, [=] (HYPRE_Int idx) \
    {																		\
	    zypre_newBoxLoopDeclare(idx)										\
	    HYPRE_Int hypre_boxD1 = 1.0;										\
	    HYPRE_Int i1 = 0;											\
		for (d = 0;d < ndim;d ++)										\
		{																\
			local_idx  = idx_local % loop_size[d];						\
			idx_local  = idx_local / loop_size[d];						\
			i1 += (local_idx*stride1[d] + hypre_IndexD(start1, d) - hypre_BoxIMinD(dbox1, d)) * hypre_boxD1; \
			hypre_boxD1 *= hypre_BoxSizeD(dbox1, d);					\
		}																\


#define zypre_newBoxLoop1End(i1)				\
	});											\
}


#define zypre_newBoxLoop2Begin(ndim, loop_size,				\
                                dbox1, start1, stride1, i1,	\
                                dbox2, start2, stride2, i2)	\
{    														\
    zypre_newBoxLoopInit(ndim)													\
	Kokkos::parallel_for (hypre__tot, [=] (HYPRE_Int idx)				\
	{																	\
	    zypre_newBoxLoopDeclare(idx)											\
	    HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0;						\
	    HYPRE_Int i1 = 0, i2 = 0;											\
		for (d = 0;d < ndim;d ++)										\
		{																\
			local_idx  = idx_local % loop_size[d];						\
			idx_local  = idx_local / loop_size[d];						\
			i1 += (local_idx*stride1[d] + start1[d] - hypre_BoxIMinD(dbox1, d)) * hypre_boxD1; \
			hypre_boxD1 *= hypre_BoxSizeD(dbox1, d);					\
			i2 += (local_idx*stride2[d] + start2[d] - hypre_BoxIMinD(dbox2, d)) * hypre_boxD2; \
			hypre_boxD2 *= hypre_BoxSizeD(dbox2, d);					\
		}


#define zypre_newBoxLoop2End(i1, i2)			\
	});											\
}


#define zypre_newBoxLoop3Begin(ndim, loop_size,\
							   dbox1, start1, stride1, i1,	\
							   dbox2, start2, stride2, i2,	\
							   dbox3, start3, stride3, i3)	\
{																	\
    zypre_newBoxLoopInit(ndim)													\
	Kokkos::parallel_for (hypre__tot, [=] (HYPRE_Int idx)				\
	{																	\
	    zypre_newBoxLoopDeclare(idx)											\
	    HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0,hypre_boxD3 = 1.0;	\
	    HYPRE_Int i1 = 0, i2 = 0, i3 = 0;											\
		for (d = 0;d < ndim;d ++)										\
		{																\
			local_idx  = idx_local % loop_size[d];						\
			idx_local  = idx_local / loop_size[d];						\
			i1 += (local_idx*stride1[d] + hypre_IndexD(start1, d) - hypre_BoxIMinD(dbox1, d)) * hypre_boxD1; \
			hypre_boxD1 *= hypre_BoxSizeD(dbox1, d);					\
			i2 += (local_idx*stride2[d] + start2[d] - hypre_BoxIMinD(dbox2, d)) * hypre_boxD2; \
			hypre_boxD2 *= hypre_BoxSizeD(dbox2, d);					\
			i3 += (local_idx*stride3[d] + start3[d] - hypre_BoxIMinD(dbox3, d)) * hypre_boxD3; \
			hypre_boxD3 *= hypre_BoxSizeD(dbox3, d);					\
		}		

#define zypre_newBoxLoop3End(i1, i2,i3)			\
	});											\
}

#define zypre_newBoxLoop4Begin(ndim, loop_size,\
							   dbox1, start1, stride1, i1,	\
							   dbox2, start2, stride2, i2,	\
							   dbox3, start3, stride3, i3,			\
							   dbox4, start4, stride4, i4)			\
{																	\
    zypre_newBoxLoopInit(ndim)													\
	Kokkos::parallel_for (hypre__tot, [=] (HYPRE_Int idx)				\
	{																	\
	    zypre_newBoxLoopDeclare(idx)											\
     	HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0,hypre_boxD3 = 1.0,hypre_boxD4 = 1.0;	\
	    HYPRE_Int i1 = 0, i2 = 0, i3 = 0, i4 = 0;								\
		for (d = 0;d < ndim;d ++)										\
		{																\
			local_idx  = idx_local % loop_size[d];						\
			idx_local  = idx_local / loop_size[d];						\
			i1 += (local_idx*stride1[d] + hypre_IndexD(start1, d) - hypre_BoxIMinD(dbox1, d)) * hypre_boxD1; \
			hypre_boxD1 *= hypre_BoxSizeD(dbox1, d);					\
			i2 += (local_idx*stride2[d] + start2[d] - hypre_BoxIMinD(dbox2, d)) * hypre_boxD2; \
			hypre_boxD2 *= hypre_BoxSizeD(dbox2, d);					\
			i3 += (local_idx*stride3[d] + start3[d] - hypre_BoxIMinD(dbox3, d)) * hypre_boxD3; \
			hypre_boxD3 *= hypre_BoxSizeD(dbox3, d);					\
			i4 += (local_idx*stride4[d] + start4[d] - hypre_BoxIMinD(dbox4, d)) * hypre_boxD4; \
			hypre_boxD4 *= hypre_BoxSizeD(dbox4, d);					\
		}		

#define zypre_newBoxLoop4End(i1, i2, i3, i4)		\
	});											\
}


#define zypre_newBoxLoop2ReductionBegin(ndim, loop_size,				\
										dbox1, start1, stride1, i1,		\
										dbox2, start2, stride2, i2,		\
										sum)							\
{    														\
    zypre_newBoxLoopInit(ndim)												\
	Kokkos::parallel_reduce (hypre__tot, [=] (HYPRE_Int idx,HYPRE_Real &sum) \
	{																	\
	    zypre_newBoxLoopDeclare(idx)											\
	    HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0;						\
	    HYPRE_Int i1 = 0, i2 = 0;											\
		for (d = 0;d < ndim;d ++)										\
		{																\
			local_idx  = idx_local % loop_size[d];						\
			idx_local  = idx_local / loop_size[d];						\
			i1 += (local_idx*stride1[d] + start1[d] - hypre_BoxIMinD(dbox1, d)) * hypre_boxD1; \
			hypre_boxD1 *= hypre_BoxSizeD(dbox1, d);					\
			i2 += (local_idx*stride2[d] + start2[d] - hypre_BoxIMinD(dbox2, d)) * hypre_boxD2; \
			hypre_boxD2 *= hypre_BoxSizeD(dbox2, d);					\
		}


#define zypre_newBoxLoop2ReductionEnd(i1, i2, sum)			\
	},sum);													\
}

#endif
