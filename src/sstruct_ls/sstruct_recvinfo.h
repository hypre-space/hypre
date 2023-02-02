/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_SStructRecvInfo data structure
 *--------------------------------------------------------------------------*/
#ifndef hypre_RECVINFODATA_HEADER
#define hypre_RECVINFODATA_HEADER


typedef struct
{
   HYPRE_Int             size;

   hypre_BoxArrayArray  *recv_boxes;
   HYPRE_Int           **recv_procs;

} hypre_SStructRecvInfoData;

#endif
