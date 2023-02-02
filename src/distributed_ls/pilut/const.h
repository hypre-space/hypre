/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef CONST_H
#define CONST_H

/*
 * const.h
 *
 * This file contains MPI specific tag constants for send/recvs
 * In the LDU solves, the lower 16bits are used for the nlevel
 */

enum Tags {
  /* io.c        */
  TAG_CSR_dist,
  TAG_CSR_rowdist,
  TAG_CSR_rowptr,
  TAG_CSR_colind,
  TAG_CSR_values,

  /* matvec.c */
  TAG_MV_rind,
  TAG_MVGather,

  /* parilut.c   */
  TAG_Comm_rrowind,
  TAG_Send_colind,
  TAG_Send_values,

  /* trifactor.c */
  TAG_SetUp_rind,
  TAG_SetUp_reord,
  TAG_SetUp_rnum,
  TAG_LDU_lx       = 0x0100,  /* uses low 16bits */
  TAG_LDU_ux       = 0x0200   /* uses low 16bits */
};

#endif
