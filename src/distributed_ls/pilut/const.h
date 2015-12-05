/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/




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
