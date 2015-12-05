/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.2 $
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
