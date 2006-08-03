/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/********************************************************************* */
/* See the file COPYRIGHT for a complete copyright notice, contact      */
/* person and disclaimer.                                               */        
/* ******************************************************************** */

/********************************************************************* */
/*          BLAS/LAPACK Utilities for Aztec/ML users                   */
/********************************************************************* */

#ifndef __MLAZTECLAPACK__
#define __MLAZTECLAPACK__



#ifdef AZTEC
#include "ml_common.h"
#include "ml_defs.h"
#define ML_IDAMAX_FUNC
#define ML_DSWAP_FUNC
#define ML_DSCAL_FUNC
#define ML_DAXPY_FUNC
#define ML_DASUM_FUNC
#define ML_DDOT_FUNC
#define ML_DNRM2_FUNC
#define ML_DCOPY_FUNC
#define ML_DGEMM_FUNC
#define ML_DTRSM_FUNC
#define ML_DTRMM_FUNC
#define ML_DGETRS_FUNC
#define ML_LSAME_FUNC
#define ML_XERBLA_FUNC
#define ML_DLASWP_FUNC
#define ML_DGEMV_FUNC
#define ML_DGETRF_FUNC
#define ML_DGER_FUNC
#define ML_DTRMV_FUNC
#define ML_DTRSV_FUNC

#ifndef FSUB_TYPE
#  if defined(ncube)
#     define  FSUB_TYPE void
#  elif defined(paragon)
#     define  FSUB_TYPE void
#  elif defined(hp)
#     define  FSUB_TYPE void
#  else
#     define  FSUB_TYPE int
#  endif
#endif


#endif
#endif
