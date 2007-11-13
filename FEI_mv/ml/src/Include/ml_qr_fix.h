/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
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
 * $Revision$
 ***********************************************************************EHEADER*/




#ifndef ML_QR_FIX_H
#define ML_QR_FIX_H

/* If we need more than 16 kernel components, define ML_QR_FIX_TYPE
 * as unsigned int, otherwise use unsigned short int to conserve memory */
#define ML_QR_FIX_TYPE unsigned int

typedef struct ML_qr_fix {

  int                 level;
  int                 numDeadNodDof;
 /* -mb: can later replace the following two with a hash structure */ 
  int                 nDeadNodDof; 
  ML_QR_FIX_TYPE     *xDeadNodDof;

} ML_qr_fix;

#ifdef __cplusplus
extern "C" {
  int ML_qr_fix_Create(const int nCoarseNod);

  int ML_qr_fix_Destroy(void);

  int ML_qr_fix_Print(ML_qr_fix* ptr);

  int ML_qr_fix_NumDeadNodDof(void);

  ML_QR_FIX_TYPE ML_qr_fix_getDeadNod(const int inx);

  void ML_qr_fix_setNumDeadNod(int num);

  void ML_qr_fix_setDeadNod( const int inx, ML_QR_FIX_TYPE val);

  int  ML_fixCoarseMtx(
          ML_Operator *Cmat,          /*-up- coarse operator in MSR format   */
          const int    CoarseMtxType  /*-in- coarse-lev mtx storage type     */
  );
 
  int  ML_qr_fix_Bitsize(void);
}
#else

int ML_qr_fix_Create(const int nCoarseNod);

int ML_qr_fix_Destroy(void);

int ML_qr_fix_Print(ML_qr_fix* ptr);

int ML_qr_fix_NumDeadNodDof(void);

ML_QR_FIX_TYPE ML_qr_fix_getDeadNod(const int inx);

void ML_qr_fix_setNumDeadNod(int num);

void ML_qr_fix_setDeadNod( const int inx, ML_QR_FIX_TYPE val);

int  ML_fixCoarseMtx(
        ML_Operator *Cmat,          /*-up- coarse operator in MSR format     */
        const int    CoarseMtxType  /*-in- coarse-lev mtx storage type       */
     );

int  ML_qr_fix_Bitsize(void);

#endif
#endif
