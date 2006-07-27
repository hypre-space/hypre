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


#ifdef MLI_SUPERLU

#ifndef __MLI_SOLVER_SEQSUPERLU_H__
#define __MLI_SOLVER_SEQSUPERLU_H__

#include <stdio.h>
#include "dsp_defs.h"
#include "superlu_util.h"
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

/******************************************************************************
 * data structure for the sequential SuperLU solution scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_SeqSuperLU : public MLI_Solver
{
   MLI_Matrix   *mliAmat_;
   int          factorized_;
   int          **permRs_;
   int          **permCs_;
   int          localNRows_;
   SuperMatrix  superLU_Lmats[100];
   SuperMatrix  superLU_Umats[100];
   int          nSubProblems_;
   int          **subProblemRowIndices_;
   int          *subProblemRowSizes_;
   int          numColors_;
   int          *myColors_;
   int          nRecvs_;
   int          *recvProcs_;
   int          *recvLengs_;
   int          nSends_;
   int          *sendProcs_;
   int          *sendLengs_;
   MPI_Comm     AComm_;
   MLI_Matrix   *PSmat_;
   MLI_Vector   *PSvec_;

public :

   MLI_Solver_SeqSuperLU(char *name);
   ~MLI_Solver_SeqSuperLU();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams(char *paramString, int argc, char **argv);
   int setupBlockColoring();
};

#endif

#else
   int bogus;
#endif

