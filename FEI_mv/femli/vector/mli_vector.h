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
 * $Revision$
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Header info for the MLI_Vector data structure
 *
 *****************************************************************************/

#ifndef __MLIVECTOR_H__
#define __MLIVECTOR_H__

class MLI_Vector;
class MLI_Matrix;

/*--------------------------------------------------------------------------
 * include files 
 *--------------------------------------------------------------------------*/

#include "utilities/utilities.h"
#include "matrix/mli_matrix.h"
#include "util/mli_utils.h"

/*--------------------------------------------------------------------------
 * MLI_Matrix data structure declaration
 *--------------------------------------------------------------------------*/

class MLI_Vector
{
   char  name_[100];
   void  *vector_;
   int   (*destroyFunc_)(void*);

public :

   MLI_Vector( void *inVec, char *inName, MLI_Function *funcPtr );
   ~MLI_Vector();
   char   *getName();
   void   *getVector();
   int    setConstantValue(double value);
   int    copy(MLI_Vector *vec2);
   int    print(char *filename);
   double norm2();
   MLI_Vector *clone();
};

#endif

