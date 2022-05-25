/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

#include "_hypre_utilities.h"
#include "mli_matrix.h"
#include "mli_utils.h"

/*--------------------------------------------------------------------------
 * MLI_Matrix data structure declaration
 *--------------------------------------------------------------------------*/

class MLI_Vector
{
   char  name_[100];
   void  *vector_;
   int   (*destroyFunc_)(void*);

public :

   MLI_Vector( void *inVec,const char *inName, MLI_Function *funcPtr );
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

