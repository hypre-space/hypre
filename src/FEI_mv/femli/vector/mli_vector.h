/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.9 $
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

#include "utilities/_hypre_utilities.h"
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

