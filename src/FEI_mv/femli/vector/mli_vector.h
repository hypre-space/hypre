/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

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

