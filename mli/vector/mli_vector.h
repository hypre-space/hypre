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

#ifndef __MLIVECTORH__
#define __MLIVECTORH__

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
   char  name[100];
   void  *vector;
   int   (*destroy_func)(void*);

public :

   MLI_Vector( void *in_vec, char *in_name, MLI_Function *func_ptr );
   ~MLI_Vector();
   char   *getName()                            { return name; }
   void   *getVector()                          { return vector; }
   int    setConstantValue(double value);
   int    copy(MLI_Vector *vec2);
   int    print(char *filename);
   double norm2();
   MLI_Vector *clone();
};

#endif

