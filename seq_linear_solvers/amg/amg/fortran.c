/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Fortran to C interfacing macros
 *
 *****************************************************************************/

#include "amg.h"


/*--------------------------------------------------------------------------
 * writeysm_
 *--------------------------------------------------------------------------*/

void     writeysm_(file_name, data, ia, ja, size, file_name_len)
char    *file_name;
double  *data;
int     *ia;
int     *ja;
int     *size;
int      file_name_len;
{
   Matrix  *matrix;


   matrix = NewMatrix(data, ia, ja, *size);

   WriteYSMP(file_name, matrix);

   tfree(matrix);

   return;
}

/*--------------------------------------------------------------------------
 * writevec_
 *--------------------------------------------------------------------------*/

void     writevec_(file_name, data, size, file_name_len)
char    *file_name;
double  *data;
int     *size;
int      file_name_len;
{
   Vector  *vector;


   vector = NewVector(data, *size);

   WriteVec(file_name, vector);

   tfree(vector);

   return;
}

