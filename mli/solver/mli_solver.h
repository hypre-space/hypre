/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for the MLI_Solver data structure
 *
 *****************************************************************************/

#ifndef __MLISOLVERH__
#define __MLISOLVERH__

/*--------------------------------------------------------------------------
 * include files 
 *--------------------------------------------------------------------------*/

#include <string.h>
#include "utilities/utilities.h"

#include "../matrix/mli_matrix.h"
#include "../vector/mli_vector.h"

/*--------------------------------------------------------------------------
 * MLI_Solver data structure declaration
 *--------------------------------------------------------------------------*/

class MLI_Solver
{
   char solver_name[100];
   int  solver_id;

public :

   MLI_Solver(char *name);
   MLI_Solver(int  id);
   ~MLI_Solver()             { }

   char* getName()           { return solver_name; }
   int   getID()             { return solver_id; }

   virtual int setup(MLI_Matrix *)=0;
   virtual int solve(MLI_Vector *, MLI_Vector *)=0;
   virtual int setParams(char *param_string,int argc,char **argv)  {return -1;}
   virtual int getParams(char *param_string,int *argc,char **argv) {return -1;}
};

/*--------------------------------------------------------------------------
 * MLI_Solver functions 
 *--------------------------------------------------------------------------*/

extern MLI_Solver *MLI_Solver_CreateFromName(char *);
extern MLI_Solver *MLI_Solver_CreateFromID(int);

#endif

