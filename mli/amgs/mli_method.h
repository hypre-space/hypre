/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for the MLI_Method data structure
 *
 *****************************************************************************/

#ifndef __MLIMETHODH__
#define __MLIMETHODH__

/****************************************************************************
 * include files 
 *--------------------------------------------------------------------------*/

#include <mpi.h>
#include "utilities/utilities.h"
#include "../base/mli.h"

class MLI;

/****************************************************************************
 * MLI_Method data structure declaration
 *--------------------------------------------------------------------------*/

class MLI_Method
{
   char     name[200];
   int      method_id;
   void     *method_data;
   MPI_Comm mpi_comm;

public :

   MLI_Method( char *str, MPI_Comm comm );
   ~MLI_Method();
   int  setup( MLI *mli );
   char *getName()      { return name; }
   int  setParams(char *name, int argc, char *argv[]);
   int  getParams(char *name, int **intvec, double **dblevec);
   int  setName( char *str );
};

#endif

