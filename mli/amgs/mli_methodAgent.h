/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for the MLI_MethodAgent data structure
 *
 *****************************************************************************/

#ifndef __MLIMETHODAGENTH__
#define __MLIMETHODHAGENT__

/****************************************************************************
 * include files 
 *--------------------------------------------------------------------------*/

#include <mpi.h>
#include "utilities/utilities.h"
#include "../base/mli.h"
#include "mli_method.h"

class MLI;

/****************************************************************************
 * MLI_MethodAgent data structure declaration
 *--------------------------------------------------------------------------*/

class MLI_MethodAgent
{
   char       name[200];
   int        method_id;
   MLI_Method *method_ptr;
   MPI_Comm   mpi_comm;

public :

   MLI_MethodAgent( MPI_Comm comm );
   ~MLI_MethodAgent();

   int  createMethod( char *str );
   int  giveMethod( MLI_Method * );
   int  setParams(char *name, int argc, char *argv[]);
   int  setup( MLI *mli );

   char       *getName()      { return name; }
   MLI_Method *takeMethod();
   int        getParams(char *name, int *argc, char *argv[]);
};

#endif

