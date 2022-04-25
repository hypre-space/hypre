/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the MLI_Method data structure
 *
 *****************************************************************************/

#ifndef __MLIMETHODH__
#define __MLIMETHODH__

/****************************************************************************
 * defines and include files 
 *--------------------------------------------------------------------------*/

#define MLI_METHOD_AMGSA_ID     701
#define MLI_METHOD_AMGSAE_ID    702
#define MLI_METHOD_AMGSADD_ID   703
#define MLI_METHOD_AMGSADDE_ID  704
#define MLI_METHOD_AMGRS_ID     705
#define MLI_METHOD_AMGCR_ID     706

#include "_hypre_utilities.h"
#include "mli.h"

class MLI;

/****************************************************************************
 * MLI_Method abstract class definition
 *--------------------------------------------------------------------------*/

class MLI_Method
{
   char     methodName_[200];
   int      methodID_;
   MPI_Comm mpiComm_;

public :

   MLI_Method( MPI_Comm comm );
   virtual ~MLI_Method();

   virtual int setup( MLI *mli );
   virtual int setParams(char *name, int argc, char *argv[]);
   virtual int getParams(char *name, int *argc, char *argv[]);

   char     *getName();
   int      setName( char *in_name );                                      
   int      setID( int in_id );
   int      getID();
   MPI_Comm getComm();
};

extern MLI_Method *MLI_Method_CreateFromName(char *,MPI_Comm);
extern MLI_Method *MLI_Method_CreateFromID(int,MPI_Comm);

#endif

