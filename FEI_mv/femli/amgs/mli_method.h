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
 * defines and include files 
 *--------------------------------------------------------------------------*/

#define MLI_METHOD_AMGSA_ID     701
#define MLI_METHOD_AMGSAE_ID    702
#define MLI_METHOD_AMGSADD_ID   703
#define MLI_METHOD_AMGSADDE_ID  704
#define MLI_METHOD_AMGRS_ID     705

#include "utilities/utilities.h"
#include "base/mli.h"

class MLI;

/****************************************************************************
 * MLI_Method abstract class definition
 *--------------------------------------------------------------------------*/

class MLI_Method
{
   char     method_name[200];
   int      method_id;
   MPI_Comm mpi_comm;

public :

   MLI_Method( MPI_Comm comm ) 
            { mpi_comm = comm; method_id = -1; 
              strcpy(method_name, "MLI_NONE");}
   virtual ~MLI_Method() {}

   virtual int setup( MLI *mli ) {(void) mli; return -1;}
   virtual int setParams(char *name, int argc, char *argv[])
               {(void) name; (void) argc; (void) argv; return -1;}
   virtual int getParams(char *name, int *argc, char *argv[])
               {(void) name; (void) argc; (void) argv; return -1;}

   char     *getName() {return method_name;}
   int      setName( char *in_name )                                      
            {strcpy( method_name, in_name); return 0;}
   int      setID( int in_id ) {method_id = in_id; return 0;}
   int      getID()            {return method_id;}
   MPI_Comm getComm()          {return mpi_comm;}
};

extern MLI_Method *MLI_Method_CreateFromName(char *,MPI_Comm);
extern MLI_Method *MLI_Method_CreateFromID(int,MPI_Comm);

#endif

