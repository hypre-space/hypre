/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * functions for the MLI_MethodAgent 
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * include files 
 *--------------------------------------------------------------------------*/

#include <string.h>
#include <iostream.h>
#include "../base/mli_defs.h"
#include "mli_methodAgent.h"
#include "mli_method_amgsa.h"

/*****************************************************************************
 * constructor 
 *--------------------------------------------------------------------------*/

MLI_MethodAgent::MLI_MethodAgent( MPI_Comm comm )
{
   strcpy( name, "MLI_NONE" );
   mpi_comm    = comm;
   method_id   = -1;
   method_ptr  = NULL;
}

/*****************************************************************************
 * destructor 
 *--------------------------------------------------------------------------*/

MLI_MethodAgent::~MLI_MethodAgent()
{
   MLI_Method_AMGSA *amgsa;

   if ( method_id == MLI_METHOD_AMGSA_ID )
   {
      amgsa = (MLI_Method_AMGSA *) method_ptr;
      delete amgsa;
      method_ptr = NULL;
   }
}

/*****************************************************************************
 * create a method 
 *--------------------------------------------------------------------------*/

int MLI_MethodAgent::createMethod( char *str )
{
   MLI_Method_AMGSA *amgsa;

   if ( !strcmp(str, "MLI_METHOD_AMGSA" ) )
   {
      strcpy( name, str );
      amgsa       = new MLI_Method_AMGSA(mpi_comm);
      method_id   = MLI_METHOD_AMGSA_ID;
      method_ptr  = (MLI_Method *) amgsa;
   }
   else
   {
      cout << "MLI_MethodAgent ERROR : method " << str << " not defined\n";
      cout.flush();
      exit(1);
   }
   return 0;
}

/*****************************************************************************
 * give a created method (give ownership also) 
 *--------------------------------------------------------------------------*/

int MLI_MethodAgent::giveMethod(MLI_Method *method)
{
   if ( method == NULL )
   {
      cout << "MLI_MethodAgent giveMethod ERROR : method NULL.\n";
      cout.flush();
      exit(1);
   }
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_MethodAgent giveMethod : probing method name.\n";
#endif
   if ( !strcmp(method->getName(), "MLI_METHOD_AMGSA" ) )
   {
      strcpy( name, "MLI_METHOD_AMGSA" );
      method_id   = MLI_METHOD_AMGSA_ID;
      method_ptr  = method;
   }
   else
   {
      cout << "MLI_MethodAgent giveMethod ERROR : method not defined\n";
      cout.flush();
      exit(1);
   }
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_MethodAgent giveMethod : successful.\n";
#endif
   return 0;
}

/*****************************************************************************
 * take the created method (take ownership also) 
 *--------------------------------------------------------------------------*/

MLI_Method *MLI_MethodAgent::takeMethod()
{
   MLI_Method *ret_method;

   if ( method_ptr  == NULL )
   {
      cout << "MLI_MethodAgent::takeMethod ERROR : method not created yet.\n";
      cout.flush();
      return ((MLI_Method *) NULL);
   }
   else 
   {
      ret_method  = method_ptr ;
      method_ptr  = NULL;
      return ( ret_method );
   }
}

/*****************************************************************************
 * ask the method itself to set up the multigrid structure (return nlevels) 
 *--------------------------------------------------------------------------*/

int MLI_MethodAgent::setup( MLI *mli )
{
   if ( method_ptr != NULL ) return (method_ptr->setup( mli ));
   else
   {
      cout << "MLI_MethodAgent setup ERROR : no method to set up " << endl;
      cout.flush();
      exit(1);
   }
   return -1;
}

/*****************************************************************************
 * set parameters 
 *--------------------------------------------------------------------------*/

int MLI_MethodAgent::setParams(char *in_name, int argc, char *argv[])
{
   if ( method_ptr != NULL ) 
      return (method_ptr->setParams(in_name, argc, argv));
   else
   {
      cout << "MLI_MethodAgent setParams ERROR : method = NULL.\n";
      cout.flush();
      exit(1);
   }
   return -1;
}

/*****************************************************************************
 * get parameters 
 *--------------------------------------------------------------------------*/

int MLI_MethodAgent::getParams(char *in_name, int *argc, char *argv[])
{
   if ( method_ptr != NULL ) 
      return (method_ptr->getParams(in_name, argc, argv));
   else
   {
      cout << "MLI_MethodAgent getParams ERROR : method = NULL.\n";
      cout.flush();
      exit(1);
   }
   return -1;
}

