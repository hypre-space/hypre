/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * functions for creating MLI_Method 
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * include files 
 *--------------------------------------------------------------------------*/

#ifdef WIN32
#define strcmp _stricmp
#endif

#include <string.h>
#include "amgs/mli_method.h"
#include "amgs/mli_method_amgsa.h"
#include "amgs/mli_method_amgrs.h"
#include "amgs/mli_method_amgcr.h"

/*****************************************************************************
 * constructor
 *--------------------------------------------------------------------------*/

MLI_Method::MLI_Method( MPI_Comm comm ) 
{ 
   mpi_comm = comm; 
   method_id = -1; 
   strcpy(method_name, "MLI_NONE");
}

/*****************************************************************************
 * destructor
 *--------------------------------------------------------------------------*/

MLI_Method::~MLI_Method() 
{ 
}

/*****************************************************************************
 * virtual setup function
 *--------------------------------------------------------------------------*/

int MLI_Method::setup( MLI *mli ) 
{
   (void) mli; 
   return -1;
}

/*****************************************************************************
 * virtual set parameter function
 *--------------------------------------------------------------------------*/

int MLI_Method::setParams(char *name, int argc, char *argv[])
{
   (void) name; 
   (void) argc;
   (void) argv;
   return -1;
}

/*****************************************************************************
 * virtual get parameter function
 *--------------------------------------------------------------------------*/

int MLI_Method::getParams(char *name, int *argc, char *argv[])
{
   (void) name;
   (void) argc;
   (void) argv;
   return -1;
}

/*****************************************************************************
 * get method name
 *--------------------------------------------------------------------------*/

char *MLI_Method::getName()
{
   return method_name;
}

/*****************************************************************************
 * set method name
 *--------------------------------------------------------------------------*/

int MLI_Method::setName( char *in_name )                                      
{
   strcpy(method_name, in_name);
   return 0;
}

/*****************************************************************************
 * set method ID
 *--------------------------------------------------------------------------*/

int MLI_Method::setID( int inID )                                      
{
   method_id = inID;
   return 0;
}

/*****************************************************************************
 * get method ID
 *--------------------------------------------------------------------------*/

int MLI_Method::getID()
{
   return method_id;}
}

/*****************************************************************************
 * get communicator 
 *--------------------------------------------------------------------------*/

MPI_Comm MLI_Method::getComm()
{
   return mpi_comm;
}

/*****************************************************************************
 * create a method from method name
 *--------------------------------------------------------------------------*/

MLI_Method *MLI_Method_CreateFromName( char *str, MPI_Comm mpi_comm )
{
   MLI_Method *method_ptr;
   char       paramString[80];

   if ( !strcmp(str, "AMGSA" ) )
   {
      method_ptr  = new MLI_Method_AMGSA(mpi_comm);
   }
   else if ( !strcmp(str, "AMGSAe" ) )
   {
      method_ptr  = new MLI_Method_AMGSA(mpi_comm);
      strcpy( paramString, "useSAMGe" );
      method_ptr->setParams( paramString, 0, NULL );
   }
   else if ( !strcmp(str, "AMGSADD" ) )
   {
      method_ptr  = new MLI_Method_AMGSA(mpi_comm);
      strcpy( paramString, "useSAMGDD" );
      method_ptr->setParams( paramString, 0, NULL );
      strcpy( paramString, "setNumLevels 2" );
      method_ptr->setParams( paramString, 0, NULL );
   }
   else if ( !strcmp(str, "AMGSADDe" ) )
   {
      method_ptr  = new MLI_Method_AMGSA(mpi_comm);
      strcpy( paramString, "useSAMGe" );
      method_ptr->setParams( paramString, 0, NULL );
      strcpy( paramString, "useSAMGDD" );
      method_ptr->setParams( paramString, 0, NULL );
      strcpy( paramString, "setNumLevels 2" );
      method_ptr->setParams( paramString, 0, NULL );
   }
   else if ( !strcmp(str, "AMGRS" ) )
   {
      method_ptr  = new MLI_Method_AMGRS(mpi_comm);
   }
   else if ( !strcmp(str, "AMGCR" ) )
   {
      method_ptr  = new MLI_Method_AMGCR(mpi_comm);
   }
   else
   {
      printf("MLI_Method_Create ERROR : method %s not defined.\n", str);
      printf("    valid ones are : \n\n");
      printf("    (1) AMGSA (%d)\n", MLI_METHOD_AMGSA_ID); 
      printf("    (2) AMGSAe (%d)\n", MLI_METHOD_AMGSAE_ID); 
      printf("    (3) AMGSADD (%d)\n", MLI_METHOD_AMGSADD_ID); 
      printf("    (4) AMGSADDe (%d)\n", MLI_METHOD_AMGSADDE_ID); 
      printf("    (5) AMGRS (%d)\n", MLI_METHOD_AMGRS_ID); 
      printf("    (6) AMGCR (%d)\n", MLI_METHOD_AMGCR_ID); 
      exit(1);
   }
   return method_ptr;
}

/*****************************************************************************
 * create a method from method ID
 *--------------------------------------------------------------------------*/

MLI_Method *MLI_Method_CreateFromID( int method_id, MPI_Comm mpi_comm )
{
   MLI_Method *method_ptr;
   char       paramString[80];

   switch ( method_id )
   {
      case MLI_METHOD_AMGSA_ID :
           method_ptr  = new MLI_Method_AMGSA(mpi_comm);
           break;
      case MLI_METHOD_AMGSAE_ID :
           method_ptr  = new MLI_Method_AMGSA(mpi_comm);
           strcpy( paramString, "useSAMGe" );
           method_ptr->setParams(paramString, 0, NULL);
           break;
      case MLI_METHOD_AMGSADD_ID :
           method_ptr  = new MLI_Method_AMGSA(mpi_comm);
           strcpy( paramString, "useSAMGDD" );
           method_ptr->setParams(paramString, 0, NULL);
           strcpy( paramString, "setNumLevels 2" );
           method_ptr->setParams(paramString, 0, NULL);
           break;
      case MLI_METHOD_AMGSADDE_ID :
           method_ptr  = new MLI_Method_AMGSA(mpi_comm);
           strcpy( paramString, "useSAMGe" );
           method_ptr->setParams(paramString, 0, NULL);
           strcpy( paramString, "useSAMGDD" );
           method_ptr->setParams(paramString, 0, NULL);
           strcpy( paramString, "setNumLevels 2" );
           method_ptr->setParams(paramString, 0, NULL);
           break;
      case MLI_METHOD_AMGRS_ID :
           method_ptr  = new MLI_Method_AMGRS(mpi_comm);
           break;
      case MLI_METHOD_AMGCR_ID :
           method_ptr  = new MLI_Method_AMGCR(mpi_comm);
           break;
      default :
           printf("MLI_Method_Create ERROR : method %d not defined\n",
                  method_id);
           printf("    valid ones are : \n\n");
           printf("    (1) AMGSA (%d)\n", MLI_METHOD_AMGSA_ID); 
           printf("    (2) AMGSAe (%d)\n", MLI_METHOD_AMGSAE_ID); 
           printf("    (3) AMGSADD (%d)\n", MLI_METHOD_AMGSADD_ID); 
           printf("    (4) AMGSADDe (%d)\n", MLI_METHOD_AMGSADDE_ID); 
           printf("    (5) AMGRS (%d)\n", MLI_METHOD_AMGRS_ID); 
           printf("    (6) AMGCR (%d)\n", MLI_METHOD_AMGCR_ID); 
           exit(1);
   }
   return method_ptr;
}

