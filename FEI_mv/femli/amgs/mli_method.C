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

#include <string.h>
#include <iostream.h>
#include "base/mli_defs.h"
#include "amgs/mli_method.h"
#include "amgs/mli_method_amgsa.h"

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
   }
   else if ( !strcmp(str, "AMGSADDe" ) )
   {
      method_ptr  = new MLI_Method_AMGSA(mpi_comm);
      strcpy( paramString, "useSAMGe" );
      method_ptr->setParams( paramString, 0, NULL );
      strcpy( paramString, "useSAMGDD" );
      method_ptr->setParams( paramString, 0, NULL );
   }
   else
   {
      cout << "MLI_Method_Create ERROR : method " << str << " not defined\n";
      cout << "    valid ones are : \n";
      cout << "            AMGSA\n";
      cout << "            AMGSAe\n";
      cout << "            AMGSADD\n";
      cout << "            AMGSADDe\n";
      cout.flush();
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
           break;
      case MLI_METHOD_AMGSADDE_ID :
           method_ptr  = new MLI_Method_AMGSA(mpi_comm);
           strcpy( paramString, "useSAMGe" );
           method_ptr->setParams(paramString, 0, NULL);
           strcpy( paramString, "useSAMGDD" );
           method_ptr->setParams(paramString, 0, NULL);
           break;
      default :
           cout << "MLI_Method_Create ERROR : method " << method_id 
                << " not defined\n";
           cout << "    valid ones are : " << endl;
           cout << "              " << MLI_METHOD_AMGSA_ID  << " (AMGSA)\n"; 
           cout << "              " << MLI_METHOD_AMGSAE_ID << " (AMGSAe)\n"; 
           cout << "              " << MLI_METHOD_AMGSADD_ID << " (AMGSAe)\n"; 
           cout << "              " << MLI_METHOD_AMGSADDE_ID << " (AMGSAe)\n"; 
           cout.flush();
           exit(1);
   }
   return method_ptr;
}

