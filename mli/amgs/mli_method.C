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
#include "../base/mli_defs.h"
#include "mli_method.h"
#include "mli_method_amgsa.h"

/*****************************************************************************
 * create a method from method name
 *--------------------------------------------------------------------------*/

MLI_Method *MLI_Method_CreateFromName( char *str, MPI_Comm mpi_comm )
{
   MLI_Method *method_ptr;

   if ( !strcmp(str, "AMGSA" ) )
   {
      method_ptr  = new MLI_Method_AMGSA(mpi_comm);
   }
   else
   {
      cout << "MLI_Method_Create ERROR : method " << str << " not defined\n";
      cout << "    valid ones are : AMGSA\n";
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

   switch ( method_id )
   {
      case MLI_METHOD_AMGSA_ID :
           method_ptr  = new MLI_Method_AMGSA(mpi_comm);
           break;
      default :
           cout << "MLI_Method_Create ERROR : method " << method_id 
                << " not defined\n";
           cout << "    valid ones are : " << MLI_METHOD_AMGSA_ID 
                << " (AMGSA)\n";
           cout.flush();
           exit(1);
   }
   return method_ptr;
}

