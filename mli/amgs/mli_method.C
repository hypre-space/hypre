/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * functions for the MLI_Method data structure
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * include files 
 *--------------------------------------------------------------------------*/

#include <string.h>
#include <iostream.h>
#include "../base/mli_defs.h"
#include "mli_method.h"
#include "mli_amgsa.h"

/*****************************************************************************
 * constructor 
 *--------------------------------------------------------------------------*/

MLI_Method::MLI_Method( char *str, MPI_Comm comm )
{
   MLI_AMGSA *amgsa;

#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_Method::MLI_Method : name = " << str << endl;
   cout.flush();
#endif

   mpi_comm = comm;
   if ( !strcmp(str, "MLI_AMGSA" ) )
   {
      strcpy( name, str );
      amgsa       = new MLI_AMGSA(comm);
      method_id   = MLI_AMGSA_ID;
      method_data = (void *) amgsa;
   }
   else if ( !strcmp(str, "MLI_AMGSA_CALIB" ) )
   {
      strcpy( name, str );
      amgsa       = new MLI_AMGSA(comm);
      method_id   = MLI_AMGSA_CALIB_ID;
      method_data = (void *) amgsa;
   }
   else
   {
      cout << "MLI_Method ERROR : method not defined = " << str << endl;
      cout.flush();
      exit(1);
   }
}

/*****************************************************************************
 * destructor 
 *--------------------------------------------------------------------------*/

MLI_Method::~MLI_Method()
{
   MLI_AMGSA *amgsa;
   if ( method_id == MLI_AMGSA_ID || method_id == MLI_AMGSA_CALIB_ID )
   {
      amgsa = (MLI_AMGSA *) method_data;
      delete amgsa;
      method_data = NULL;
   }
}  

/*****************************************************************************
 * set parameters 
 *--------------------------------------------------------------------------*/

int MLI_Method::setup( MLI *mli )
{
   int       nlevels=1;
   MLI_AMGSA *amgsa;

   if ( method_data != NULL && method_id == MLI_AMGSA_ID )
   {
      amgsa   = (MLI_AMGSA *) method_data;
      nlevels = amgsa->genMLStructure( mli );
   }
   else if ( method_data != NULL && method_id == MLI_AMGSA_CALIB_ID )
   {
      amgsa   = (MLI_AMGSA *) method_data;
      nlevels = amgsa->genMLStructureCalibration( mli );
   }
   return nlevels;
}

/*****************************************************************************
 * set parameters 
 *--------------------------------------------------------------------------*/

int MLI_Method::setParams(char *in_name, int argc, char *argv[])
{
   int        level, size, ndofs, num_ns, length, nsweeps=1, set_id;
   int        pre_post, nnodes;
   double     thresh, pweight, *nullspace, *weights=NULL, *coords, *scales;
   char       param1[256], param2[256];
   MLI_AMGSA  *amgsa;

   if ( method_id == MLI_AMGSA_ID || method_id == MLI_AMGSA_CALIB_ID )
   {
      amgsa = (MLI_AMGSA *) method_data;
      sscanf(in_name, "%s", param1);
      if ( !strcmp(param1, "setOutputLevel" ))
      {
         sscanf(in_name,"%s %d", param1, &level);
         return ( amgsa->setOutputLevel( level ) );
      }
      else if ( !strcmp(param1, "setNumLevels" ))
      {
         sscanf(in_name,"%s %d", param1, &level);
         return ( amgsa->setNumLevels( level ) );
      }
      else if ( !strcmp(param1, "setCoarsenScheme" ))
      {
         sscanf(in_name,"%s %s", param1, param2);
         if ( !strcmp(param2, "local" ) )
         {
            return ( amgsa->setCoarsenScheme( MLI_AMGSA_LOCAL ) );
         }
         else      
         {
            cout << "MLI_AMGSA ERROR : coarsen scheme not valid." << endl;
            cout << "     valid options are : local\n";
            return 1;
         }
      }
      else if ( !strcmp(param1, "setMinCoarseSize" ))
      {
         sscanf(in_name,"%s %d", param1, &size);
         return ( amgsa->setMinCoarseSize( size ) );
      }
      else if ( !strcmp(param1, "setStrengthThreshold" ))
      {
         sscanf(in_name,"%s %lg", param1, &thresh);
         return ( amgsa->setStrengthThreshold( thresh ) );
      }
      else if ( !strcmp(param1, "setPweight" ))
      {
         sscanf(in_name,"%s %lg", param1, &pweight);
         return ( amgsa->setPweight( pweight ) );
      }
      else if ( !strcmp(param1, "setCalcSpectralNorm" ))
      {
         return ( amgsa->setCalcSpectralNorm() );
      }
      else if ( !strcmp(param1, "setCalibrationSize" ))
      {
         sscanf(in_name,"%s %d", param1, &size);
         return ( amgsa->setCalibrationSize( size ) );
      }
      else if ( !strcmp(param1, "setPreSmoother" ))
      {
         sscanf(in_name,"%s %s", param1, param2);
         if      (!strcmp(param2, "Jacobi"))    set_id = MLI_SOLVER_JACOBI_ID;
         else if (!strcmp(param2, "GS"))        set_id = MLI_SOLVER_GS_ID;
         else if (!strcmp(param2, "SGS"))       set_id = MLI_SOLVER_SGS_ID;
         else if (!strcmp(param2, "Schwarz"))   set_id = MLI_SOLVER_SCHWARZ_ID;
         else if (!strcmp(param2, "MLS"))       set_id = MLI_SOLVER_MLS_ID;
         else if (!strcmp(param2, "ParaSails")) set_id = MLI_SOLVER_PARASAILS_ID;
         else 
         {
            cout << "MLI_Method ERROR : setSmoother - invalid smoother.\n";
            cout << "     valid options are : Jacobi, GS, SGS, Schwarz\n";
            cout << "                         MLS, ParaSails\n";
            return 1;
         } 
         if ( argc != 2 )
         {
            cout << "MLI_Method ERROR : setSmoother needs 2 arguments.\n";
            cout << "     argument[0] : number of relaxation sweeps \n";
            cout << "     argument[1] : relaxation weights\n";
            return 1;
         } 
         pre_post = MLI_SMOOTHER_PRE;
         nsweeps   = *(int *)   argv[0];
         weights   = (double *) argv[1];
         return ( amgsa->setSmoother(pre_post,set_id,nsweeps,weights) );
      }
      else if ( !strcmp(param1, "setPostSmoother" ))
      {
         sscanf(in_name,"%s %s", param1, param2);
         if      (!strcmp(param2, "Jacobi"))    set_id = MLI_SOLVER_JACOBI_ID;
         else if (!strcmp(param2, "GS"))        set_id = MLI_SOLVER_GS_ID;
         else if (!strcmp(param2, "SGS"))       set_id = MLI_SOLVER_SGS_ID;
         else if (!strcmp(param2, "Schwarz"))   set_id = MLI_SOLVER_SCHWARZ_ID;
         else if (!strcmp(param2, "MLS"))       set_id = MLI_SOLVER_MLS_ID;
         else if (!strcmp(param2, "ParaSails")) set_id = MLI_SOLVER_PARASAILS_ID;
         else 
         {
            cout << "MLI_Method ERROR : setSmoother - invalid smoother.\n";
            cout << "     valid options are : Jacobi, GS, SGS, Schwarz\n";
            cout << "                         MLS, ParaSails\n";
            return 1;
         } 
         if ( argc != 2 )
         {
            cout << "MLI_Method ERROR : setSmoother needs 2 arguments.\n";
            cout << "     argument[0] : number of relaxation sweeps \n";
            cout << "     argument[1] : relaxation weights\n";
            return 1;
         } 
         pre_post = MLI_SMOOTHER_POST;
         nsweeps   = *(int *)   argv[0];
         weights   = (double *) argv[1];
         return ( amgsa->setSmoother(pre_post,set_id,nsweeps,weights) );
      }
      else if ( !strcmp(param1, "setCoarseSolver" ))
      {
         sscanf(in_name,"%s %s", param1, param2);
         if      (!strcmp(param2, "Jacobi"))    set_id = MLI_SOLVER_JACOBI_ID;
         else if (!strcmp(param2, "GS"))        set_id = MLI_SOLVER_GS_ID;
         else if (!strcmp(param2, "SGS"))       set_id = MLI_SOLVER_SGS_ID;
         else if (!strcmp(param2, "Schwarz"))   set_id = MLI_SOLVER_SCHWARZ_ID;
         else if (!strcmp(param2, "ParaSails")) set_id = MLI_SOLVER_PARASAILS_ID;
         else if (!strcmp(param2, "SuperLU"))   set_id = MLI_SOLVER_SUPERLU_ID;
         else 
         {
            cerr << "MLI_Method ERROR : setCoarseSolver - invalid smoother.\n";
            cout << "     valid options are : Jacobi, GS, SGS, Schwarz\n";
            cout << "                         ParaSails, SuperLu\n";
            return 1;
         } 
         if ( set_id != MLI_SOLVER_SUPERLU_ID && argc != 2 )
         {
            cerr << "MLI_Method ERROR : setCoarseSolver needs 2 arguments.\n";
            cout << "     argument[0] : number of relaxation sweeps \n";
            cout << "     argument[1] : relaxation weights\n";
            return 1;
         } 
         else if ( set_id != MLI_SOLVER_SUPERLU_ID )
         {
            nsweeps   = *(int *)   argv[0];
            weights   = (double *) argv[1];
         }
         return ( amgsa->setCoarseSolver(set_id,nsweeps,weights) );
      }
      else if ( !strcmp(param1, "setNullSpace" ))
      {
         if ( argc != 4 )
         {
            cout << "MLI_Method ERROR : setNullSpace needs 4 arguments.\n";
            cout << "     argument[0] : node degree of freedom\n";
            cout << "     argument[1] : number of null space vectors\n";
            cout << "     argument[2] : null space information\n";
            cout << "     argument[3] : vector length\n";
            return 1;
         } 
         ndofs     = *(int *)   argv[0];
         num_ns    = *(int *)   argv[1];
         nullspace = (double *) argv[2];
         length    = *(int *)   argv[3];
         return ( amgsa->setNullSpace(ndofs,num_ns,nullspace,length) );
      }
      else if ( !strcmp(param1, "setNodalCoord" ))
      {
         if ( argc != 3 && argc != 4 )
         {
            cout << "MLI_Method ERROR : setNodalCoord needs 4 arguments.\n";
            cout << "     argument[0] : number of nodes\n";
            cout << "     argument[1] : node degree of freedom\n";
            cout << "     argument[2] : coordinate information\n";
            cout << "     argument[3] : scalings (can be null)\n";
            return 1;
         } 
         nnodes = *(int *)   argv[0];
         ndofs  = *(int *)   argv[1];
         coords = (double *) argv[2];
         if ( argc == 4 ) scales = (double *) argv[3]; else scales = NULL;
         return ( amgsa->setNodalCoordinates(nnodes,ndofs,coords,scales) );
      }
      else if ( !strcmp(param1, "reinitialize" ))
      {
         return ( amgsa->reinitialize() );
      }
      else if ( !strcmp(param1, "print" ))
      {
         return ( amgsa->print() );
      }
   }
   return 1;
}

/*****************************************************************************
 * get parameters 
 *--------------------------------------------------------------------------*/

int MLI_Method::getParams(char *in_name, int **int_vec, double **dble_vec)
{
   int        i, level, node_dofs, num_ns, length;
   double     *nullspace;
   MLI_AMGSA  *amgsa;

   (*int_vec)  = NULL;
   (*dble_vec) = NULL;
   if ( method_id == MLI_AMGSA_ID || method_id == MLI_AMGSA_CALIB_ID )
   {
      if ( !strcmp(in_name, "getNullSpace" ))
      {
         amgsa = (MLI_AMGSA *) method_data;
         amgsa->getNullSpace(node_dofs,num_ns,nullspace,length);
         (*int_vec) = new int[3];
         if ( nullspace != NULL )
         {
            (*dble_vec) = new double[length*num_ns];
            for ( i = 0; i < length*num_ns; i++ ) (*dble_vec)[i] = nullspace[i];
         }
         (*int_vec)[0] = node_dofs;
         (*int_vec)[1] = num_ns;
         (*int_vec)[2] = length;
         return 0;
      }
   }
   return 1;
}

/*****************************************************************************
 * set new name without changing anything else (only for MLI_AMGSA_CALIB)
 *--------------------------------------------------------------------------*/

int MLI_Method::setName( char *str )
{
cout << "MLI_Method::setName : name = " << str << endl;
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_Method::setName : name = " << str << endl;
   cout.flush();
#endif

   if ( !strcmp(name, "MLI_AMGSA_CALIB" ) )
   {
      strcpy( name, str );
      method_id = MLI_AMGSA_ID;
   }
   else
   {
      cout << "ML_Method::setName ERROR : not allowed - " << name << endl;
      cout.flush();
      exit(1);
   }
   return 0;
}

