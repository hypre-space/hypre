/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.12 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * functions for the MLI_Solver
 *
 *****************************************************************************/

#define habs(x) ((x > 0) ? x : -(x))

/*--------------------------------------------------------------------------
 * include files 
 *--------------------------------------------------------------------------*/

#include <string.h>
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "seq_mv/seq_mv.h"
#include "solver/mli_solver.h"
#include "solver/mli_solver_jacobi.h"
#include "solver/mli_solver_bjacobi.h"
#include "solver/mli_solver_gs.h"
#include "solver/mli_solver_sgs.h"
#include "solver/mli_solver_hsgs.h"
#include "solver/mli_solver_bsgs.h"
#include "solver/mli_solver_hschwarz.h"
#include "solver/mli_solver_parasails.h"
#include "solver/mli_solver_mls.h"
#include "solver/mli_solver_chebyshev.h"
#include "solver/mli_solver_cg.h"
#include "solver/mli_solver_gmres.h"
#include "solver/mli_solver_superlu.h"
#include "solver/mli_solver_seqsuperlu.h"
#include "solver/mli_solver_arpacksuperlu.h"
#include "solver/mli_solver_kaczmarz.h"
#include "solver/mli_solver_mli.h"
#include "solver/mli_solver_amg.h"

/*****************************************************************************
 * constructor 
 *--------------------------------------------------------------------------*/

MLI_Solver::MLI_Solver(char *str)
{
   strcpy( solver_name, str );
}

/*****************************************************************************
 * destructor 
 *--------------------------------------------------------------------------*/

MLI_Solver::~MLI_Solver()
{
}

/*****************************************************************************
 * getName function 
 *--------------------------------------------------------------------------*/

char* MLI_Solver::getName()
{ 
   return solver_name;
}

/*****************************************************************************
 * set parameter function 
 *--------------------------------------------------------------------------*/

int MLI_Solver::setParams(char *paramString,int argc,char **argv)  
{
   (void) paramString;
   (void) argc;
   (void) argv;
   return -1;
}

/*****************************************************************************
 * get parameter function 
 *--------------------------------------------------------------------------*/

int MLI_Solver::getParams(char *paramString,int *argc,char **argv)
{
   (void) paramString;
   (void) argc;
   (void) argv;
   return -1;
}

/*****************************************************************************
 * create a solver from its name 
 *--------------------------------------------------------------------------*/

MLI_Solver *MLI_Solver_CreateFromName( char *str )
{
   char       paramString[100];
   MLI_Solver *solver_ptr=NULL;

   if      (!strcmp(str,"Jacobi"))  solver_ptr = new MLI_Solver_Jacobi(str);
   else if (!strcmp(str,"BJacobi")) solver_ptr = new MLI_Solver_BJacobi(str);
   else if (!strcmp(str,"GS"))      solver_ptr = new MLI_Solver_GS(str);
   else if (!strcmp(str,"SGS"))     solver_ptr = new MLI_Solver_SGS(str);
   else if (!strcmp(str,"HSGS"))    solver_ptr = new MLI_Solver_HSGS(str);
   else if (!strcmp(str,"HSchwarz"))solver_ptr = new MLI_Solver_HSchwarz(str);
   else if (!strcmp(str,"MCSGS")) 
   {
      solver_ptr = new MLI_Solver_SGS(str);
      strcpy( paramString, "setScheme multicolor");
      solver_ptr->setParams( paramString, 0, NULL);
   }
   else if (!strcmp(str,"BSGS"))    solver_ptr = new MLI_Solver_BSGS(str);
   else if (!strcmp(str,"MCBSGS")) 
   {
      solver_ptr = new MLI_Solver_BSGS(str);
      strcpy( paramString, "setScheme multicolor");
      solver_ptr->setParams( paramString, 0, NULL);
   }
   else if (!strcmp(str,"ParaSails"))solver_ptr = new MLI_Solver_ParaSails(str);
   else if (!strcmp(str,"MLS"))      solver_ptr = new MLI_Solver_MLS(str);
   else if (!strcmp(str,"Chebyshev"))solver_ptr = new MLI_Solver_Chebyshev(str);
   else if (!strcmp(str,"CGJacobi")) 
   {
      solver_ptr = new MLI_Solver_CG(str);
      strcpy( paramString, "baseMethod Jacobi");
      solver_ptr->setParams(paramString, 0, NULL);
   }
   else if (!strcmp(str,"CGBJacobi")) 
   {
      solver_ptr = new MLI_Solver_CG(str);
      strcpy( paramString, "baseMethod BJacobi");
      solver_ptr->setParams(paramString, 0, NULL);
   }
   else if (!strcmp(str,"CGSGS")) 
   {
      solver_ptr = new MLI_Solver_CG(str);
      strcpy( paramString, "baseMethod SGS");
      solver_ptr->setParams(paramString, 0, NULL);
   }
   else if (!strcmp(str,"CGBSGS")) 
   {
      solver_ptr = new MLI_Solver_CG(str);
      strcpy( paramString, "baseMethod BSGS");
      solver_ptr->setParams(paramString, 0, NULL);
   }
   else if (!strcmp(str,"CGMLI")) 
   {
      solver_ptr = new MLI_Solver_CG(str);
      strcpy( paramString, "baseMethod MLI");
      solver_ptr->setParams(paramString, 0, NULL);
   }
   else if (!strcmp(str,"CGAMG")) 
   {
      solver_ptr = new MLI_Solver_CG(str);
      strcpy( paramString, "baseMethod AMG");
      solver_ptr->setParams(paramString, 0, NULL);
   }
   else if (!strcmp(str,"CGILU")) 
   {
      solver_ptr = new MLI_Solver_CG(str);
      strcpy( paramString, "baseMethod ILU");
      solver_ptr->setParams(paramString, 0, NULL);
   }
   else if (!strcmp(str,"GMRESJacobi")) 
   {
      solver_ptr = new MLI_Solver_GMRES(str);
      strcpy( paramString, "baseMethod Jacobi");
      solver_ptr->setParams(paramString, 0, NULL);
   }
   else if (!strcmp(str,"GMRESSGS")) 
   {
      solver_ptr = new MLI_Solver_GMRES(str);
      strcpy( paramString, "baseMethod SGS");
      solver_ptr->setParams(paramString, 0, NULL);
   }
   else if (!strcmp(str,"GMRESMLI")) 
   {
      solver_ptr = new MLI_Solver_GMRES(str);
      strcpy( paramString, "baseMethod MLI");
      solver_ptr->setParams(paramString, 0, NULL);
   }
   else if (!strcmp(str, "Kaczmarz"))   
   {
      solver_ptr = new MLI_Solver_Kaczmarz(str);
   }
   else if (!strcmp(str,"SuperLU"))   
   {
#ifdef MLI_SUPERLU
      solver_ptr = new MLI_Solver_SuperLU(str);
#else
      printf("MLI_Solver_Create ERROR : SuperLU not available\n");
      exit(1);
#endif
   }
   else if (!strcmp(str,"SeqSuperLU"))   
   {
#ifdef MLI_SUPERLU
      solver_ptr = new MLI_Solver_SeqSuperLU(str);
#else
      printf("MLI_Solver_Create ERROR : SuperLU not available\n");
      exit(1);
#endif
   }
   else if (!strcmp(str, "ARPACKSuperLU"))   
   {
#ifdef MLI_SUPERLU
      solver_ptr = new MLI_Solver_ARPACKSuperLU(str);
#else
      printf("MLI_Solver_Create ERROR : SuperLU not available\n");
      exit(1);
#endif
   }
   else
   {
      printf("MLI_Solver_Create ERROR : solver %s undefined.\n",str);
      printf("Valid ones are : \n");
      printf("\t Jacobi \n");
      printf("\t BJacobi \n");
      printf("\t GS \n");
      printf("\t SGS \n");
      printf("\t MCSGS \n");
      printf("\t BSGS \n");
      printf("\t HSGS (BoomerAMG SGS) \n");
      printf("\t HSchwarz (BoomerAMG Schwarz) \n");
      printf("\t MCBSGS \n");
      printf("\t ParaSails \n");
      printf("\t MLS \n");
      printf("\t Chebyshev \n");
      printf("\t CGJacobi \n");
      printf("\t CGBJacobi \n");
      printf("\t CGSGS \n");
      printf("\t CGBSGS \n");
      printf("\t CGMLI \n");
      printf("\t CGAMG \n");
      printf("\t CGILU \n");
      printf("\t GMRESJacobi \n");
      printf("\t GMRESSGS \n");
      printf("\t GMRESMLI \n");
      printf("\t Kaczmarz\n"); 
      printf("\t SuperLU\n");
      printf("\t SeqSuperLU\n");
      printf("\t ARPACKSuperLU\n"); 
      fflush(stdout);
      exit(1);
   }
   return solver_ptr;
}

