/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

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
#include <strings.h>
#include "base/mli_defs.h"
#include "parcsr_mv/parcsr_mv.h"
#include "seq_mv/seq_mv.h"
#include "solver/mli_solver.h"
#include "solver/mli_solver_jacobi.h"
#include "solver/mli_solver_bjacobi.h"
#include "solver/mli_solver_gs.h"
#include "solver/mli_solver_sgs.h"
#include "solver/mli_solver_bsgs.h"
#include "solver/mli_solver_parasails.h"
#include "solver/mli_solver_mls.h"
#include "solver/mli_solver_chebyshev.h"
#include "solver/mli_solver_cg.h"
#include "solver/mli_solver_superlu.h"
#include "solver/mli_solver_seqsuperlu.h"
#include "solver/mli_solver_arpacksuperlu.h"

/*****************************************************************************
 * constructor 
 *--------------------------------------------------------------------------*/

MLI_Solver::MLI_Solver( char *str )
{
   strcpy( solver_name, str );
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
      solver_ptr->setParams( paramString, 0, NULL);
   }
   else if (!strcmp(str,"CGBJacobi")) 
   {
      solver_ptr = new MLI_Solver_CG(str);
      strcpy( paramString, "baseMethod BJacobi");
      solver_ptr->setParams( paramString, 0, NULL);
   }
   else if (!strcmp(str,"CGSGS")) 
   {
      solver_ptr = new MLI_Solver_CG(str);
      strcpy( paramString, "baseMethod SGS");
      solver_ptr->setParams( paramString, 0, NULL);
   }
   else if (!strcmp(str,"CGBSGS")) 
   {
      solver_ptr = new MLI_Solver_CG(str);
      strcpy( paramString, "baseMethod BSGS");
      solver_ptr->setParams( paramString, 0, NULL);
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
      printf("\t MCBSGS \n");
      printf("\t ParaSails \n");
      printf("\t MLS \n");
      printf("\t Chebyshev \n");
      printf("\t CGJacobi \n");
      printf("\t CGBJacobi \n");
      printf("\t CGSGS \n");
      printf("\t CGBSGS \n");
      printf("\t SuperLU\n");
      printf("\t SeqSuperLU\n");
      printf("\t ARPACKSuperLU\n"); 
      fflush(stdout);
      exit(1);
   }
   return solver_ptr;
}

