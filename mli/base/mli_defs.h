/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * global definitions
 *
 *****************************************************************************/

#ifndef __MLIDEFS__
#define __MLIDEFS__

#define   MLI_FALSE                0
#define   MLI_TRUE                 1
#define   MLI_NONE                -1
#define   MLI_DEFAULT             -1
 
#define   MLI_SMOOTHER_PRE         1
#define   MLI_SMOOTHER_POST        2
#define   MLI_SMOOTHER_BOTH        3

#define   MLI_SOLVER_JACOBI_ID    301
#define   MLI_SOLVER_GS_ID        302
#define   MLI_SOLVER_SGS_ID       303
#define   MLI_SOLVER_PARASAILS_ID 304
#define   MLI_SOLVER_SCHWARZ_ID   305
#define   MLI_SOLVER_MLS_ID       306
#define   MLI_SOLVER_SUPERLU_ID   307

#define   MLI_METHOD_AMGSA_ID     701
#define   MLI_METHOD_AMGRS_ID     702

#endif

