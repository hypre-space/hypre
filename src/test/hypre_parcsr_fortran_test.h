/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.5 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 * Definitions of ParCSR Fortran interface routines
 *****************************************************************************/

#define HYPRE_ParCSRMatrixCreate  \
        hypre_F90_NAME(fhypre_parcsrmatrixcreate, FHYPRE_PARCSRMATRIXCREATE)
extern void hypre_F90_NAME(fhypre_parcsrmatrixcreate, FHYPRE_PARCSRMATRIXCREATE)
            (HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_ParCSRMatrixDestroy  \
        hypre_F90_NAME(fhypre_parcsrmatrixdestroy, FHYPRE_PARCSRMATRIXDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrmatrixdestroy, FHYPRE_PARCSRMATRIXDESTROY)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRMatrixInitialize  \
        hypre_F90_NAME(fhypre_parcsrmatrixinitialize, FHYPRE_PARCSRMATRIXINITIALIZE)
extern void hypre_F90_NAME(fhypre_parcsrmatrixinitialize, FHYPRE_PARCSRMATRIXINITIALIZE)
                      (hypre_F90_Obj *);

#define HYPRE_ParCSRMatrixRead  \
        hypre_F90_NAME(fhypre_parcsrmatrixread, FHYPRE_PARCSRMATRIXREAD)
extern void hypre_F90_NAME(fhypre_parcsrmatrixread, FHYPRE_PARCSRMATRIXREAD)
                      (HYPRE_Int *, char *, hypre_F90_Obj *);

#define HYPRE_ParCSRMatrixPrint  \
        hypre_F90_NAME(fhypre_parcsrmatrixprint, FHYPRE_PARCSRMATRIXPRINT)
extern void hypre_F90_NAME(fhypre_parcsrmatrixprint, FHYPRE_PARCSRMATRIXPRINT)
                      (hypre_F90_Obj *, char *, HYPRE_Int *);

#define HYPRE_ParCSRMatrixGetComm  \
        hypre_F90_NAME(fhypre_parcsrmatrixgetcomm, FHYPRE_PARCSRMATRIXGETCOMM)
extern void hypre_F90_NAME(fhypre_parcsrmatrixgetcomm, FHYPRE_PARCSRMATRIXGETCOMM)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRMatrixGetDims  \
        hypre_F90_NAME(fhypre_parcsrmatrixgetdims, FHYPRE_PARCSRMATRIXGETDIMS)
extern void hypre_F90_NAME(fhypre_parcsrmatrixgetdims, FHYPRE_PARCSRMATRIXGETDIMS)
                      (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *);

#define HYPRE_ParCSRMatrixGetRowPartitioning  \
        hypre_F90_NAME(fhypre_parcsrmatrixgetrowpartit, FHYPRE_PARCSRMATRIXGETROWPARTIT)
extern void hypre_F90_NAME(fhypre_parcsrmatrixgetrowpartit, FHYPRE_PARCSRMATRIXGETROWPARTIT)
                      (hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRMatrixGetColPartitioning  \
        hypre_F90_NAME(fhypre_parcsrmatrixgetcolpartit, FHYPRE_PARCSRMATRIXGETCOLPARTIT)
extern void hypre_F90_NAME(fhypre_parcsrmatrixgetcolpartit, FHYPRE_PARCSRMATRIXGETCOLPARTIT)
                      (hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRMatrixGetLocalRange  \
        hypre_F90_NAME(fhypre_parcsrmatrixgetlocalrang, FHYPRE_PARCSRMATRIXGETLOCALRANG)
extern void hypre_F90_NAME(fhypre_parcsrmatrixgetlocalrang, FHYPRE_PARCSRMATRIXGETLOCALRANG)
                      (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *);

#define HYPRE_ParCSRMatrixGetRow  \
        hypre_F90_NAME(fhypre_parcsrmatrixgetrow, FHYPRE_PARCSRMATRIXGETROW)
extern void hypre_F90_NAME(fhypre_parcsrmatrixgetrow, FHYPRE_PARCSRMATRIXGETROW)
                      (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRMatrixRestoreRow  \
        hypre_F90_NAME(fhypre_parcsrmatrixrestorerow, FHYPRE_PARCSRMATRIXRESTOREROW)
extern void hypre_F90_NAME(fhypre_parcsrmatrixrestorerow, FHYPRE_PARCSRMATRIXRESTOREROW)
                      (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_CSRMatrixtoParCSRMatrix  \
        hypre_F90_NAME(fhypre_csrmatrixtoparcsrmatrix, FHYPRE_CSRMATRIXTOPARCSRMATRIX)
extern void hypre_F90_NAME(fhypre_csrmatrixtoparcsrmatrix, FHYPRE_CSRMATRIXTOPARCSRMATRIX)
                      (HYPRE_Int *, hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_ParCSRMatrixMatvec  \
        hypre_F90_NAME(fhypre_parcsrmatrixmatvec, FHYPRE_PARCSRMATRIXMATVEC)
extern void hypre_F90_NAME(fhypre_parcsrmatrixmatvec, FHYPRE_PARCSRMATRIXMATVEC)
                      (double *, hypre_F90_Obj *, hypre_F90_Obj *, double *, hypre_F90_Obj *);  

#define HYPRE_ParCSRMatrixMatvecT  \
        hypre_F90_NAME(fhypre_parcsrmatrixmatvect, FHYPRE_PARCSRMATRIXMATVECT)
extern void hypre_F90_NAME(fhypre_parcsrmatrixmatvect, FHYPRE_PARCSRMATRIXMATVECT)
                      (double *, hypre_F90_Obj *, hypre_F90_Obj *, double *, hypre_F90_Obj *);



#define HYPRE_ParVectorCreate  \
        hypre_F90_NAME(fhypre_parvectorcreate, FHYPRE_PARVECTORCREATE)
extern void hypre_F90_NAME(fhypre_parvectorcreate, FHYPRE_PARVECTORCREATE)
                      (HYPRE_Int *, HYPRE_Int *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParMultiVectorCreate  \
        hypre_F90_NAME(fhypre_parmultivectorcreate, FHYPRE_PARMULTIVECTORCREATE)
extern void hypre_F90_NAME(fhypre_parmultivectorcreate, FHYPRE_PARMULTIVECTORCREATE)
                      (HYPRE_Int *, HYPRE_Int *, hypre_F90_Obj *, HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_ParVectorDestroy  \
        hypre_F90_NAME(fhypre_parvectordestroy, FHYPRE_PARVECTORDESTROY)
extern void hypre_F90_NAME(fhypre_parvectordestroy, FHYPRE_PARVECTORDESTROY)
                      (hypre_F90_Obj *);

#define HYPRE_ParVectorInitialize  \
        hypre_F90_NAME(fhypre_parvectorinitialize, FHYPRE_PARVECTORINITIALIZE)
extern void hypre_F90_NAME(fhypre_parvectorinitialize, FHYPRE_PARVECTORINITIALIZE)
                      (hypre_F90_Obj *);

#define HYPRE_ParVectorRead  \
        hypre_F90_NAME(fhypre_parvectorread, FHYPRE_PARVECTORREAD)
extern void hypre_F90_NAME(fhypre_parvectorread, FHYPRE_PARVECTORREAD)
                      (HYPRE_Int *, hypre_F90_Obj *, char *);

#define HYPRE_ParVectorPrint  \
        hypre_F90_NAME(fhypre_parvectorprint, FHYPRE_PARVECTORPRINT)
extern void hypre_F90_NAME(fhypre_parvectorprint, FHYPRE_PARVECTORPRINT)
                      (hypre_F90_Obj *, char *, HYPRE_Int *);

#define HYPRE_ParVectorSetConstantValues  \
        hypre_F90_NAME(fhypre_parvectorsetconstantvalu, FHYPRE_PARVECTORSETCONSTANTVALU)
extern void hypre_F90_NAME(fhypre_parvectorsetconstantvalu, FHYPRE_PARVECTORSETCONSTANTVALU)
                      (hypre_F90_Obj *, double *);

#define HYPRE_ParVectorSetRandomValues  \
        hypre_F90_NAME(fhypre_parvectorsetrandomvalues, FHYPRE_PARVECTORSETRANDOMVALUES)
extern void hypre_F90_NAME(fhypre_parvectorsetrandomvalues, FHYPRE_PARVECTORSETRANDOMVALUES)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParVectorCopy  \
        hypre_F90_NAME(fhypre_parvectorcopy, FHYPRE_PARVECTORCOPY)
extern void hypre_F90_NAME(fhypre_parvectorcopy, FHYPRE_PARVECTORCOPY)
                      (hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParVectorCloneShallow  \
        hypre_F90_NAME(fhypre_parvectorcloneshallow, FHYPRE_PARVECTORCLONESHALLOW)
extern void hypre_F90_NAME(fhypre_parvectorcloneshallow, FHYPRE_PARVECTORCLONESHALLOW)
                      (hypre_F90_Obj *);

#define HYPRE_ParVectorScale  \
        hypre_F90_NAME(fhypre_parvectorscale, FHYPRE_PARVECTORSCALE)
extern void hypre_F90_NAME(fhypre_parvectorscale, FHYPRE_PARVECTORSCALE)
                      (hypre_F90_Obj *, double *);

#define HYPRE_ParVectorAxpy  \
        hypre_F90_NAME(fhypre_parvectoraxpy, FHYPRE_PARVECTORAXPY)
extern void hypre_F90_NAME(fhypre_parvectoraxpy, FHYPRE_PARVECTORAXPY)
                      (double *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParVectorInnerProd  \
        hypre_F90_NAME(fhypre_parvectorinnerprod, FHYPRE_PARVECTORINNERPROD)
extern void hypre_F90_NAME(fhypre_parvectorinnerprod, FHYPRE_PARVECTORINNERPROD)
                      (hypre_F90_Obj *, hypre_F90_Obj *, double *);

#define hypre_ParCSRMatrixGlobalNumRows  \
        hypre_F90_NAME(fhypre_parcsrmatrixglobalnumrow, FHYPRE_PARCSRMATRIXGLOBALNUMROW)
extern void hypre_F90_NAME(fhypre_parcsrmatrixglobalnumrow, FHYPRE_PARCSRMATRIXGLOBALNUMROW)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define hypre_ParCSRMatrixRowStarts  \
        hypre_F90_NAME(fhypre_parcsrmatrixrowstarts, FHYPRE_PARCSRMATRIXROWSTARTS)
extern void hypre_F90_NAME(fhypre_parcsrmatrixrowstarts, FHYPRE_PARCSRMATRIXROWSTARTS)
                      (hypre_F90_Obj *, hypre_F90_Obj *);

#define hypre_ParVectorSetDataOwner  \
        hypre_F90_NAME(fhypre_setparvectordataowner, FHYPRE_SETPARVECTORDATAOWNER)
extern void hypre_F90_NAME(fhypre_setparvectordataowner, FHYPRE_SETPARVECTORDATAOWNER)
                      (hypre_F90_Obj *, HYPRE_Int *);



#define GenerateLaplacian  \
        hypre_F90_NAME(fgeneratelaplacian, FHYPRE_GENERATELAPLACIAN)
extern void hypre_F90_NAME(fgeneratelaplacian, FHYPRE_GENERATELAPLACIAN)
                      (HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *,
                       HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, double *, hypre_F90_Obj *);



#define HYPRE_BoomerAMGCreate  \
        hypre_F90_NAME(fhypre_boomeramgcreate, FHYPRE_BOOMERAMGCREATE)
extern void hypre_F90_NAME(fhypre_boomeramgcreate, FHYPRE_BOOMERAMGCREATE)
                      (hypre_F90_Obj *);

#define HYPRE_BoomerAMGDestroy  \
        hypre_F90_NAME(fhypre_boomeramgdestroy, FHYPRE_BOOMERAMGDESTROY)
extern void hypre_F90_NAME(fhypre_boomeramgdestroy, FHYPRE_BOOMERAMGDESTROY)
                      (hypre_F90_Obj *);

#define HYPRE_BoomerAMGSetup  \
        hypre_F90_NAME(fhypre_boomeramgsetup, FHYPRE_BOOMERAMGSETUP)
extern void hypre_F90_NAME(fhypre_boomeramgsetup, FHYPRE_BOOMERAMGSETUP)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_BoomerAMGSolve  \
        hypre_F90_NAME(fhypre_boomeramgsolve, FHYPRE_BOOMERAMGSOLVE)
extern void hypre_F90_NAME(fhypre_boomeramgsolve, FHYPRE_BOOMERAMGSOLVE)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_BoomerAMGSolveT  \
        hypre_F90_NAME(fhypre_boomeramgsolvet, FHYPRE_BOOMERAMGSOLVET)
extern void hypre_F90_NAME(fhypre_boomeramgsolvet, FHYPRE_BOOMERAMGSOLVET)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_BoomerAMGSetRestriction  \
        hypre_F90_NAME(fhypre_boomeramgsetrestriction, FHYPRE_BOOMERAMGSETRESTRICTION)
extern void hypre_F90_NAME(fhypre_boomeramgsetrestriction, FHYPRE_BOOMERAMGSETRESTRICTION)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetMaxLevels  \
        hypre_F90_NAME(fhypre_boomeramgsetmaxlevels, FHYPRE_BOOMERAMGSETMAXLEVELS)
extern void hypre_F90_NAME(fhypre_boomeramgsetmaxlevels, FHYPRE_BOOMERAMGSETMAXLEVELS)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetMaxLevels  \
        hypre_F90_NAME(fhypre_boomeramggetmaxlevels, FHYPRE_BOOMERAMGGETMAXLEVELS)
extern void hypre_F90_NAME(fhypre_boomeramggetmaxlevels, FHYPRE_BOOMERAMGGETMAXLEVELS)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetStrongThreshold  \
        hypre_F90_NAME(fhypre_boomeramgsetstrongthrshl, FHYPRE_BOOMERAMGSETSTRONGTHRSHL)
extern void hypre_F90_NAME(fhypre_boomeramgsetstrongthrshl, FHYPRE_BOOMERAMGSETSTRONGTHRSHL)
                      (hypre_F90_Obj *, double *);

#define HYPRE_BoomerAMGGetStrongThreshold  \
        hypre_F90_NAME(fhypre_boomeramggetstrongthrshl, FHYPRE_BOOMERAMGGETSTRONGTHRSHL)
extern void hypre_F90_NAME(fhypre_boomeramggetstrongthrshl, FHYPRE_BOOMERAMGGETSTRONGTHRSHL)
                      (hypre_F90_Obj *, double *);

#define HYPRE_BoomerAMGSetMaxRowSum  \
        hypre_F90_NAME(fhypre_boomeramgsetmaxrowsum, FHYPRE_BOOMERAMGSETMAXROWSUM)
extern void hypre_F90_NAME(fhypre_boomeramgsetmaxrowsum, FHYPRE_BOOMERAMGSETMAXROWSUM)
                      (hypre_F90_Obj *, double *);

#define HYPRE_BoomerAMGGetMaxRowSum  \
        hypre_F90_NAME(fhypre_boomeramggetmaxrowsum, FHYPRE_BOOMERAMGGETMAXROWSUM)
extern void hypre_F90_NAME(fhypre_boomeramggetmaxrowsum, FHYPRE_BOOMERAMGGETMAXROWSUM)
                      (hypre_F90_Obj *, double *);

#define HYPRE_BoomerAMGSetTruncFactor  \
        hypre_F90_NAME(fhypre_boomeramgsettruncfactor, FHYPRE_BOOMERAMGSETTRUNCFACTOR)
extern void hypre_F90_NAME(fhypre_boomeramgsettruncfactor, FHYPRE_BOOMERAMGSETTRUNCFACTOR)
                      (hypre_F90_Obj *, double *);

#define HYPRE_BoomerAMGGetTruncFactor  \
        hypre_F90_NAME(fhypre_boomeramggettruncfactor, FHYPRE_BOOMERAMGGETTRUNCFACTOR)
extern void hypre_F90_NAME(fhypre_boomeramggettruncfactor, FHYPRE_BOOMERAMGGETTRUNCFACTOR)
                      (hypre_F90_Obj *, double *);

#define HYPRE_BoomerAMGSetSCommPkgSwitch  \
        hypre_F90_NAME(fhypre_boomeramgsetscommpkgswit, FHYPRE_BOOMERAMGSETSCOMMPKGSWIT)
extern void hypre_F90_NAME(fhypre_boomeramgsetscommpkgswit, FHYPRE_BOOMERAMGSETSCOMMPKGSWIT)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetInterpType  \
        hypre_F90_NAME(fhypre_boomeramgsetinterptype, FHYPRE_BOOMERAMGSETINTERPTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetinterptype, FHYPRE_BOOMERAMGSETINTERPTYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetMinIter  \
        hypre_F90_NAME(fhypre_boomeramgsetminiter, FHYPRE_BOOMERAMGSETMINITER)
extern void hypre_F90_NAME(fhypre_boomeramgsetminiter, FHYPRE_BOOMERAMGSETMINITER)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetMaxIter  \
        hypre_F90_NAME(fhypre_boomeramgsetmaxiter, FHYPRE_BOOMERAMGSETMAXITER)
extern void hypre_F90_NAME(fhypre_boomeramgsetmaxiter, FHYPRE_BOOMERAMGSETMAXITER)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetMaxIter  \
        hypre_F90_NAME(fhypre_boomeramggetmaxiter, FHYPRE_BOOMERAMGGETMAXITER)
extern void hypre_F90_NAME(fhypre_boomeramggetmaxiter, FHYPRE_BOOMERAMGGETMAXITER)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetCoarsenType  \
        hypre_F90_NAME(fhypre_boomeramgsetcoarsentype, FHYPRE_BOOMERAMGSETCOARSENTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetcoarsentype, FHYPRE_BOOMERAMGSETCOARSENTYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetCoarsenType  \
        hypre_F90_NAME(fhypre_boomeramggetcoarsentype, FHYPRE_BOOMERAMGGETCOARSENTYPE)
extern void hypre_F90_NAME(fhypre_boomeramggetcoarsentype, FHYPRE_BOOMERAMGGETCOARSENTYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetMeasureType  \
        hypre_F90_NAME(fhypre_boomeramgsetmeasuretype, FHYPRE_BOOMERAMGSETMEASURETYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetmeasuretype, FHYPRE_BOOMERAMGSETMEASURETYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetMeasureType  \
        hypre_F90_NAME(fhypre_boomeramggetmeasuretype, FHYPRE_BOOMERAMGGETMEASURETYPE)
extern void hypre_F90_NAME(fhypre_boomeramggetmeasuretype, FHYPRE_BOOMERAMGGETMEASURETYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetSetupType  \
        hypre_F90_NAME(fhypre_boomeramgsetsetuptype, FHYPRE_BOOMERAMGSETSETUPTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetsetuptype, FHYPRE_BOOMERAMGSETSETUPTYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetCycleType  \
        hypre_F90_NAME(fhypre_boomeramgsetcycletype, FHYPRE_BOOMERAMGSETCYCLETYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetcycletype, FHYPRE_BOOMERAMGSETCYCLETYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetCycleType  \
        hypre_F90_NAME(fhypre_boomeramggetcycletype, FHYPRE_BOOMERAMGGETCYCLETYPE)
extern void hypre_F90_NAME(fhypre_boomeramggetcycletype, FHYPRE_BOOMERAMGGETCYCLETYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetTol  \
        hypre_F90_NAME(fhypre_boomeramgsettol, FHYPRE_BOOMERAMGSETTOL)
extern void hypre_F90_NAME(fhypre_boomeramgsettol, FHYPRE_BOOMERAMGSETTOL)
                      (hypre_F90_Obj *, double *);

#define HYPRE_BoomerAMGGetTol  \
        hypre_F90_NAME(fhypre_boomeramggettol, FHYPRE_BOOMERAMGGETTOL)
extern void hypre_F90_NAME(fhypre_boomeramggettol, FHYPRE_BOOMERAMGGETTOL)
                      (hypre_F90_Obj *, double *);

#define HYPRE_BoomerAMGSetNumSweeps  \
        hypre_F90_NAME(fhypre_boomeramgsetnumsweeps, FHYPRE_BOOMERAMGSETNUMSWEEPS)
extern void hypre_F90_NAME(fhypre_boomeramgsetnumsweeps, FHYPRE_BOOMERAMGSETNUMSWEEPS)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetCycleNumSweeps  \
        hypre_F90_NAME(fhypre_boomeramgsetcyclenumswee, FHYPRE_BOOMERAMGSETCYCLENUMSWEE)
extern void hypre_F90_NAME(fhypre_boomeramgsetcyclenumswee, FHYPRE_BOOMERAMGSETCYCLENUMSWEE)
                      (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetCycleNumSweeps  \
        hypre_F90_NAME(fhypre_boomeramggetcyclenumswee, FHYPRE_BOOMERAMGGETCYCLENUMSWEE)
extern void hypre_F90_NAME(fhypre_boomeramggetcyclenumswee, FHYPRE_BOOMERAMGGETCYCLENUMSWEE)
                      (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *);

#define HYPRE_BoomerAMGInitGridRelaxation  \
        hypre_F90_NAME(fhypre_boomeramginitgridrelaxat, FHYPRE_BOOMERAMGINITGRIDRELAXAT)
extern void hypre_F90_NAME(fhypre_boomeramginitgridrelaxat, FHYPRE_BOOMERAMGINITGRIDRELAXAT)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, HYPRE_Int *, hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGFinalizeGridRelaxation  \
        hypre_F90_NAME(fhypre_boomeramgfingridrelaxatn, FHYPRE_BOOMERAMGFINGRIDRELAXATN)
extern void hypre_F90_NAME(fhypre_boomeramgfingridrelaxatn, FHYPRE_BOOMERAMGFINGRIDRELAXATN)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_BoomerAMGSetRelaxType  \
        hypre_F90_NAME(fhypre_boomeramgsetrelaxtype, FHYPRE_BOOMERAMGSETRELAXTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetrelaxtype, FHYPRE_BOOMERAMGSETRELAXTYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetRelaxType  \
        hypre_F90_NAME(fhypre_boomeramgsetrelaxtype, FHYPRE_BOOMERAMGSETRELAXTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetrelaxtype, FHYPRE_BOOMERAMGSETRELAXTYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetRelaxOrder  \
        hypre_F90_NAME(fhypre_boomeramgsetrelaxorder, FHYPRE_BOOMERAMGSETRELAXORDER)
extern void hypre_F90_NAME(fhypre_boomeramgsetrelaxorder, FHYPRE_BOOMERAMGSETRELAXORDER)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetRelaxWeight  \
        hypre_F90_NAME(fhypre_boomeramgsetrelaxweight, FHYPRE_BOOMERAMGSETRELAXWEIGHT)
extern void hypre_F90_NAME(fhypre_boomeramgsetrelaxweight, FHYPRE_BOOMERAMGSETRELAXWEIGHT)
                      (hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_BoomerAMGSetRelaxWt  \
        hypre_F90_NAME(fhypre_boomeramgsetrelaxwt, FHYPRE_BOOMERAMGSETRELAXWT)
extern void hypre_F90_NAME(fhypre_boomeramgsetrelaxwt, FHYPRE_BOOMERAMGSETRELAXWT)
                      (hypre_F90_Obj *, double *);

#define HYPRE_BoomerAMGSetLevelRelaxWt  \
        hypre_F90_NAME(fhypre_boomeramgsetlevelrelaxwt, FHYPRE_BOOMERAMGSETLEVELRELAXWT)
extern void hypre_F90_NAME(fhypre_boomeramgsetlevelrelaxwt, FHYPRE_BOOMERAMGSETLEVELRELAXWT)
                      (hypre_F90_Obj *, double *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetOuterWt  \
        hypre_F90_NAME(fhypre_boomeramgsetouterwt, FHYPRE_BOOMERAMGSETOUTERWT)
extern void hypre_F90_NAME(fhypre_boomeramgsetouterwt, FHYPRE_BOOMERAMGSETOUTERWT)
                      (hypre_F90_Obj *, double *);

#define HYPRE_BoomerAMGSetLevelOuterWt  \
        hypre_F90_NAME(fhypre_boomeramgsetlevelouterwt, FHYPRE_BOOMERAMGSETLEVELOUTERWT)
extern void hypre_F90_NAME(fhypre_boomeramgsetlevelouterwt, FHYPRE_BOOMERAMGSETLEVELOUTERWT)
                      (hypre_F90_Obj *, double *);

#define HYPRE_BoomerAMGSetSmoothType  \
        hypre_F90_NAME(fhypre_boomeramgsetsmoothtype, FHYPRE_BOOMERAMGSETSMOOTHTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetsmoothtype, FHYPRE_BOOMERAMGSETSMOOTHTYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetSmoothType  \
        hypre_F90_NAME(fhypre_boomeramggetsmoothtype, FHYPRE_BOOMERAMGGETSMOOTHTYPE)
extern void hypre_F90_NAME(fhypre_boomeramggetsmoothtype, FHYPRE_BOOMERAMGGETSMOOTHTYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetSmoothNumLvls  \
        hypre_F90_NAME(fhypre_boomeramgsetsmoothnumlvl, FHYPRE_BOOMERAMGSETSMOOTHNUMLVL)
extern void hypre_F90_NAME(fhypre_boomeramgsetsmoothnumlvl, FHYPRE_BOOMERAMGSETSMOOTHNUMLVL)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetSmoothNumLvls  \
        hypre_F90_NAME(fhypre_boomeramggetsmoothnumlvl, FHYPRE_BOOMERAMGGETSMOOTHNUMLVL)
extern void hypre_F90_NAME(fhypre_boomeramggetsmoothnumlvl, FHYPRE_BOOMERAMGGETSMOOTHNUMLVL)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetSmoothNumSwps  \
        hypre_F90_NAME(fhypre_boomeramgsetsmoothnumswp, FHYPRE_BOOMERAMGSETSMOOTHNUMSWP)
extern void hypre_F90_NAME(fhypre_boomeramgsetsmoothnumswp, FHYPRE_BOOMERAMGSETSMOOTHNUMSWP)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetSmoothNumSwps  \
        hypre_F90_NAME(fhypre_boomeramggetsmoothnumswp, FHYPRE_BOOMERAMGGETSMOOTHNUMSWP)
extern void hypre_F90_NAME(fhypre_boomeramggetsmoothnumswp, FHYPRE_BOOMERAMGGETSMOOTHNUMSWP)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetLogging  \
        hypre_F90_NAME(fhypre_boomeramgsetlogging, FHYPRE_BOOMERAMGSETLOGGING)
extern void hypre_F90_NAME(fhypre_boomeramgsetlogging, FHYPRE_BOOMERAMGSETLOGGING)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetLogging  \
        hypre_F90_NAME(fhypre_boomeramggetlogging, FHYPRE_BOOMERAMGGETLOGGING)
extern void hypre_F90_NAME(fhypre_boomeramggetlogging, FHYPRE_BOOMERAMGGETLOGGING)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetPrintLevel  \
        hypre_F90_NAME(fhypre_boomeramgsetprintlevel, FHYPRE_BOOMERAMGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_boomeramgsetprintlevel, FHYPRE_BOOMERAMGSETPRINTLEVEL)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetPrintLevel  \
        hypre_F90_NAME(fhypre_boomeramggetprintlevel, FHYPRE_BOOMERAMGGETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_boomeramggetprintlevel, FHYPRE_BOOMERAMGGETPRINTLEVEL)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetPrintFileName  \
        hypre_F90_NAME(fhypre_boomeramgsetprintfilenam, FHYPRE_BOOMERAMGSETPRINTFILENAM)
extern void hypre_F90_NAME(fhypre_boomeramgsetprintfilenam, FHYPRE_BOOMERAMGSETPRINTFILENAM)
                      (hypre_F90_Obj *, char *);

#define HYPRE_BoomerAMGSetDebugFlag  \
        hypre_F90_NAME(fhypre_boomeramgsetdebugflag, FHYPRE_BOOMERAMGSETDEBUGFLAG)
extern void hypre_F90_NAME(fhypre_boomeramgsetdebugflag, FHYPRE_BOOMERAMGSETDEBUGFLAG)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetDebugFlag  \
        hypre_F90_NAME(fhypre_boomeramggetdebugflag, FHYPRE_BOOMERAMGGETDEBUGFLAG)
extern void hypre_F90_NAME(fhypre_boomeramggetdebugflag, FHYPRE_BOOMERAMGGETDEBUGFLAG)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetNumIterations  \
        hypre_F90_NAME(fhypre_boomeramggetnumiteration, FHYPRE_BOOMERAMGGETNUMITERATION)
extern void hypre_F90_NAME(fhypre_boomeramggetnumiteration, FHYPRE_BOOMERAMGGETNUMITERATION)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetCumNumIterations  \
        hypre_F90_NAME(fhypre_boomeramggetcumnumiterat, FHYPRE_BOOMERAMGGETCUMNUMITERAT)
extern void hypre_F90_NAME(fhypre_boomeramggetcumnumiterat, FHYPRE_BOOMERAMGGETCUMNUMITERAT)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetResidual  \
        hypre_F90_NAME(fhypre_boomeramggetresidual, FHYPRE_BOOMERAMGGETRESIDUAL)
extern void hypre_F90_NAME(fhypre_boomeramggetresidual, FHYPRE_BOOMERAMGGETRESIDUAL)
                      (hypre_F90_Obj *, double *);

#define HYPRE_BoomerAMGGetFinalRelativeResidualNorm  \
        hypre_F90_NAME(fhypre_boomeramggetfinalreltvre, FHYPRE_BOOMERAMGGETFINALRELTVRE)
extern void hypre_F90_NAME(fhypre_boomeramggetfinalreltvre, FHYPRE_BOOMERAMGGETFINALRELTVRE)
                      (hypre_F90_Obj *, double *);

#define HYPRE_BoomerAMGSetVariant  \
        hypre_F90_NAME(fhypre_boomeramgsetvariant, FHYPRE_BOOMERAMGSETVARIANT)
extern void hypre_F90_NAME(fhypre_boomeramgsetvariant, FHYPRE_BOOMERAMGSETVARIANT)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetVariant  \
        hypre_F90_NAME(fhypre_boomeramggetvariant, FHYPRE_BOOMERAMGGETVARIANT)
extern void hypre_F90_NAME(fhypre_boomeramggetvariant, FHYPRE_BOOMERAMGGETVARIANT)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetOverlap  \
        hypre_F90_NAME(fhypre_boomeramgsetoverlap, FHYPRE_BOOMERAMGSETOVERLAP)
extern void hypre_F90_NAME(fhypre_boomeramgsetoverlap, FHYPRE_BOOMERAMGSETOVERLAP)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetOverlap  \
        hypre_F90_NAME(fhypre_boomeramggetoverlap, FHYPRE_BOOMERAMGGETOVERLAP)
extern void hypre_F90_NAME(fhypre_boomeramggetoverlap, FHYPRE_BOOMERAMGGETOVERLAP)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetDomainType  \
        hypre_F90_NAME(fhypre_boomeramgsetdomaintype, FHYPRE_BOOMERAMGSETDOMAINTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetdomaintype, FHYPRE_BOOMERAMGSETDOMAINTYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetDomainType  \
        hypre_F90_NAME(fhypre_boomeramggetdomaintype, FHYPRE_BOOMERAMGGETDOMAINTYPE)
extern void hypre_F90_NAME(fhypre_boomeramggetdomaintype, FHYPRE_BOOMERAMGGETDOMAINTYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetSchwarzRlxWt  \
        hypre_F90_NAME(fhypre_boomeramgsetschwarzrlxwt, FHYPRE_BOOMERAMGSETSCHWARZRLXWT)
extern void hypre_F90_NAME(fhypre_boomeramgsetschwarzrlxwt, FHYPRE_BOOMERAMGSETSCHWARZRLXWT)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetSchwarzRlxWt  \
        hypre_F90_NAME(fhypre_boomeramggetschwarzrlxwt, FHYPRE_BOOMERAMGGETSCHWARZRLXWT)
extern void hypre_F90_NAME(fhypre_boomeramggetschwarzrlxwt, FHYPRE_BOOMERAMGGETSCHWARZRLXWT)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetSym  \
        hypre_F90_NAME(fhypre_boomeramgsetsym, FHYPRE_BOOMERAMGSETSYM)
extern void hypre_F90_NAME(fhypre_boomeramgsetsym, FHYPRE_BOOMERAMGSETSYM)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetLevel  \
        hypre_F90_NAME(fhypre_boomeramgsetlevel, FHYPRE_BOOMERAMGSETLEVEL)
extern void hypre_F90_NAME(fhypre_boomeramgsetlevel, FHYPRE_BOOMERAMGSETLEVEL)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetFilter  \
        hypre_F90_NAME(fhypre_boomeramgsetfilter, FHYPRE_BOOMERAMGSETFILTER)
extern void hypre_F90_NAME(fhypre_boomeramgsetfilter, FHYPRE_BOOMERAMGSETFILTER)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetDropTol  \
        hypre_F90_NAME(fhypre_boomeramgsetdroptol, FHYPRE_BOOMERAMGSETDROPTOL)
extern void hypre_F90_NAME(fhypre_boomeramgsetdroptol, FHYPRE_BOOMERAMGSETDROPTOL)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetMaxNzPerRow  \
        hypre_F90_NAME(fhypre_boomeramgsetmaxnzperrow, FHYPRE_BOOMERAMGSETMAXNZPERROW)
extern void hypre_F90_NAME(fhypre_boomeramgsetmaxnzperrow, FHYPRE_BOOMERAMGSETMAXNZPERROW)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetEuclidFile  \
        hypre_F90_NAME(fhypre_boomeramgseteuclidfile, FHYPRE_BOOMERAMGSETEUCLIDFILE)
extern void hypre_F90_NAME(fhypre_boomeramgseteuclidfile, FHYPRE_BOOMERAMGSETEUCLIDFILE)
                      (hypre_F90_Obj *, char *);

#define HYPRE_BoomerAMGSetNumFunctions  \
        hypre_F90_NAME(fhypre_boomeramgsetnumfunctions, FHYPRE_BOOMERAMGSETNUMFUNCTIONS)
extern void hypre_F90_NAME(fhypre_boomeramgsetnumfunctions, FHYPRE_BOOMERAMGSETNUMFUNCTIONS)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGGetNumFunctions  \
        hypre_F90_NAME(fhypre_boomeramggetnumfunctions, FHYPRE_BOOMERAMGGETNUMFUNCTIONS)
extern void hypre_F90_NAME(fhypre_boomeramggetnumfunctions, FHYPRE_BOOMERAMGGETNUMFUNCTIONS)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetNodal  \
        hypre_F90_NAME(fhypre_boomeramgsetnodal, FHYPRE_BOOMERAMGSETNODAL)
extern void hypre_F90_NAME(fhypre_boomeramgsetnodal, FHYPRE_BOOMERAMGSETNODAL)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetDofFunc  \
        hypre_F90_NAME(fhypre_boomeramgsetdoffunc, FHYPRE_BOOMERAMGSETDOFFUNC)
extern void hypre_F90_NAME(fhypre_boomeramgsetdoffunc, FHYPRE_BOOMERAMGSETDOFFUNC)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetNumPaths  \
        hypre_F90_NAME(fhypre_boomeramgsetnumpaths, FHYPRE_BOOMERAMGSETNUMPATHS)
extern void hypre_F90_NAME(fhypre_boomeramgsetnumpaths, FHYPRE_BOOMERAMGSETNUMPATHS)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetAggNumLevels  \
        hypre_F90_NAME(fhypre_boomeramgsetaggnumlevels, FHYPRE_BOOMERAMGSETAGGNUMLEVELS)
extern void hypre_F90_NAME(fhypre_boomeramgsetaggnumlevels, FHYPRE_BOOMERAMGSETAGGNUMLEVELS)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetGSMG  \
        hypre_F90_NAME(fhypre_boomeramgsetgsmg, FHYPRE_BOOMERAMGSETGSMG)
extern void hypre_F90_NAME(fhypre_boomeramgsetgsmg, FHYPRE_BOOMERAMGSETGSMG)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BoomerAMGSetNumSamples  \
        hypre_F90_NAME(fhypre_boomeramgsetnumsamples, FHYPRE_BOOMERAMGSETNUMSAMPLES)
extern void hypre_F90_NAME(fhypre_boomeramgsetsamples, FHYPRE_BOOMERAMGSETNUMSAMPLES)
                      (hypre_F90_Obj *, HYPRE_Int *);



#define HYPRE_ParCSRBiCGSTABCreate  \
        hypre_F90_NAME(fhypre_parcsrbicgstabcreate, FHYPRE_PARCSRBICGSTABCREATE)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabcreate, FHYPRE_PARCSRBICGSTABCREATE)
                      (HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_ParCSRBiCGSTABDestroy  \
        hypre_F90_NAME(fhypre_parcsrbicgstabdestroy, FHYPRE_PARCSRBICGSTABDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabdestroy, FHYPRE_PARCSRBICGSTABDESTROY)
                      (hypre_F90_Obj *);

#define HYPRE_ParCSRBiCGSTABSetup  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetup, FHYPRE_PARCSRBICGSTABSETUP)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetup, FHYPRE_PARCSRBICGSTABSETUP)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRBiCGSTABSolve  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsolve, FHYPRE_PARCSRBICGSTABSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsolve, FHYPRE_PARCSRBICGSTABSOLVE)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRBiCGSTABSetTol  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsettol, FHYPRE_PARCSRBICGSTABSETTOL)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsettol, FHYPRE_PARCSRBICGSTABSETTOL)
                      (hypre_F90_Obj *, double *);

#define HYPRE_ParCSRBiCGSTABSetMinIter  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetminiter, FHYPRE_PARCSRBICGSTABSETMINITER)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetminiter, FHYPRE_PARCSRBICGSTABSETMINITER)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRBiCGSTABSetMaxIter  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetmaxiter, FHYPRE_PARCSRBICGSTABSETMAXITER)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetmaxiter, FHYPRE_PARCSRBICGSTABSETMAXITER)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRBiCGSTABSetStopCrit  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetstopcri, FHYPRE_PARCSRBICGSTABSETSTOPCRI)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetstopcri, FHYPRE_PARCSRBICGSTABSETSTOPCRI)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRBiCGSTABSetPrecond  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetprecond, FHYPRE_PARCSRBICGSTABSETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetprecond, FHYPRE_PARCSRBICGSTABSETPRECOND)
                      (hypre_F90_Obj *, HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_ParCSRBiCGSTABGetPrecond  \
        hypre_F90_NAME(fhypre_parcsrbicgstabgetprecond, FHYPRE_PARCSRBICGSTABGETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabgetprecond, FHYPRE_PARCSRBICGSTABGETPRECOND)
                      (hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRBiCGSTABSetLogging  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetlogging, FHYPRE_PARCSRBICGSTABSETLOGGING)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetlogging, FHYPRE_PARCSRBICGSTABSETLOGGING)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRBiCGSTABSetPrintLevel  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetprintle, FHYPRE_PARCSRBICGSTABSETPRINTLE)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetprintle, FHYPRE_PARCSRBICGSTABSETPRINTLE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRBiCGSTABGetNumIter  \
        hypre_F90_NAME(fhypre_parcsrbicgstabgetnumiter, FHYPRE_PARCSRBICGSTABGETNUMITER)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabgetnumiter, FHYPRE_PARCSRBICGSTABGETNUMITER)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRBiCGSTABGetFinalRel  \
        hypre_F90_NAME(fhypre_parcsrbicgstabgetfinalre, FHYPRE_PARCSRBICGSTABGETFINALRE)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabgetfinalre, FHYPRE_PARCSRBICGSTABGETFINALRE)
                      (hypre_F90_Obj *, double *);



#define HYPRE_BlockTridiagCreate  \
	hypre_F90_NAME(fhypre_blocktridiagcreate, FHYPRE_BLOCKTRIDIAGCREATE)
extern void hypre_F90_NAME(fhypre_blocktridiagcreate, FHYPRE_BLOCKTRIDIAGCREATE)
                      (hypre_F90_Obj *);

#define HYPRE_BlockTridiagDestroy  \
	hypre_F90_NAME(fhypre_blocktridiagdestroy, FHYPRE_BLOCKTRIDIAGDESTROY)
extern void hypre_F90_NAME(fhypre_blocktridiagdestroy, FHYPRE_BLOCKTRIDIAGDESTROY)
                      (hypre_F90_Obj *);

#define HYPRE_BlockTridiagSetup  \
	hypre_F90_NAME(fhypre_blocktridiagsetup, FHYPRE_BLOCKTRIDIAGSETUP)
extern void hypre_F90_NAME(fhypre_blocktridiagsetup, FHYPRE_BLOCKTRIDIAGSETUP)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_BlockTridiagSolve  \
	hypre_F90_NAME(fhypre_blocktridiagsolve, FHYPRE_BLOCKTRIDIAGSOLVE)
extern void hypre_F90_NAME(fhypre_blocktridiagsolve, FHYPRE_BLOCKTRIDIAGSOLVE)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_BlockTridiagSetIndexSet  \
	hypre_F90_NAME(fhypre_blocktridiagsetindexset, FHYPRE_BLOCKTRIDIAGSETINDEXSET)
extern void hypre_F90_NAME(fhypre_blocktridiagsetindexset, FHYPRE_BLOCKTRIDIAGSETINDEXSET)
                      (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *);

#define HYPRE_BlockTridiagSetAMGStrengthThreshold  \
	hypre_F90_NAME(fhypre_blocktridiagsetamgstreng, FHYPRE_BLOCKTRIDIAGSETAMGSTRENG)
extern void hypre_F90_NAME(fhypre_blocktridiagsetamgstreng, FHYPRE_BLOCKTRIDIAGSETAMGSTRENG)
                      (hypre_F90_Obj *, double *);

#define HYPRE_BlockTridiagSetAMGNumSweeps  \
	hypre_F90_NAME(fhypre_blocktridiagsetamgnumswe, FHYPRE_BLOCKTRIDIAGSETAMGNUMSWE)
extern void hypre_F90_NAME(fhypre_blocktridiagsetamgnumswe, FHYPRE_BLOCKTRIDIAGSETAMGNUMSWE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BlockTridiagSetAMGRelaxType  \
	hypre_F90_NAME(fhypre_blocktridiagsetamgrelaxt, FHYPRE_BLOCKTRIDIAGSETAMGRELAXT)
extern void hypre_F90_NAME(fhypre_blocktridiagsetamgrelaxt, FHYPRE_BLOCKTRIDIAGSETAMGRELAXT)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_BlockTridiagSetPrintLevel  \
	hypre_F90_NAME(fhypre_blocktridiagsetprintleve, FHYPRE_BLOCKTRIDIAGSETPRINTLEVE)
extern void hypre_F90_NAME(fhypre_blocktridiagsetprintleve, FHYPRE_BLOCKTRIDIAGSETPRINTLEVE)
                      (hypre_F90_Obj *, HYPRE_Int *);



#define HYPRE_ParCSRCGNRCreate  \
	hypre_F90_NAME(fhypre_parcsrcgnrcreate, FHYPRE_PARCSRCGNRCREATE)
extern void hypre_F90_NAME(fhypre_parcsrcgnrcreate, FHYPRE_PARCSRCGNRCREATE)
                      (HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_ParCSRCGNRDestroy  \
        hypre_F90_NAME(fhypre_parcsrcgnrdestroy, FHYPRE_PARCSRCGNRDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrcgnrdestroy, FHYPRE_PARCSRCGNRDESTROY)
                      (hypre_F90_Obj *);

#define HYPRE_ParCSRCGNRSetup  \
        hypre_F90_NAME(fhypre_parcsrcgnrsetup, FHYPRE_PARCSRCGNRSETUP)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsetup, FHYPRE_PARCSRCGNRSETUP)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRCGNRSolve  \
        hypre_F90_NAME(fhypre_parcsrcgnrsolve, FHYPRE_PARCSRCGNRSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsolve, FHYPRE_PARCSRCGNRSOLVE)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRCGNRSetTol  \
        hypre_F90_NAME(fhypre_parcsrcgnrsettol, FHYPRE_PARCSRCGNRSETTOL)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsettol, FHYPRE_PARCSRCGNRSETTOL)
                      (hypre_F90_Obj *, double *);

#define HYPRE_ParCSRCGNRSetMinIter  \
        hypre_F90_NAME(fhypre_parcsrcgnrsetminiter, FHYPRE_PARCSRCGNRSETMINITER)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsetminiter, FHYPRE_PARCSRCGNRSETMINITER)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRCGNRSetMaxIter  \
        hypre_F90_NAME(fhypre_parcsrcgnrsetmaxiter, FHYPRE_PARCSRCGNRSETMAXITER)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsetmaxiter, FHYPRE_PARCSRCGNRSETMAXITER)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRCGNRSetStopCrit  \
        hypre_F90_NAME(fhypre_parcsrcgnrsetstopcri, FHYPRE_PARCSRCGNRSETSTOPCRI)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsetstopcri, FHYPRE_PARCSRCGNRSETSTOPCRI)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRCGNRSetPrecond  \
        hypre_F90_NAME(fhypre_parcsrcgnrsetprecond, FHYPRE_PARCSRCGNRSETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsetprecond, FHYPRE_PARCSRCGNRSETPRECOND)
                      (hypre_F90_Obj *, HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_ParCSRCGNRGetPrecond  \
        hypre_F90_NAME(fhypre_parcsrcgnrgetprecond, FHYPRE_PARCSRCGNRGETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrcgnrgetprecond, FHYPRE_PARCSRCGNRGETPRECOND)
                      (hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRCGNRSetLogging  \
        hypre_F90_NAME(fhypre_parcsrcgnrsetlogging, FHYPRE_PARCSRCGNRSETLOGGING)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsetlogging, FHYPRE_PARCSRCGNRSETLOGGING)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRCGNRGetNumIteration  \
        hypre_F90_NAME(fhypre_parcsrcgnrgetnumiteratio, FHYPRE_PARCSRCGNRGETNUMITERATIO)
extern void hypre_F90_NAME(fhypre_parcsrcgnrgetnumiteratio, FHYPRE_PARCSRCGNRGETNUMITERATIO)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm  \
        hypre_F90_NAME(fhypre_parcsrcgnrgetfinalrelati, FHYPRE_PARCSRCGNRGETFINALRELATI)
extern void hypre_F90_NAME(fhypre_parcsrcgnrgetfinalrelati, FHYPRE_PARCSRCGNRGETFINALRELATI)
                      (hypre_F90_Obj *, double *);



#define HYPRE_EuclidCreate  \
        hypre_F90_NAME(fhypre_euclidcreate, FHYPRE_EUCLIDCREATE)
extern void hypre_F90_NAME(fhypre_euclidcreate, FHYPRE_EUCLIDCREATE)
                      (HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_EuclidDestroy  \
        hypre_F90_NAME(fhypre_eucliddestroy, FHYPRE_EUCLIDDESTROY)
extern void hypre_F90_NAME(fhypre_eucliddestroy, FHYPRE_EUCLIDDESTROY)
                      (hypre_F90_Obj *);

#define HYPRE_EuclidSetup  \
        hypre_F90_NAME(fhypre_euclidsetup, FHYPRE_EUCLIDSETUP)
extern void hypre_F90_NAME(fhypre_euclidsetup, FHYPRE_EUCLIDSETUP)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_EuclidSolve  \
        hypre_F90_NAME(fhypre_euclidsolve, FHYPRE_EUCLIDSOLVE)
extern void hypre_F90_NAME(fhypre_euclidsolve, FHYPRE_EUCLIDSOLVE)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_EuclidSetParams  \
        hypre_F90_NAME(fhypre_euclidsetparams, FHYPRE_EUCLIDSETPARAMS)
extern void hypre_F90_NAME(fhypre_euclidsetparams, FHYPRE_EUCLIDSETPARAMS)
                      (hypre_F90_Obj *, HYPRE_Int *, char *);

#define HYPRE_EuclidSetParamsFromFile  \
        hypre_F90_NAME(fhypre_euclidsetparamsfromfile, FHYPRE_EUCLIDSETPARAMSFROMFILE)
extern void hypre_F90_NAME(fhypre_euclidsetparamsfromfile, FHYPRE_EUCLIDSETPARAMSFROMFILE)
                      (hypre_F90_Obj *, char *);



#define HYPRE_ParCSRGMRESCreate  \
        hypre_F90_NAME(fhypre_parcsrgmrescreate, FHYPRE_PARCSRGMRESCREATE)
extern void hypre_F90_NAME(fhypre_parcsrgmrescreate, FHYPRE_PARCSRGMRESCREATE)
                      (HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_ParCSRGMRESDestroy  \
        hypre_F90_NAME(fhypre_parcsrgmresdestroy, FHYPRE_PARCSRGMRESDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrgmresdestroy, FHYPRE_PARCSRGMRESDESTROY)
                      (hypre_F90_Obj *);

#define HYPRE_ParCSRGMRESSetup  \
        hypre_F90_NAME(fhypre_parcsrgmressetup, FHYPRE_PARCSRGMRESSETUP)
extern void hypre_F90_NAME(fhypre_parcsrgmressetup, FHYPRE_PARCSRGMRESSETUP)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRGMRESSolve  \
        hypre_F90_NAME(fhypre_parcsrgmressolve, FHYPRE_PARCSRGMRESSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrgmressolve, FHYPRE_PARCSRGMRESSOLVE)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRGMRESSetKDim  \
        hypre_F90_NAME(fhypre_parcsrgmressetkdim, FHYPRE_PARCSRGMRESSETKDIM)
extern void hypre_F90_NAME(fhypre_parcsrgmressetkdim, FHYPRE_PARCSRGMRESSETKDIM)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRGMRESSetTol  \
        hypre_F90_NAME(fhypre_parcsrgmressettol, FHYPRE_PARCSRGMRESSETTOL)
extern void hypre_F90_NAME(fhypre_parcsrgmressettol, FHYPRE_PARCSRGMRESSETTOL)
                      (hypre_F90_Obj *, double *);

#define HYPRE_ParCSRGMRESSetMinIter  \
        hypre_F90_NAME(fhypre_parcsrgmressetminiter, FHYPRE_PARCSRGMRESSETMINITER)
extern void hypre_F90_NAME(fhypre_parcsrgmressetminiter, FHYPRE_PARCSRGMRESSETMINITER)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRGMRESSetMaxIter  \
        hypre_F90_NAME(fhypre_parcsrgmressetmaxiter, FHYPRE_PARCSRGMRESSETMAXITER)
extern void hypre_F90_NAME(fhypre_parcsrgmressetmaxiter, FHYPRE_PARCSRGMRESSETMAXITER)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRGMRESSetPrecond  \
        hypre_F90_NAME(fhypre_parcsrgmressetprecond, FHYPRE_PARCSRGMRESSETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrgmressetprecond, FHYPRE_PARCSRGMRESSETPRECOND)
                      (hypre_F90_Obj *, HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_ParCSRGMRESGetPrecond  \
        hypre_F90_NAME(fhypre_parcsrgmresgetprecond, FHYPRE_PARCSRGMRESGETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrgmresgetprecond, FHYPRE_PARCSRGMRESGETPRECOND)
                      (hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRGMRESSetLogging  \
        hypre_F90_NAME(fhypre_parcsrgmressetlogging, FHYPRE_PARCSRGMRESSETLOGGING)
extern void hypre_F90_NAME(fhypre_parcsrgmressetlogging, FHYPRE_PARCSRGMRESSETLOGGING)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRGMRESSetPrintLevel  \
        hypre_F90_NAME(fhypre_parcsrgmressetprintlevel, FHYPRE_PARCSRGMRESSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_parcsrgmressetprintlevel, FHYPRE_PARCSRGMRESSETPRINTLEVEL)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRGMRESGetNumIterations  \
        hypre_F90_NAME(fhypre_parcsrgmresgetnumiterati, FHYPRE_PARCSRGMRESGETNUMITERATI)
extern void hypre_F90_NAME(fhypre_parcsrgmresgetnumiterati, FHYPRE_PARCSRGMRESGETNUMITERATI)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm  \
        hypre_F90_NAME(fhypre_parcsrgmresgetfinalrelat, FHYPRE_PARCSRGMRESGETFINALRELAT)
extern void hypre_F90_NAME(fhypre_parcsrgmresgetfinalrelat, FHYPRE_PARCSRGMRESGETFINALRELAT)
                      (hypre_F90_Obj *, double *);



#define HYPRE_ParCSRHybridCreate \
        hypre_F90_NAME(fhypre_parcsrhybridcreate, FHYPRE_PARCSRHYBRIDCREATE)
extern void hypre_F90_NAME(fhypre_parcsrhybridcreate, FHYPRE_PARCSRHYBRIDCREATE)
                      (hypre_F90_Obj *);

#define HYPRE_ParCSRHybridDestroy \
        hypre_F90_NAME(fhypre_parcsrhybriddestroy, FHYPRE_PARCSRHYBRIDDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrhybriddestroy, FHYPRE_PARCSRHYBRIDDESTROY)
                      (hypre_F90_Obj *);

#define HYPRE_ParCSRHybridSetup \
        hypre_F90_NAME(fhypre_parcsrhybridsetup, FHYPRE_PARCSRHYBRIDSETUP)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetup, FHYPRE_PARCSRHYBRIDSETUP)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRHybridSolve \
        hypre_F90_NAME(fhypre_parcsrhybridsolve, FHYPRE_PARCSRHYBRIDSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrhybridsolve, FHYPRE_PARCSRHYBRIDSOLVE)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRHybridSetTol \
        hypre_F90_NAME(fhypre_parcsrhybridsettol, FHYPRE_PARCSRHYBRIDSETTOL)
extern void hypre_F90_NAME(fhypre_parcsrhybridsettol, FHYPRE_PARCSRHYBRIDSETTOL)
                      (hypre_F90_Obj *, double *);

#define HYPRE_ParCSRHybridSetConvergenceTol \
        hypre_F90_NAME(fhypre_parcsrhybridsetconvergen, FHYPRE_PARCSRHYBRIDSETCONVERGEN)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetconvergen, FHYPRE_PARCSRHYBRIDSETCONVERGEN)
                      (hypre_F90_Obj *, double *);

#define HYPRE_ParCSRHybridSetDSCGMaxIter \
        hypre_F90_NAME(fhypre_parcsrhybridsetdscgmaxit, FHYPRE_PARCSRHYBRIDSETDSCGMAXIT)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetdscgmaxit, FHYPRE_PARCSRHYBRIDSETDSCGMAXIT)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetPCGMaxIter \
        hypre_F90_NAME(fhypre_parcsrhybridsetpcgmaxite, FHYPRE_PARCSRHYBRIDSETPCGMAXITE)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetpcgmaxite, FHYPRE_PARCSRHYBRIDSETPCGMAXITE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetSolverType \
        hypre_F90_NAME(fhypre_parcsrhybridsetsolvertyp, FHYPRE_PARCSRHYBRIDSETSOLVERTYP)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetsolvertyp, FHYPRE_PARCSRHYBRIDSETSOLVERTYP)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetKDim \
        hypre_F90_NAME(fhypre_parcsrhybridsetkdim, FHYPRE_PARCSRHYBRIDSETKDIM)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetkdim, FHYPRE_PARCSRHYBRIDSETKDIM)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetTwoNorm \
        hypre_F90_NAME(fhypre_parcsrhybridsettwonorm, FHYPRE_PARCSRHYBRIDSETTWONORM)
extern void hypre_F90_NAME(fhypre_parcsrhybridsettwonorm, FHYPRE_PARCSRHYBRIDSETTWONORM)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetStopCrit \
        hypre_F90_NAME(fhypre_parcsrhybridsetstopcrit, FHYPRE_PARCSRSETSTOPCRIT)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetstopcrit, FHYPRE_PARCSRSETSTOPCRIT)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetRelChange \
        hypre_F90_NAME(fhypre_parcsrhybridsetrelchange, FHYPRE_PARCSRHYBRIDSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetrelchange, FHYPRE_PARCSRHYBRIDSETRELCHANGE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetPrecond \
        hypre_F90_NAME(fhypre_parcsrhybridsetprecond, FHYPRE_PARCSRHYBRIDSETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetprecond, FHYPRE_PARCSRHYBRIDSETPRECOND)
                      (hypre_F90_Obj *, HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_ParCSRHybridSetLogging \
        hypre_F90_NAME(fhypre_parcsrhybridsetlogging, FHYPRE_PARCSRHYBRIDSETLOGGING)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetlogging, FHYPRE_PARCSRHYBRIDSETLOGGING)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetPrintLevel \
        hypre_F90_NAME(fhypre_parcsrhybridsetprintleve, FHYPRE_PARCSRHYBRIDSETPRINTLEVE)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetprintleve, FHYPRE_PARCSRHYBRIDSETPRINTLEVE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetStrongThreshold \
        hypre_F90_NAME(fhypre_parcsrhybridsetstrongthr, FHYPRE_PARCSRHYBRIDSETSTRONGTHR)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetstrongthr, FHYPRE_PARCSRHYBRIDSETSTRONGTHR)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetMaxRowSum \
        hypre_F90_NAME(fhypre_parcsrhybridsetmaxrowsum, FHYPRE_PARCSRHYBRIDSETMAXROWSUM)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetmaxrowsum, FHYPRE_PARCSRHYBRIDSETMAXROWSUM)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetTruncFactor \
        hypre_F90_NAME(fhypre_parcsrhybridsettruncfact, FHYPRE_PARCSRHYBRIDSETTRUNCFACT)
extern void hypre_F90_NAME(fhypre_parcsrhybridsettruncfact, FHYPRE_PARCSRHYBRIDSETTRUNCFACT)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetMaxLevels \
        hypre_F90_NAME(fhypre_parcsrhybridsetmaxlevels, FHYPRE_PARCSRHYBRIDSETMAXLEVELS)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetmaxlevels, FHYPRE_PARCSRHYBRIDSETMAXLEVELS)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetMeasureType \
        hypre_F90_NAME(fhypre_parcsrhybridsetmeasurety, FHYPRE_PARCSRHYBRIDSETMEASURETY)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetmeasurety, FHYPRE_PARCSRHYBRIDSETMEASURETY)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetCoarsenType \
        hypre_F90_NAME(fhypre_parcsrhybridsetcoarsenty, FHYPRE_PARCSRHYBRIDSETCOARSENTY)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetcoarsenty, FHYPRE_PARCSRHYBRIDSETCOARSENTY)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetCycleType \
        hypre_F90_NAME(fhypre_parcsrhybridsetcycletype, FHYPRE_PARCSRHYBRIDSETCYCLETYPE)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetcycletype, FHYPRE_PARCSRHYBRIDSETCYCLETYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetNumGridSweeps \
        hypre_F90_NAME(fhypre_parcsrhybridsetnumgridsw, FHYPRE_PARCSRHYBRIDSETNUMGRIDSW)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetnumgridsw, FHYPRE_PARCSRHYBRIDSETNUMGRIDSW)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetGridRelaxType \
        hypre_F90_NAME(fhypre_parcsrhybridsetgridrlxty, FHYPRE_PARCSRHYBRIDSETGRIDRLXTY)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetgridrlxty, FHYPRE_PARCSRHYBRIDSETGRIDRLXTY)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetGridRelaxPoints \
        hypre_F90_NAME(fhypre_parcsrhybridsetgridrlxpt, FHYPRE_PARCSRHYBRIDSETGRIDRLXPT)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetgridrlxpt, FHYPRE_PARCSRHYBRIDSETGRIDRLXPT)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetNumSweeps \
        hypre_F90_NAME(fhypre_parcsrhybridsetnumsweeps, FHYPRE_PARCSRHYBRIDSETNUMSWEEPS)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetnumsweeps, FHYPRE_PARCSRHYBRIDSETNUMSWEEPS)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetCycleNumSweeps \
        hypre_F90_NAME(fhypre_parcsrhybridsetcyclenums, FHYPRE_PARCSRHYBRIDSETCYCLENUMS)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetcyclenums, FHYPRE_PARCSRHYBRIDSETCYCLENUMS)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetRelaxType \
        hypre_F90_NAME(fhypre_parcsrhybridsetrelaxtype, FHYPRE_PARCSRHYBRIDSETRELAXTYPE)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetrelaxtype, FHYPRE_PARCSRHYBRIDSETRELAXTYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetCycleRelaxType \
        hypre_F90_NAME(fhypre_parcsrhybridsetcyclerela, FHYPRE_PARCSRHYBRIDSETCYCLERELA)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetcyclerela, FHYPRE_PARCSRHYBRIDSETCYCLERELA)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetRelaxOrder \
        hypre_F90_NAME(fhypre_parcsrhybridsetrelaxorde, FHYPRE_PARCSRHYBRIDSETRELAXORDE)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetrelaxorde, FHYPRE_PARCSRHYBRIDSETRELAXORDE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetRelaxWt \
        hypre_F90_NAME(fhypre_parcsrhybridsetrelaxwt, FHYPRE_PARCSRHYBRIDSETRELAXWT)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetrelaxwt, FHYPRE_PARCSRHYBRIDSETRELAXWT)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetLevelRelaxWt \
        hypre_F90_NAME(fhypre_parcsrhybridsetlevelrela, FHYPRE_PARCSRHYBRIDSETLEVELRELA)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetlevelrela, FHYPRE_PARCSRHYBRIDSETLEVELRELA)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetOuterWt \
        hypre_F90_NAME(fhypre_parcsrhybridsetouterwt, FHYPRE_PARCSRHYBRIDSETOUTERWT)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetouterwt, FHYPRE_PARCSRHYBRIDSETOUTERWT)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetLevelOuterWt \
        hypre_F90_NAME(fhypre_parcsrhybridsetleveloute, FHYPRE_PARCSRHYBRIDSETLEVELOUTE)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetleveloute, FHYPRE_PARCSRHYBRIDSETLEVELOUTE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetRelaxWeight \
        hypre_F90_NAME(fhypre_parcsrhybridsetrelaxweig, FHYPRE_PARCSRHYBRIDSETRELAXWEIG)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetrelaxweig, FHYPRE_PARCSRHYBRIDSETRELAXWEIG)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridSetOmega \
        hypre_F90_NAME(fhypre_parcsrhybridsetomega, FHYPRE_PARCSRHYBRIDSETOMEGA)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetomega, FHYPRE_PARCSRHYBRIDSETOMEGA)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridGetNumIterations \
        hypre_F90_NAME(fhypre_parcsrhybridgetnumiterat, FHYPRE_PARCSRHYBRIDGETNUMITERAT)
extern void hypre_F90_NAME(fhypre_parcsrhybridgetnumiterat, FHYPRE_PARCSRHYBRIDGETNUMITERAT)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridGetDSCGNumIterations \
        hypre_F90_NAME(fhypre_parcsrhybridgetdscgnumit, FHYPRE_PARCSRHYBRIDGETDSCGNUMIT)
extern void hypre_F90_NAME(fhypre_parcsrhybridgetdscgnumit, FHYPRE_PARCSRHYBRIDGETDSCGNUMIT)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridGetPCGNumIterations \
        hypre_F90_NAME(fhypre_parcsrhybridgetpcgnumite, FHYPRE_PARCSRHYBRIDGETPCGNUMITE)
extern void hypre_F90_NAME(fhypre_parcsrhybridgetpcgnumite, FHYPRE_PARCSRHYBRIDGETPCGNUMITE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRHybridGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_parcsrhybridgetfinalrela, FHYPRE_PARCSRHYBRIDGETFINALRELA)
extern void hypre_F90_NAME(fhypre_parcsrhybridgetfinalrela, FHYPRE_PARCSRHYBRIDGETFINALRELA)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParSetRandomValues \
        hypre_F90_NAME(fhypre_parsetrandomvalues, FHYPRE_PARSETRANDOMVALUES)
extern void hypre_F90_NAME(fhypre_parsetrandomvalues, FHYPRE_PARSETRANDOMVALUES)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParPrintVector \
        hypre_F90_NAME(fhypre_parprintvector, FHYPRE_PARPRINTVECTOR)
extern void hypre_F90_NAME(fhypre_parprintvector, FHYPRE_PARPRINTVECTOR)
                      (hypre_F90_Obj *, char *);

#define HYPRE_ParReadVector \
        hypre_F90_NAME(fhypre_parreadvector, FHYPRE_PARREADVECTOR)
extern void hypre_F90_NAME(fhypre_parreadvector, FHYPRE_PARREADVECTOR)
                      (HYPRE_Int *, char *);

#define HYPRE_ParVectorSize \
        hypre_F90_NAME(fhypre_parvectorsize, FHYPRE_PARVECTORSIZE)
extern void hypre_F90_NAME(fhypre_parvectorsize, FHYPRE_PARVECTORSIZE)
                      (HYPRE_Int *);

#define HYPRE_ParCSRMultiVectorPrint \
        hypre_F90_NAME(fhypre_parcsrmultivectorprint, FHYPRE_PARCSRMULTIVECTORPRINT)
extern void hypre_F90_NAME(fhypre_parcsrmultivectorprint, FHYPRE_PARCSRMULTIVECTORPRINT)
                      (HYPRE_Int *, char *);

#define HYPRE_ParCSRMultiVectorRead \
        hypre_F90_NAME(fhypre_parcsrmultivectorread, FHYPRE_PARCSRMULTIVECTORREAD)
extern void hypre_F90_NAME(fhypre_parcsrmultivectorread, FHYPRE_PARCSRMULTIVECTORREAD)
                      (HYPRE_Int *, hypre_F90_Obj *, char *);

#define aux_maskCount \
        hypre_F90_NAME(fhypre_aux_maskcount, FHYPRE_AUX_MASKCOUNT)
extern void hypre_F90_NAME(fhypre_aux_maskcount, FHYPRE_AUX_MASKCOUNT)
                      (HYPRE_Int *, HYPRE_Int *);

#define aux_indexFromMask \
        hypre_F90_NAME(fhypre_auxindexfrommask, FHYPRE_AUXINDEXFROMMASK)
extern void hypre_F90_NAME(fhypre_auxindexfrommask, FHYPRE_AUXINDEXFROMMASK)
                      (HYPRE_Int *, HYPRE_Int *, HYPRE_Int *);

#define HYPRE_TempParCSRSetupInterpreter \
        hypre_F90_NAME(fhypre_tempparcsrsetupinterpret, FHYPRE_TEMPPARCSRSETUPINTERPRET)
extern void hypre_F90_NAME(fhypre_tempparcsrsetupinterpret, FHYPRE_TEMPPARCSRSETUPINTERPRET)
                      (hypre_F90_Obj *);

#define HYPRE_ParCSRSetupInterpreter \
        hypre_F90_NAME(fhypre_parcsrsetupinterpreter, FHYPRE_PARCSRSETUPINTERPRETER)
extern void hypre_F90_NAME(fhypre_parcsrsetupinterpreter, FHYPRE_PARCSRSETUPINTERPRETER)
                      (hypre_F90_Obj *);

#define HYPRE_ParCSRSetupMatvec \
        hypre_F90_NAME(fhypre_parcsrsetupmatvec, FHYPRE_PARCSRSETUPMATVEC)
extern void hypre_F90_NAME(fhypre_parcsrsetupmatvec, FHYPRE_PARCSRSETUPMATVEC)
                      (hypre_F90_Obj *);



#define HYPRE_ParaSailsCreate  \
        hypre_F90_NAME(fhypre_parasailscreate, FHYPRE_PARASAILSCREATE)
extern void hypre_F90_NAME(fhypre_parasailscreate, FHYPRE_PARASAILSCREATE)
                      (HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_ParaSailsDestroy  \
        hypre_F90_NAME(fhypre_parasailsdestroy, FHYPRE_PARASAILSDESTROY)
extern void hypre_F90_NAME(fhypre_parasailsdestroy, FHYPRE_PARASAILSDESTROY)
                      (hypre_F90_Obj *);

#define HYPRE_ParaSailsSetup  \
        hypre_F90_NAME(fhypre_parasailssetup, FHYPRE_PARASAILSSETUP)
extern void hypre_F90_NAME(fhypre_parasailssetup, FHYPRE_PARASAILSSETUP)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParaSailsSolve  \
        hypre_F90_NAME(fhypre_parasailssolve, FHYPRE_PARASAILSSOLVE)
extern void hypre_F90_NAME(fhypre_parasailssolve, FHYPRE_PARASAILSSOLVE)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParaSailsSetParams  \
        hypre_F90_NAME(fhypre_parasailssetparams, FHYPRE_PARASAILSSETPARAMS)
extern void hypre_F90_NAME(fhypre_parasailssetparams, FHYPRE_PARASAILSSETPARAMS)
                      (hypre_F90_Obj *, double *, HYPRE_Int *);

#define HYPRE_ParaSailsSetThresh  \
        hypre_F90_NAME(fhypre_parasailssetthresh, FHYPRE_PARASAILSSETTHRESH)
extern void hypre_F90_NAME(fhypre_parasailssetthresh, FHYPRE_PARASAILSSETTHRESH)
                      (hypre_F90_Obj *, double *);

#define HYPRE_ParaSailsGetThresh  \
        hypre_F90_NAME(fhypre_parasailsgetthresh, FHYPRE_PARASAILSGETTHRESH)
extern void hypre_F90_NAME(fhypre_parasailsgetthresh, FHYPRE_PARASAILSGETTHRESH)
                      (hypre_F90_Obj *, double *);

#define HYPRE_ParaSailsSetNlevels  \
        hypre_F90_NAME(fhypre_parasailssetnlevels, FHYPRE_PARASAILSSETNLEVELS)
extern void hypre_F90_NAME(fhypre_parasailssetnlevels, FHYPRE_PARASAILSSETNLEVELS)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParaSailsGetNlevels  \
        hypre_F90_NAME(fhypre_parasailsgetnlevels, FHYPRE_PARASAILSGETNLEVELS)
extern void hypre_F90_NAME(fhypre_parasailsgetnlevels, FHYPRE_PARASAILSGETNLEVELS)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParaSailsSetFilter  \
        hypre_F90_NAME(fhypre_parasailssetfilter, FHYPRE_PARASAILSSETFILTER)
extern void hypre_F90_NAME(fhypre_parasailssetfilter, FHYPRE_PARASAILSSETFILTER)
                      (hypre_F90_Obj *, double *);

#define HYPRE_ParaSailsGetFilter  \
        hypre_F90_NAME(fhypre_parasailsgetfilter, FHYPRE_PARASAILSGETFILTER)
extern void hypre_F90_NAME(fhypre_parasailsgetfilter, FHYPRE_PARASAILSGETFILTER)
                      (hypre_F90_Obj *, double *);

#define HYPRE_ParaSailsSetSym  \
        hypre_F90_NAME(fhypre_parasailssetsym, FHYPRE_PARASAILSSETSYM)
extern void hypre_F90_NAME(fhypre_parasailssetsym, FHYPRE_PARASAILSSETSYM)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParaSailsGetSym  \
        hypre_F90_NAME(fhypre_parasailsgetsym, FHYPRE_PARASAILSGETSYM)
extern void hypre_F90_NAME(fhypre_parasailsgetsym, FHYPRE_PARASAILSGETSYM)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParaSailsSetLoadbal  \
        hypre_F90_NAME(fhypre_parasailssetloadbal, FHYPRE_PARASAILSSETLOADBAL)
extern void hypre_F90_NAME(fhypre_parasailssetloadbal, FHYPRE_PARASAILSSETLOADBAL)
                      (hypre_F90_Obj *, double *);

#define HYPRE_ParaSailsGetLoadbal  \
        hypre_F90_NAME(fhypre_parasailsgetloadbal, FHYPRE_PARASAILSGETLOADBAL)
extern void hypre_F90_NAME(fhypre_parasailsgetloadbal, FHYPRE_PARASAILSGETLOADBAL)
                      (hypre_F90_Obj *, double *);

#define HYPRE_ParaSailsSetReuse  \
        hypre_F90_NAME(fhypre_parasailssetreuse, FHYPRE_PARASAILSSETREUSE)
extern void hypre_F90_NAME(fhypre_parasailssetreuse, FHYPRE_PARASAILSSETREUSE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParaSailsGetReuse  \
        hypre_F90_NAME(fhypre_parasailsgetreuse, FHYPRE_PARASAILSGETREUSE)
extern void hypre_F90_NAME(fhypre_parasailsgetreuse, FHYPRE_PARASAILSGETREUSE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParaSailsSetLogging  \
        hypre_F90_NAME(fhypre_parasailssetlogging, FHYPRE_PARASAILSSETLOGGING)
extern void hypre_F90_NAME(fhypre_parasailssetlogging, FHYPRE_PARASAILSSETLOGGING)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParaSailsGetLogging  \
        hypre_F90_NAME(fhypre_parasailsgetlogging, FHYPRE_PARASAILSGETLOGGING)
extern void hypre_F90_NAME(fhypre_parasailsgetlogging, FHYPRE_PARASAILSGETLOGGING)
                      (hypre_F90_Obj *, HYPRE_Int *);



#define HYPRE_ParCSRPCGCreate  \
        hypre_F90_NAME(fhypre_parcsrpcgcreate, FHYPRE_PARCSRPCGCREATE)
extern void hypre_F90_NAME(fhypre_parcsrpcgcreate, FHYPRE_PARCSRPCGCREATE)
                      (HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_ParCSRPCGDestroy  \
        hypre_F90_NAME(fhypre_parcsrpcgdestroy, FHYPRE_PARCSRPCGDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrpcgdestroy, FHYPRE_PARCSRPCGDESTROY)
                      (hypre_F90_Obj *);

#define HYPRE_ParCSRPCGSetup  \
        hypre_F90_NAME(fhypre_parcsrpcgsetup, FHYPRE_PARCSRPCGSETUP)
extern void hypre_F90_NAME(fhypre_parcsrpcgsetup, FHYPRE_PARCSRPCGSETUP)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRPCGSolve  \
        hypre_F90_NAME(fhypre_parcsrpcgsolve, FHYPRE_PARCSRPCGSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrpcgsolve, FHYPRE_PARCSRPCGSOLVE)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRPCGSetTol  \
        hypre_F90_NAME(fhypre_parcsrpcgsettol, FHYPRE_PARCSRPCGSETTOL)
extern void hypre_F90_NAME(fhypre_parcsrpcgsettol, FHYPRE_PARCSRPCGSETTOL)
                      (hypre_F90_Obj *, double *);

#define HYPRE_ParCSRPCGSetMaxIter  \
        hypre_F90_NAME(fhypre_parcsrpcgsetmaxiter, FHYPRE_PARCSRPCGSETMAXITER)
extern void hypre_F90_NAME(fhypre_parcsrpcgsetmaxiter, FHYPRE_PARCSRPCGSETMAXITER)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRPCGSetStopCrit  \
        hypre_F90_NAME(fhypre_parcsrpcgsetstopcrit, FHYPRE_PARCSRPCGSETSTOPCRIT)
extern void hypre_F90_NAME(fhypre_parcsrpcgsetstopcrit, FHYPRE_PARCSRPCGSETSTOPCRIT)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRPCGSetTwoNorm  \
        hypre_F90_NAME(fhypre_parcsrpcgsettwonorm, FHYPRE_PARCSRPCGSETTWONORM)
extern void hypre_F90_NAME(fhypre_parcsrpcgsettwonorm, FHYPRE_PARCSRPCGSETTWONORM)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRPCGSetRelChange  \
        hypre_F90_NAME(fhypre_parcsrpcgsetrelchange, FHYPRE_PARCSRPCGSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_parcsrpcgsetrelchange, FHYPRE_PARCSRPCGSETRELCHANGE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRPCGSetPrecond  \
        hypre_F90_NAME(fhypre_parcsrpcgsetprecond, FHYPRE_PARCSRPCGSETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrpcgsetprecond, FHYPRE_PARCSRPCGSETPRECOND)
                      (hypre_F90_Obj *, HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_ParCSRPCGGetPrecond  \
        hypre_F90_NAME(fhypre_parcsrpcggetprecond, FHYPRE_PARCSRPCGGETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrpcggetprecond, FHYPRE_PARCSRPCGGETPRECOND)
                      (hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRPCGSetPrintLevel  \
        hypre_F90_NAME(fhypre_parcsrpcgsetprintlevel, FHYPRE_PARCSRPCGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_parcsrpcgsetprintlevel, FHYPRE_PARCSRPCGSETPRINTLEVEL)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRPCGGetNumIterations  \
        hypre_F90_NAME(fhypre_parcsrpcggetnumiteration, FHYPRE_PARCSRPCGGETNUMITERATION)
extern void hypre_F90_NAME(fhypre_parcsrpcggetnumiteration, FHYPRE_PARCSRPCGGETNUMITERATION)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRPCGGetFinalRelativeResidualNorm  \
        hypre_F90_NAME(fhypre_parcsrpcggetfinalrelativ, FHYPRE_PARCSRPCGGETFINALRELATIV)
extern void hypre_F90_NAME(fhypre_parcsrpcggetfinalrelativ, FHYPRE_PARCSRPCGGETFINALRELATIV)
                      (hypre_F90_Obj *, double *);



#define HYPRE_ParCSRDiagScaleSetup  \
        hypre_F90_NAME(fhypre_parcsrdiagscalesetup, FHYPRE_PARCSRDIAGSCALESETUP)
extern void hypre_F90_NAME(fhypre_parcsrdiagscalesetup, FHYPRE_PARCSRDIAGSCALESETUP)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRDiagScale  \
        hypre_F90_NAME(fhypre_parcsrdiagscale, FHYPRE_PARCSRDIAGSCALE)
extern void hypre_F90_NAME(fhypre_parcsrdiagscale, FHYPRE_PARCSRDIAGSCALE)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);



#define HYPRE_ParCSRPilutCreate  \
        hypre_F90_NAME(fhypre_parcsrpilutcreate, FHYPRE_PARCSRPILUTCREATE)
extern void hypre_F90_NAME(fhypre_parcsrpilutcreate, FHYPRE_PARCSRPILUTCREATE)
                      (HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_ParCSRPilutDestroy  \
        hypre_F90_NAME(fhypre_parcsrpilutdestroy, FHYPRE_PARCSRPILUTDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrpilutdestroy, FHYPRE_PARCSRPILUTDESTROY)
                      (hypre_F90_Obj *);

#define HYPRE_ParCSRPilutSetup  \
        hypre_F90_NAME(fhypre_parcsrpilutsetup, FHYPRE_PARCSRPILUTSETUP)
extern void hypre_F90_NAME(fhypre_parcsrpilutsetup, FHYPRE_PARCSRPILUTSETUP)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRPilutSolve  \
        hypre_F90_NAME(fhypre_parcsrpilutsolve, FHYPRE_PARCSRPILUTSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrpilutsolve, FHYPRE_PARCSRPILUTSOLVE)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_ParCSRPilutSetMaxIter  \
        hypre_F90_NAME(fhypre_parcsrpilutsetmaxiter, FHYPRE_PARCSRPILUTSETMAXITER)
extern void hypre_F90_NAME(fhypre_parcsrpilutsetmaxiter, FHYPRE_PARCSRPILUTSETMAXITER)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_ParCSRPilutSetDropToleran  \
        hypre_F90_NAME(fhypre_parcsrpilutsetdroptolera, FHYPRE_PARCSRPILUTSETDROPTOLERA)
extern void hypre_F90_NAME(fhypre_parcsrpilutsetdroptolera, FHYPRE_PARCSRPILUTSETDROPTOLERA)
                      (hypre_F90_Obj *, double *);

#define HYPRE_ParCSRPilutSetFacRowSize  \
        hypre_F90_NAME(fhypre_parcsrpilutsetfacrowsize, FHYPRE_PARCSRPILUTSETFACROWSIZE)
extern void hypre_F90_NAME(fhypre_parcsrpilutsetfacrowsize, FHYPRE_PARCSRPILUTSETFACROWSIZE)
                      (hypre_F90_Obj *, HYPRE_Int *);



#define HYPRE_SchwarzCreate \
        hypre_F90_NAME(fhypre_schwarzcreate, FHYPRE_SCHWARZCREATE)
extern void hypre_F90_NAME(fhypre_schwarzcreate, FHYPRE_SCHWARZCREATE)
                      (hypre_F90_Obj *);

#define HYPRE_SchwarzDestroy \
        hypre_F90_NAME(fhypre_schwarzdestroy, FHYPRE_SCHWARZDESTROY)
extern void hypre_F90_NAME(fhypre_schwarzdestroy, FHYPRE_SCHWARZDESTROY)
                      (hypre_F90_Obj *);

#define HYPRE_SchwarzSetup \
        hypre_F90_NAME(fhypre_schwarzsetup, FHYPRE_SCHWARZSETUP)
extern void hypre_F90_NAME(fhypre_schwarzsetup, FHYPRE_SCHWARZSETUP)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj*);

#define HYPRE_SchwarzSolve \
        hypre_F90_NAME(fhypre_schwarzsolve, FHYPRE_SCHWARZSOLVE)
extern void hypre_F90_NAME(fhypre_schwarzsolve, FHYPRE_SCHWARZSOLVE)
                      (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj*);

#define HYPRE_SchwarzSetVariant \
        hypre_F90_NAME(fhypre_schwarzsetvariant, FHYPRE_SCHWARZVARIANT)
extern void hypre_F90_NAME(fhypre_schwarzsetvariant, FHYPRE_SCHWARZVARIANT)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_SchwarzSetOverlap \
        hypre_F90_NAME(fhypre_schwarzsetoverlap, FHYPRE_SCHWARZOVERLAP)
extern void hypre_F90_NAME(fhypre_schwarzsetoverlap, FHYPRE_SCHWARZOVERLAP)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_SchwarzSetDomainType \
        hypre_F90_NAME(fhypre_schwarzsetdomaintype, FHYPRE_SVHWARZSETDOMAINTYPE)
extern void hypre_F90_NAME(fhypre_schwarzsetdomaintype, FHYPRE_SVHWARZSETDOMAINTYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_SchwarzSetDomainStructure \
        hypre_F90_NAME(fhypre_schwarzsetdomainstructur, FHYPRE_SCHWARZSETDOMAINSTRUCTUR)
extern void hypre_F90_NAME(fhypre_schwarzsetdomainstructur, FHYPRE_SCHWARZSETDOMAINSTRUCTUR)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_SchwarzSetNumFunctions \
        hypre_F90_NAME(fhypre_schwarzsetnumfunctions, FHYPRE_SCHWARZSETNUMFUNCTIONS)
extern void hypre_F90_NAME(fhypre_schwarzsetnumfunctions, FHYPRE_SCHWARZSETNUMFUNCTIONS)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_SchwarzSetRelaxWeight \
        hypre_F90_NAME(fhypre_schwarzsetrelaxweight, FHYPRE_SCHWARZSETRELAXWEIGHT)
extern void hypre_F90_NAME(fhypre_schwarzsetrelaxweight, FHYPRE_SCHWARZSETRELAXWEIGHT)
                      (hypre_F90_Obj *, double *);

#define HYPRE_SchwarzSetDofFunc \
        hypre_F90_NAME(fhypre_schwarzsetdoffunc, FHYPRE_SCHWARZSETDOFFUNC)
extern void hypre_F90_NAME(fhypre_schwarzsetdoffunc, FHYPRE_SCHWARZSETDOFFUNC)
                      (hypre_F90_Obj *, HYPRE_Int *);
