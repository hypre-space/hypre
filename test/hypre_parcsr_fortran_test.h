/******************************************************************************
 * Definitions of ParCSR Fortran interface routines
 *****************************************************************************/

#define HYPRE_ParCSRMatrixCreate  \
        hypre_F90_NAME(fhypre_parcsrmatrixcreate, FHYPRE_PARCSRMATRIXCREATE)
extern void hypre_F90_NAME(fhypre_parcsrmatrixcreate, FHYPRE_PARCSRMATRIXCREATE)
            (int *, int *, int *, int *, int *, int *, int *, int *, long int *);

#define HYPRE_ParCSRMatrixDestroy  \
        hypre_F90_NAME(fhypre_parcsrmatrixdestroy, FHYPRE_PARCSRMATRIXDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrmatrixdestroy, FHYPRE_PARCSRMATRIXDESTROY)
                      (long int *, int *);

#define HYPRE_ParCSRMatrixInitialize  \
        hypre_F90_NAME(fhypre_parcsrmatrixinitialize, FHYPRE_PARCSRMATRIXINITIALIZE)
extern void hypre_F90_NAME(fhypre_parcsrmatrixinitialize, FHYPRE_PARCSRMATRIXINITIALIZE)
                      (long int *);

#define HYPRE_ParCSRMatrixRead  \
        hypre_F90_NAME(fhypre_parcsrmatrixread, FHYPRE_PARCSRMATRIXREAD)
extern void hypre_F90_NAME(fhypre_parcsrmatrixread, FHYPRE_PARCSRMATRIXREAD)
                      (int *, char *, long int *);

#define HYPRE_ParCSRMatrixPrint  \
        hypre_F90_NAME(fhypre_parcsrmatrixprint, FHYPRE_PARCSRMATRIXPRINT)
extern void hypre_F90_NAME(fhypre_parcsrmatrixprint, FHYPRE_PARCSRMATRIXPRINT)
                      (long int *, char *, int *);

#define HYPRE_ParCSRMatrixGetComm  \
        hypre_F90_NAME(fhypre_parcsrmatrixgetcomm, FHYPRE_PARCSRMATRIXGETCOMM)
extern void hypre_F90_NAME(fhypre_parcsrmatrixgetcomm, FHYPRE_PARCSRMATRIXGETCOMM)
                      (long int *, int *);

#define HYPRE_ParCSRMatrixGetDims  \
        hypre_F90_NAME(fhypre_parcsrmatrixgetdims, FHYPRE_PARCSRMATRIXGETDIMS)
extern void hypre_F90_NAME(fhypre_parcsrmatrixgetdims, FHYPRE_PARCSRMATRIXGETDIMS)
                      (long int *, int *, int *);

#define HYPRE_ParCSRMatrixGetLocalRange  \
        hypre_F90_NAME(fhypre_parcsrmatrixgetlocalrange, FHYPRE_PARCSRMATRIXGETLOCALRANGE)
extern void hypre_F90_NAME(fhypre_parcsrmatrixgetlocalrange, FHYPRE_PARCSRMATRIXGETLOCALRANGE)
                      (long int *, int *, int *, int *, int *);

#define HYPRE_ParCSRMatrixGetRow  \
        hypre_F90_NAME(fhypre_parcsrmatrixgetrow, FHYPRE_PARCSRMATRIXGETROW)
extern void hypre_F90_NAME(fhypre_parcsrmatrixgetrow, FHYPRE_PARCSRMATRIXGETROW)
                      (long int *, int *, int *, long int *, long int *);

#define HYPRE_ParCSRMatrixRestoreRow  \
        hypre_F90_NAME(fhypre_parcsrmatrixrestorerow, FHYPRE_PARCSRMATRIXRESTOREROW)
extern void hypre_F90_NAME(fhypre_parcsrmatrixrestorerow, FHYPRE_PARCSRMATRIXRESTOREROW)
                      (long int *, int *, int *, long int *, long int *);

#define HYPRE_ParCSRMatrixMatvec  \
        hypre_F90_NAME(fhypre_parcsrmatrixmatvec, FHYPRE_PARCSRMATRIXMATVEC)
extern void hypre_F90_NAME(fhypre_parcsrmatrixmatvec, FHYPRE_PARCSRMATRIXMATVEC)
                      (double *, long int *, long int *, double *, long int *);  

#define HYPRE_ParCSRMatrixMatvecT  \
        hypre_F90_NAME(fhypre_parcsrmatrixmatvect, FHYPRE_PARCSRMATRIXMATVECT)
extern void hypre_F90_NAME(fhypre_parcsrmatrixmatvect, FHYPRE_PARCSRMATRIXMATVECT)
                      (double *, long int *, long int *, double *, long int *);



#define HYPRE_ParVectorCreate  \
        hypre_F90_NAME(fhypre_parvectorcreate, FHYPRE_PARVECTORCREATE)
extern void hypre_F90_NAME(fhypre_parvectorcreate, FHYPRE_PARVECTORCREATE)
                      (int *, int *, long int *, long int *);

#define HYPRE_ParVectorDestroy  \
        hypre_F90_NAME(fhypre_parvectordestroy, FHYPRE_PARVECTORDESTROY)
extern void hypre_F90_NAME(fhypre_parvectordestroy, FHYPRE_PARVECTORDESTROY)
                      (long int *);

#define HYPRE_ParVectorInitialize  \
        hypre_F90_NAME(fhypre_parvectorinitialize, FHYPRE_PARVECTORINITIALIZE)
extern void hypre_F90_NAME(fhypre_parvectorinitialize, FHYPRE_PARVECTORINITIALIZE)
                      (long int *);

#define HYPRE_ParVectorRead  \
        hypre_F90_NAME(fhypre_parvectorread, FHYPRE_PARVECTORREAD)
extern void hypre_F90_NAME(fhypre_parvectorread, FHYPRE_PARVECTORREAD)
                      (int *, long int *, char *);

#define HYPRE_ParVectorPrint  \
        hypre_F90_NAME(fhypre_parvectorprint, FHYPRE_PARVECTORPRINT)
extern void hypre_F90_NAME(fhypre_parvectorprint, FHYPRE_PARVECTORPRINT)
                      (long int *, char *, int *);



#define GenerateLaplacian  \
        hypre_F90_NAME(fgeneratelaplacian, FHYPRE_GENERATELAPLACIAN)
extern void hypre_F90_NAME(fgeneratelaplacian, FHYPRE_GENERATELAPLACIAN)
                      (int *, int *, int *, int *, int *, int *, int *,
                       int *, int *, int *, double *, long int *);



#define HYPRE_BoomerAMGCreate  \
        hypre_F90_NAME(fhypre_boomeramgcreate, FHYPRE_BOOMERAMGCREATE)
extern void hypre_F90_NAME(fhypre_boomeramgcreate, FHYPRE_BOOMERAMGCREATE)
                      (long int *);

#define HYPRE_BoomerAMGDestroy  \
        hypre_F90_NAME(fhypre_boomeramgdestroy, FHYPRE_BOOMERAMGDESTROY)
extern void hypre_F90_NAME(fhypre_boomeramgdestroy, FHYPRE_BOOMERAMGDESTROY)
                      (long int *);

#define HYPRE_BoomerAMGSetup  \
        hypre_F90_NAME(fhypre_boomeramgsetup, FHYPRE_BOOMERAMGSETUP)
extern void hypre_F90_NAME(fhypre_boomeramgsetup, FHYPRE_BOOMERAMGSETUP)
                      (long int *, long int *, long int *, long int *);

#define HYPRE_BoomerAMGSolve  \
        hypre_F90_NAME(fhypre_boomeramgsolve, FHYPRE_BOOMERAMGSOLVE)
extern void hypre_F90_NAME(fhypre_boomeramgsolve, FHYPRE_BOOMERAMGSOLVE)
                      (long int *, long int *, long int *, long int *);

#define HYPRE_BoomerAMGSolveT  \
        hypre_F90_NAME(fhypre_boomeramgsolvet, FHYPRE_BOOMERAMGSOLVET)
extern void hypre_F90_NAME(fhypre_boomeramgsolvet, FHYPRE_BOOMERAMGSOLVET)
                      (long int *, long int *, long int *, long int *);

#define HYPRE_BoomerAMGSetRestriction  \
        hypre_F90_NAME(fhypre_boomeramgsetrestriction, FHYPRE_BOOMERAMGSETRESTRICTION)
extern void hypre_F90_NAME(fhypre_boomeramgsetrestriction, FHYPRE_BOOMERAMGSETRESTRICTION)
                      (long int *, int *);

#define HYPRE_BoomerAMGSetMaxLevels  \
        hypre_F90_NAME(fhypre_boomeramgsetmaxlevels, FHYPRE_BOOMERAMGSETMAXLEVELS)
extern void hypre_F90_NAME(fhypre_boomeramgsetmaxlevels, FHYPRE_BOOMERAMGSETMAXLEVELS)
                      (long int *, int *);

#define HYPRE_BoomerAMGSetStrongThreshold  \
        hypre_F90_NAME(fhypre_boomeramgsetstrongthrshl, FHYPRE_BOOMERAMGSETSTRONGTHRSHL)
extern void hypre_F90_NAME(fhypre_boomeramgsetstrongthrshl, FHYPRE_BOOMERAMGSETSTRONGTHRSHL)
                      (long int *, double *);

#define HYPRE_BoomerAMGSetMaxRowSum  \
        hypre_F90_NAME(fhypre_boomeramgsetmaxrowsum, FHYPRE_BOOMERAMGSETMAXROWSUM)
extern void hypre_F90_NAME(fhypre_boomeramgsetmaxrowsum, FHYPRE_BOOMERAMGSETMAXROWSUM)
                      (long int *, double *);

#define HYPRE_BoomerAMGSetTruncFactor  \
        hypre_F90_NAME(fhypre_boomeramgsettruncfactor, FHYPRE_BOOMERAMGSETTRUNCFACTOR)
extern void hypre_F90_NAME(fhypre_boomeramgsettruncfactor, FHYPRE_BOOMERAMGSETTRUNCFACTOR)
                      (long int *, double *);

#define HYPRE_BoomerAMGSetInterpType  \
        hypre_F90_NAME(fhypre_boomeramgsetinterptype, FHYPRE_BOOMERAMGSETINTERPTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetinterptype, FHYPRE_BOOMERAMGSETINTERPTYPE)
                      (long int *, int *);

#define HYPRE_BoomerAMGSetMinIter  \
        hypre_F90_NAME(fhypre_boomeramgsetminiter, FHYPRE_BOOMERAMGSETMINITER)
extern void hypre_F90_NAME(fhypre_boomeramgsetminiter, FHYPRE_BOOMERAMGSETMINITER)
                      (long int *, int *);

#define HYPRE_BoomerAMGSetMaxIter  \
        hypre_F90_NAME(fhypre_boomeramgsetmaxiter, FHYPRE_BOOMERAMGSETMAXITER)
extern void hypre_F90_NAME(fhypre_boomeramgsetmaxiter, FHYPRE_BOOMERAMGSETMAXITER)
                      (long int *, int *);

#define HYPRE_BoomerAMGSetCoarsenType  \
        hypre_F90_NAME(fhypre_boomeramgsetcoarsentype, FHYPRE_BOOMERAMGSETCOARSENTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetcoarsentype, FHYPRE_BOOMERAMGSETCOARSENTYPE)
                      (long int *, int *);

#define HYPRE_BoomerAMGSetMeasureType  \
        hypre_F90_NAME(fhypre_boomeramgsetmeasuretype, FHYPRE_BOOMERAMGSETMEASURETYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetmeasuretype, FHYPRE_BOOMERAMGSETMEASURETYPE)
                      (long int *, int *);

#define HYPRE_BoomerAMGSetSetupType  \
        hypre_F90_NAME(fhypre_boomeramgsetsetuptype, FHYPRE_BOOMERAMGSETSETUPTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetsetuptype, FHYPRE_BOOMERAMGSETSETUPTYPE)
                      (long int *, int *);

#define HYPRE_BoomerAMGSetCycleType  \
        hypre_F90_NAME(fhypre_boomeramgsetcycletype, FHYPRE_BOOMERAMGSETCYCLETYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetcycletype, FHYPRE_BOOMERAMGSETCYCLETYPE)
                      (long int *, int *);

#define HYPRE_BoomerAMGSetTol  \
        hypre_F90_NAME(fhypre_boomeramgsettol, FHYPRE_BOOMERAMGSETTOL)
extern void hypre_F90_NAME(fhypre_boomeramgsettol, FHYPRE_BOOMERAMGSETTOL)
                      (long int *, double *);

#define HYPRE_BoomerAMGSetNumGridSweeps  \
        hypre_F90_NAME(fhypre_boomeramgsetnumgridsweeps, FHYPRE_BOOMERAMGSETNUMGRIDSWEEPS)
extern void hypre_F90_NAME(fhypre_boomeramgsetnumgridsweeps, FHYPRE_BOOMERAMGSETNUMGRIDSWEEPS)
                      (long int *, long int *);

#define HYPRE_BoomerAMGSetNumSweeps  \
        hypre_F90_NAME(fhypre_boomeramgsetnumsweeps, FHYPRE_BOOMERAMGSETNUMSWEEPS)
extern void hypre_F90_NAME(fhypre_boomeramgsetnumsweeps, FHYPRE_BOOMERAMGSETNUMSWEEPS)
                      (long int *, int *);

#define HYPRE_BoomerAMGSetCycleNumSweeps  \
        hypre_F90_NAME(fhypre_boomeramgsetcyclenumsweeps, FHYPRE_BOOMERAMGSETCYCLENUMSWEEPS)
extern void hypre_F90_NAME(fhypre_boomeramgsetcyclenumsweeps, FHYPRE_BOOMERAMGSETCYCLENUMSWEEPS)
                      (long int *, int *, int *);

#define HYPRE_BoomerAMGInitGridRelaxation  \
        hypre_F90_NAME(fhypre_boomeramginitgridrelaxatn, FHYPRE_BOOMERAMGINITGRIDRELAXATN)
extern void hypre_F90_NAME(fhypre_boomeramginitgridrelaxatn, FHYPRE_BOOMERAMGINITGRIDRELAXATN)
                      (long int *, long int *, long int *, int *, long int *, int *);

#define HYPRE_BoomerAMGFinalizeGridRelaxation  \
        hypre_F90_NAME(fhypre_boomeramgfingridrelaxatn, FHYPRE_BOOMERAMGFINGRIDRELAXATN)
extern void hypre_F90_NAME(fhypre_boomeramgfingridrelaxatn, FHYPRE_BOOMERAMGFINGRIDRELAXATN)
                      (long int *, long int *, long int *, long int *);

#define HYPRE_BoomerAMGSetRelaxType  \
        hypre_F90_NAME(fhypre_boomeramgsetrelaxtype, FHYPRE_BOOMERAMGSETRELAXTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetrelaxtype, FHYPRE_BOOMERAMGSETRELAXTYPE)
                      (long int *, int *);

#define HYPRE_BoomerAMGSetCycleRelaxType  \
        hypre_F90_NAME(fhypre_boomeramgsetcyclerelaxtype, FHYPRE_BOOMERAMGSETCYCLERELAXTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetcyclerelaxtype, FHYPRE_BOOMERAMGSETCYCLERELAXTYPE)
                      (long int *, int *, int *);

#define HYPRE_BoomerAMGSetRelaxWeight  \
        hypre_F90_NAME(fhypre_boomeramgsetrelaxweight, FHYPRE_BOOMERAMGSETRELAXWEIGHT)
extern void hypre_F90_NAME(fhypre_boomeramgsetrelaxweight, FHYPRE_BOOMERAMGSETRELAXWEIGHT)
                      (long int *, long int *);

#define HYPRE_BoomerAMGSetRelaxWt  \
        hypre_F90_NAME(fhypre_boomeramgsetrelaxwt, FHYPRE_BOOMERAMGSETRELAXWT)
extern void hypre_F90_NAME(fhypre_boomeramgsetrelaxwt, FHYPRE_BOOMERAMGSETRELAXWT)
                      (long int *, double *);

#define HYPRE_BoomerAMGSetLevelRelaxWt  \
        hypre_F90_NAME(fhypre_boomeramgsetlevelrelaxwt, FHYPRE_BOOMERAMGSETLEVELRELAXWT)
extern void hypre_F90_NAME(fhypre_boomeramgsetlevelrelaxwt, FHYPRE_BOOMERAMGSETLEVELRELAXWT)
                      (long int *, double *, int *);

#define HYPRE_BoomerAMGSetPrintLevel  \
        hypre_F90_NAME(fhypre_boomeramgsetprintlevel, FHYPRE_BOOMERAMGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_boomeramgsetprintlevel, FHYPRE_BOOMERAMGSETPRINTLEVEL)
                      (long int *, int *);

#define HYPRE_BoomerAMGSetPrintFileName  \
        hypre_F90_NAME(fhypre_boomeramgsetprintfilename, FHYPRE_BOOMERAMGSETPRINTFILENAME)
extern void hypre_F90_NAME(fhypre_boomeramgsetprintfilename, FHYPRE_BOOMERAMGSETPRINTFILENAME)
                      (long int *, char *);

#define HYPRE_BoomerAMGSetDebugFlag  \
        hypre_F90_NAME(fhypre_boomeramgsetdebugflag, FHYPRE_BOOMERAMGSETDEBUGFLAG)
extern void hypre_F90_NAME(fhypre_boomeramgsetdebugflag, FHYPRE_BOOMERAMGSETDEBUGFLAG)
                      (long int *, int *);

#define HYPRE_BoomerAMGGetNumIterations  \
        hypre_F90_NAME(fhypre_boomeramggetnumiterations, FHYPRE_BOOMERAMGGETNUMITERATIONS)
extern void hypre_F90_NAME(fhypre_boomeramggetnumiterations, FHYPRE_BOOMERAMGGETNUMITERATIONS)
                      (long int *, int *);

#define HYPRE_BoomerAMGGetFinalRelativeResidualNorm  \
        hypre_F90_NAME(fhypre_boomeramggetfinalreltvre, FHYPRE_BOOMERAMGGETFINALRELTVRE)
extern void hypre_F90_NAME(fhypre_boomeramggetfinalreltvre, FHYPRE_BOOMERAMGGETFINALRELTVRE)
                      (long int *, double *);

#define HYPRE_BoomerAMGSetGSMG  \
        hypre_F90_NAME(fhypre_boomeramgsetgsmg, FHYPRE_BOOMERAMGSETGSMG)
extern void hypre_F90_NAME(fhypre_boomeramgsetgsmg, FHYPRE_BOOMERAMGSETGSMG)
                      (long int *, int *);



#define HYPRE_ParCSRBiCGSTABCreate  \
        hypre_F90_NAME(fhypre_parcsrbicgstabcreate, FHYPRE_PARCSRBICGSTABCREATE)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabcreate, FHYPRE_PARCSRBICGSTABCREATE)
                      (int *, long int *);

#define HYPRE_ParCSRBiCGSTABDestroy  \
        hypre_F90_NAME(fhypre_parcsrbicgstabdestroy, FHYPRE_PARCSRBICGSTABDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabdestroy, FHYPRE_PARCSRBICGSTABDESTROY)
                      (long int *);

#define HYPRE_ParCSRBiCGSTABSetup  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetup, FHYPRE_PARCSRBICGSTABSETUP)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetup, FHYPRE_PARCSRBICGSTABSETUP)
                      (long int *, long int *, long int *, long int *);

#define HYPRE_ParCSRBiCGSTABSolve  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsolve, FHYPRE_PARCSRBICGSTABSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsolve, FHYPRE_PARCSRBICGSTABSOLVE)
                      (long int *, long int *, long int *, long int *);

#define HYPRE_ParCSRBiCGSTABSetTol  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsettol, FHYPRE_PARCSRBICGSTABSETTOL)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsettol, FHYPRE_PARCSRBICGSTABSETTOL)
                      (long int *, double *);

#define HYPRE_ParCSRBiCGSTABSetMinIter  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetminiter, FHYPRE_PARCSRBICGSTABSETMINITER)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetminiter, FHYPRE_PARCSRBICGSTABSETMINITER)
                      (long int *, int *);

#define HYPRE_ParCSRBiCGSTABSetMaxIter  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetmaxiter, FHYPRE_PARCSRBICGSTABSETMAXITER)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetmaxiter, FHYPRE_PARCSRBICGSTABSETMAXITER)
                      (long int *, int *);

#define HYPRE_ParCSRBiCGSTABSetPrecond  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetprecond, FHYPRE_PARCSRBICGSTABSETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetprecond, FHYPRE_PARCSRBICGSTABSETPRECOND)
                      (long int *, int *, long int *);

#define HYPRE_ParCSRBiCGSTABGetPrecond  \
        hypre_F90_NAME(fhypre_parcsrbicgstabgetprecond, FHYPRE_PARCSRBICGSTABGETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabgetprecond, FHYPRE_PARCSRBICGSTABGETPRECOND)
                      (long int *, long int *);

#define HYPRE_ParCSRBiCGSTABSetLogging  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetlogging, FHYPRE_PARCSRBICGSTABSETLOGGING)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetlogging, FHYPRE_PARCSRBICGSTABSETLOGGING)
                      (long int *, int *);

#define HYPRE_ParCSRBiCGSTABGetNumIter  \
        hypre_F90_NAME(fhypre_parcsrbicgstabgetnumiter, FHYPRE_PARCSRBICGSTABGETNUMITER)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabgetnumiter, FHYPRE_PARCSRBICGSTABGETNUMITER)
                      (long int *, int *);

#define HYPRE_ParCSRBiCGSTABGetFinalRel  \
        hypre_F90_NAME(fhypre_parcsrbicgstabgetfinalre, FHYPRE_PARCSRBICGSTABGETFINALRE)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabgetfinalre, FHYPRE_PARCSRBICGSTABGETFINALRE)
                      (long int *, double *);



#define HYPRE_ParCSRCGNRCreate  \
	hypre_F90_NAME(fhypre_parcsrcgnrcreate, FHYPRE_PARCSRCGNRCREATE)
extern void hypre_F90_NAME(fhypre_parcsrcgnrcreate, FHYPRE_PARCSRCGNRCREATE)
                      (int *, long int *);

#define HYPRE_ParCSRCGNRDestroy  \
        hypre_F90_NAME(fhypre_parcsrcgnrdestroy, FHYPRE_PARCSRCGNRDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrcgnrdestroy, FHYPRE_PARCSRCGNRDESTROY)
                      (long int *);

#define HYPRE_ParCSRCGNRSetup  \
        hypre_F90_NAME(fhypre_parcsrcgnrsetup, FHYPRE_PARCSRCGNRSETUP)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsetup, FHYPRE_PARCSRCGNRSETUP)
                      (long int *, long int *, long int *, long int *);

#define HYPRE_ParCSRCGNRSolve  \
        hypre_F90_NAME(fhypre_parcsrcgnrsolve, FHYPRE_PARCSRCGNRSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsolve, FHYPRE_PARCSRCGNRSOLVE)
                      (long int *, long int *, long int *, long int *);

#define HYPRE_ParCSRCGNRSetTol  \
        hypre_F90_NAME(fhypre_parcsrcgnrsettol, FHYPRE_PARCSRCGNRSETTOL)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsettol, FHYPRE_PARCSRCGNRSETTOL)
                      (long int *, double *);

#define HYPRE_ParCSRCGNRSetMaxIter  \
        hypre_F90_NAME(fhypre_parcsrcgnrsetmaxiter, FHYPRE_PARCSRCGNRSETMAXITER)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsetmaxiter, FHYPRE_PARCSRCGNRSETMAXITER)
                      (long int *, int *);

#define HYPRE_ParCSRCGNRSetPrecond  \
        hypre_F90_NAME(fhypre_parcsrcgnrsetprecond, FHYPRE_PARCSRCGNRSETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsetprecond, FHYPRE_PARCSRCGNRSETPRECOND)
                      (long int *, int *, long int *);

#define HYPRE_ParCSRCGNRGetPrecond  \
        hypre_F90_NAME(fhypre_parcsrcgnrgetprecond, FHYPRE_PARCSRCGNRGETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrcgnrgetprecond, FHYPRE_PARCSRCGNRGETPRECOND)
                      (long int *, long int *);

#define HYPRE_ParCSRCGNRSetLogging  \
        hypre_F90_NAME(fhypre_parcsrcgnrsetlogging, FHYPRE_PARCSRCGNRSETLOGGING)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsetlogging, FHYPRE_PARCSRCGNRSETLOGGING)
                      (long int *, int *);

#define HYPRE_ParCSRCGNRGetNumIteration  \
        hypre_F90_NAME(fhypre_parcsrcgnrgetnumiteratio, FHYPRE_PARCSRCGNRGETNUMITERATIO)
extern void hypre_F90_NAME(fhypre_parcsrcgnrgetnumiteratio, FHYPRE_PARCSRCGNRGETNUMITERATIO)
                      (long int *, int *);

#define HYPRE_ParCSRCGNRGetFinalRelativ  \
        hypre_F90_NAME(fhypre_parcsrcgnrgetfinalrelati, FHYPRE_PARCSRCGNRGETFINALRELATI)
extern void hypre_F90_NAME(fhypre_parcsrcgnrgetfinalrelati, FHYPRE_PARCSRCGNRGETFINALRELATI)
                      (long int *, double *);



#define HYPRE_ParCSRGMRESCreate  \
        hypre_F90_NAME(fhypre_parcsrgmrescreate, FHYPRE_PARCSRGMRESCREATE)
extern void hypre_F90_NAME(fhypre_parcsrgmrescreate, FHYPRE_PARCSRGMRESCREATE)
                      (int *, long int *);

#define HYPRE_ParCSRGMRESDestroy  \
        hypre_F90_NAME(fhypre_parcsrgmresdestroy, FHYPRE_PARCSRGMRESDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrgmresdestroy, FHYPRE_PARCSRGMRESDESTROY)
                      (long int *);

#define HYPRE_ParCSRGMRESSetup  \
        hypre_F90_NAME(fhypre_parcsrgmressetup, FHYPRE_PARCSRGMRESSETUP)
extern void hypre_F90_NAME(fhypre_parcsrgmressetup, FHYPRE_PARCSRGMRESSETUP)
                      (long int *, long int *, long int *, long int *);

#define HYPRE_ParCSRGMRESSolve  \
        hypre_F90_NAME(fhypre_parcsrgmressolve, FHYPRE_PARCSRGMRESSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrgmressolve, FHYPRE_PARCSRGMRESSOLVE)
                      (long int *, long int *, long int *, long int *);

#define HYPRE_ParCSRGMRESSetKDim  \
        hypre_F90_NAME(fhypre_parcsrgmressetkdim, FHYPRE_PARCSRGMRESSETKDIM)
extern void hypre_F90_NAME(fhypre_parcsrgmressetkdim, FHYPRE_PARCSRGMRESSETKDIM)
                      (long int *, int *);

#define HYPRE_ParCSRGMRESSetTol  \
        hypre_F90_NAME(fhypre_parcsrgmressettol, FHYPRE_PARCSRGMRESSETTOL)
extern void hypre_F90_NAME(fhypre_parcsrgmressettol, FHYPRE_PARCSRGMRESSETTOL)
                      (long int *, double *);

#define HYPRE_ParCSRGMRESSetMinIter  \
        hypre_F90_NAME(fhypre_parcsrgmressetminiter, FHYPRE_PARCSRGMRESSETMINITER)
extern void hypre_F90_NAME(fhypre_parcsrgmressetminiter, FHYPRE_PARCSRGMRESSETMINITER)
                      (long int *, int *);

#define HYPRE_ParCSRGMRESSetMaxIter  \
        hypre_F90_NAME(fhypre_parcsrgmressetmaxiter, FHYPRE_PARCSRGMRESSETMAXITER)
extern void hypre_F90_NAME(fhypre_parcsrgmressetmaxiter, FHYPRE_PARCSRGMRESSETMAXITER)
                      (long int *, int *);

#define HYPRE_ParCSRGMRESSetPrecond  \
        hypre_F90_NAME(fhypre_parcsrgmressetprecond, FHYPRE_PARCSRGMRESSETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrgmressetprecond, FHYPRE_PARCSRGMRESSETPRECOND)
                      (long int *, int *, long int *);

#define HYPRE_ParCSRGMRESGetPrecond  \
        hypre_F90_NAME(fhypre_parcsrgmresgetprecond, FHYPRE_PARCSRGMRESGETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrgmresgetprecond, FHYPRE_PARCSRGMRESGETPRECOND)
                      (long int *, long int *);

#define HYPRE_ParCSRGMRESSetLogging  \
        hypre_F90_NAME(fhypre_parcsrgmressetlogging, FHYPRE_PARCSRGMRESSETLOGGING)
extern void hypre_F90_NAME(fhypre_parcsrgmressetlogging, FHYPRE_PARCSRGMRESSETLOGGING)
                      (long int *, int *);

#define HYPRE_ParCSRGMRESGetNumIter  \
        hypre_F90_NAME(fhypre_parcsrgmresgetnumiterati, FHYPRE_PARCSRGMRESGETNUMITERATI)
extern void hypre_F90_NAME(fhypre_parcsrgmresgetnumiterati, FHYPRE_PARCSRGMRESGETNUMITERATI)
                      (long int *, int *);

#define HYPRE_ParCSRGMRESGetFinalRelati  \
        hypre_F90_NAME(fhypre_parcsrgmresgetfinalrelat, FHYPRE_PARCSRGMRESGETFINALRELAT)
extern void hypre_F90_NAME(fhypre_parcsrgmresgetfinalrelat, FHYPRE_PARCSRGMRESGETFINALRELAT)
                      (long int *, double *);



#define HYPRE_ParaSailsCreate  \
        hypre_F90_NAME(fhypre_parasailscreate, FHYPRE_PARASAILSCREATE)
extern void hypre_F90_NAME(fhypre_parasailscreate, FHYPRE_PARASAILSCREATE)
                      (int *, long int *);

#define HYPRE_ParaSailsDestroy  \
        hypre_F90_NAME(fhypre_parasailsdestroy, FHYPRE_PARASAILSDESTROY)
extern void hypre_F90_NAME(fhypre_parasailsdestroy, FHYPRE_PARASAILSDESTROY)
                      (long int *);

#define HYPRE_ParaSailsSetup  \
        hypre_F90_NAME(fhypre_parasailssetup, FHYPRE_PARASAILSSETUP)
extern void hypre_F90_NAME(fhypre_parasailssetup, FHYPRE_PARASAILSSETUP)
                      (long int *, long int *, long int *, long int *);

#define HYPRE_ParaSailsSolve  \
        hypre_F90_NAME(fhypre_parasailssolve, FHYPRE_PARASAILSSOLVE)
extern void hypre_F90_NAME(fhypre_parasailssolve, FHYPRE_PARASAILSSOLVE)
                      (long int *, long int *, long int *, long int *);

#define HYPRE_ParaSailsSetParams  \
        hypre_F90_NAME(fhypre_parasailssetparams, FHYPRE_PARASAILSSETPARAMS)
extern void hypre_F90_NAME(fhypre_parasailssetparams, FHYPRE_PARASAILSSETPARAMS)
                      (long int *, double *, int *);

#define HYPRE_ParaSailsSetFilter  \
        hypre_F90_NAME(fhypre_parasailssetfilter, FHYPRE_PARASAILSSETFILTER)
extern void hypre_F90_NAME(fhypre_parasailssetfilter, FHYPRE_PARASAILSSETFILTER)
                      (long int *, double *);

#define HYPRE_ParaSailsSetSym  \
        hypre_F90_NAME(fhypre_parasailssetsym, FHYPRE_PARASAILSSETSYM)
extern void hypre_F90_NAME(fhypre_parasailssetsym, FHYPRE_PARASAILSSETSYM)
                      (long int *, int *);

#define HYPRE_ParaSailsSetLogging  \
        hypre_F90_NAME(fhypre_parasailssetlogging, FHYPRE_PARASAILSSETLOGGING)
extern void hypre_F90_NAME(fhypre_parasailssetlogging, FHYPRE_PARASAILSSETLOGGING)
                      (long int *, int *);



#define HYPRE_ParCSRPCGCreate  \
        hypre_F90_NAME(fhypre_parcsrpcgcreate, FHYPRE_PARCSRPCGCREATE)
extern void hypre_F90_NAME(fhypre_parcsrpcgcreate, FHYPRE_PARCSRPCGCREATE)
                      (int *, long int *);

#define HYPRE_ParCSRPCGDestroy  \
        hypre_F90_NAME(fhypre_parcsrpcgdestroy, FHYPRE_PARCSRPCGDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrpcgdestroy, FHYPRE_PARCSRPCGDESTROY)
                      (long int *);

#define HYPRE_ParCSRPCGSetup  \
        hypre_F90_NAME(fhypre_parcsrpcgsetup, FHYPRE_PARCSRPCGSETUP)
extern void hypre_F90_NAME(fhypre_parcsrpcgsetup, FHYPRE_PARCSRPCGSETUP)
                      (long int *, long int *, long int *, long int *);

#define HYPRE_ParCSRPCGSolve  \
        hypre_F90_NAME(fhypre_parcsrpcgsolve, FHYPRE_PARCSRPCGSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrpcgsolve, FHYPRE_PARCSRPCGSOLVE)
                      (long int *, long int *, long int *, long int *);

#define HYPRE_ParCSRPCGSetTol  \
        hypre_F90_NAME(fhypre_parcsrpcgsettol, FHYPRE_PARCSRPCGSETTOL)
extern void hypre_F90_NAME(fhypre_parcsrpcgsettol, FHYPRE_PARCSRPCGSETTOL)
                      (long int *, double *);

#define HYPRE_ParCSRPCGSetMaxIter  \
        hypre_F90_NAME(fhypre_parcsrpcgsetmaxiter, FHYPRE_PARCSRPCGSETMAXITER)
extern void hypre_F90_NAME(fhypre_parcsrpcgsetmaxiter, FHYPRE_PARCSRPCGSETMAXITER)
                      (long int *, int *);

#define HYPRE_ParCSRPCGSetTwoNorm  \
        hypre_F90_NAME(fhypre_parcsrpcgsettwonorm, FHYPRE_PARCSRPCGSETTWONORM)
extern void hypre_F90_NAME(fhypre_parcsrpcgsettwonorm, FHYPRE_PARCSRPCGSETTWONORM)
                      (long int *, int *);

#define HYPRE_ParCSRPCGSetRelChange  \
        hypre_F90_NAME(fhypre_parcsrpcgsetrelchange, FHYPRE_PARCSRPCGSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_parcsrpcgsetrelchange, FHYPRE_PARCSRPCGSETRELCHANGE)
                      (long int *, int *);

#define HYPRE_ParCSRPCGSetPrecond  \
        hypre_F90_NAME(fhypre_parcsrpcgsetprecond, FHYPRE_PARCSRPCGSETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrpcgsetprecond, FHYPRE_PARCSRPCGSETPRECOND)
                      (long int *, int *, long int *);

#define HYPRE_ParCSRPCGGetPrecond  \
        hypre_F90_NAME(fhypre_parcsrpcggetprecond, FHYPRE_PARCSRPCGGETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrpcggetprecond, FHYPRE_PARCSRPCGGETPRECOND)
                      (long int *, long int *);

#define HYPRE_ParCSRPCGSetPrintLevel  \
        hypre_F90_NAME(fhypre_parcsrpcgsetprintlevel, FHYPRE_PARCSRPCGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_parcsrpcgsetprintlevel, FHYPRE_PARCSRPCGSETPRINTLEVEL)
                      (long int *, int *);

#define HYPRE_ParCSRPCGGetNumIterations  \
        hypre_F90_NAME(fhypre_parcsrpcggetnumiteration, FHYPRE_PARCSRPCGGETNUMITERATION)
extern void hypre_F90_NAME(fhypre_parcsrpcggetnumiteration, FHYPRE_PARCSRPCGGETNUMITERATION)
                      (long int *, int *);

#define HYPRE_ParCSRPCGGetFinalRelativeResidualNorm  \
        hypre_F90_NAME(fhypre_parcsrpcggetfinalrelativ, FHYPRE_PARCSRPCGGETFINALRELATIV)
extern void hypre_F90_NAME(fhypre_parcsrpcggetfinalrelativ, FHYPRE_PARCSRPCGGETFINALRELATIV)
                      (long int *, double *);



#define HYPRE_ParCSRDiagScaleSetup  \
        hypre_F90_NAME(fhypre_parcsrdiagscalesetup, FHYPRE_PARCSRDIAGSCALESETUP)
extern void hypre_F90_NAME(fhypre_parcsrdiagscalesetup, FHYPRE_PARCSRDIAGSCALESETUP)
                      (long int *, long int *, long int *, long int *);

#define HYPRE_ParCSRDiagScale  \
        hypre_F90_NAME(fhypre_parcsrdiagscale, FHYPRE_PARCSRDIAGSCALE)
extern void hypre_F90_NAME(fhypre_parcsrdiagscale, FHYPRE_PARCSRDIAGSCALE)
                      (long int *, long int *, long int *, long int *);



#define HYPRE_ParCSRPilutCreate  \
        hypre_F90_NAME(fhypre_parcsrpilutcreate, FHYPRE_PARCSRPILUTCREATE)
extern void hypre_F90_NAME(fhypre_parcsrpilutcreate, FHYPRE_PARCSRPILUTCREATE)
                      (int *, long int *);

#define HYPRE_ParCSRPilutDestroy  \
        hypre_F90_NAME(fhypre_parcsrpilutdestroy, FHYPRE_PARCSRPILUTDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrpilutdestroy, FHYPRE_PARCSRPILUTDESTROY)
                      (long int *);

#define HYPRE_ParCSRPilutSetup  \
        hypre_F90_NAME(fhypre_parcsrpilutsetup, FHYPRE_PARCSRPILUTSETUP)
extern void hypre_F90_NAME(fhypre_parcsrpilutsetup, FHYPRE_PARCSRPILUTSETUP)
                      (long int *, long int *, long int *, long int *);

#define HYPRE_ParCSRPilutSolve  \
        hypre_F90_NAME(fhypre_parcsrpilutsolve, FHYPRE_PARCSRPILUTSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrpilutsolve, FHYPRE_PARCSRPILUTSOLVE)
                      (long int *, long int *, long int *, long int *);

#define HYPRE_ParCSRPilutSetMaxIter  \
        hypre_F90_NAME(fhypre_parcsrpilutsetmaxiter, FHYPRE_PARCSRPILUTSETMAXITER)
extern void hypre_F90_NAME(fhypre_parcsrpilutsetmaxiter, FHYPRE_PARCSRPILUTSETMAXITER)
                      (long int *, int *);

#define HYPRE_ParCSRPilutSetDropToleran  \
        hypre_F90_NAME(fhypre_parcsrpilutsetdroptolera, FHYPRE_PARCSRPILUTSETDROPTOLERA)
extern void hypre_F90_NAME(fhypre_parcsrpilutsetdroptolera, FHYPRE_PARCSRPILUTSETDROPTOLERA)
                      (long int *, double *);

#define HYPRE_ParCSRPilutSetFacRowSize  \
        hypre_F90_NAME(fhypre_parcsrpilutsetfacrowsize, FHYPRE_PARCSRPILUTSETFACROWSIZE)
extern void hypre_F90_NAME(fhypre_parcsrpilutsetfacrowsize, FHYPRE_PARCSRPILUTSETFACROWSIZE)
                      (long int *, int *);
