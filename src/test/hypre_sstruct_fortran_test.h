/******************************************************************************
 *  Definitions of sstruct fortran interface routines
 *****************************************************************************/

#define HYPRE_SStructGraphCreate \
        hypre_F90_NAME(fhypre_sstructgraphcreate, FHYPRE_SSTRUCTGRAPHCREATE)
extern void hypre_F90_NAME(fhypre_sstructgraphcreate, FHYPRE_SSTRUCTGRAPHCREATE)
                          (int *, long int *, long int *);

#define HYPRE_SStructGraphDestroy \
        hypre_F90_NAME(fhypre_sstructgraphdestroy, FHYPRE_SSTRUCTGRAPHDESTROY)
extern void hypre_F90_NAME(fhypre_sstructgraphdestroy, FHYPRE_SSTRUCTGRAPHDESTROY)
                          (long int *);

#define HYPRE_SStructGraphSetStencil \
        hypre_F90_NAME(fhypre_sstructgraphsetstencil, FHYPRE_SSTRUCTGRAPHSETSTENCIL)
extern void hypre_F90_NAME(fhypre_sstructgraphsetstencil, FHYPRE_SSTRUCTGRAPHSETSTENCIL)
                          (long int *, int *, int *, long int *);

#define HYPRE_SStructGraphAddEntries \
        hypre_F90_NAME(fhypre_sstructgraphaddentries, FHYPRE_SSTRUCTGRAPHADDENTRIES)
extern void hypre_F90_NAME(fhypre_sstructgraphaddentries, FHYPRE_SSTRUCTGRAPHADDENTRIES)
                          (long int *, int *, int *, int *, int *, int *, int *);

#define HYPRE_SStructGraphAssemble \
        hypre_F90_NAME(fhypre_sstructgraphassemble, FHYPRE_SSTRUCTGRAPHASSEMBLE)
extern void hypre_F90_NAME(fhypre_sstructgraphassemble, FHYPRE_SSTRUCTGRAPHASSEMBLE)
                          (long int *);

#define HYPRE_SStructGraphSetObjectType \
        hypre_F90_NAME(fhypre_sstructgraphsetobjecttyp, FHYPRE_SSTRUCTGRAPHSETOBJECTTYP)

extern void hypre_F90_NAME(fhypre_sstructgraphsetobjecttyp, FHYPRE_SSTRUCTGRAPHSETOBJECTTYP)
                          (long int *, int *);



#define HYPRE_SStructGridCreate \
        hypre_F90_NAME(fhypre_sstructgridcreate, FHYPRE_SSTRUCTGRIDCREATE)
extern void hypre_F90_NAME(fhypre_sstructgridcreate, FHYPRE_SSTRUCTGRIDCREATE)
                          (int *, int *, int *, long int *);

#define HYPRE_SStructGridDestroy \
        hypre_F90_NAME(fhypre_sstructgriddestroy, FHYPRE_SSTRUCTGRIDDESTROY)
extern void hypre_F90_NAME(fhypre_sstructgriddestroy, FHYPRE_SSTRUCTGRIDDESTROY)
                          (long int *);

#define HYPRE_SStructGridSetExtents \
        hypre_F90_NAME(fhypre_sstructgridsetextents, FHYPRE_SSTRUCTGRIDSETEXTENTS)
extern void hypre_F90_NAME(fhypre_sstructgridsetextents, FHYPRE_SSTRUCTGRIDSETEXTENTS)
                          (long int *, int *, int *, int *);

#define HYPRE_SStructGridSetVariables \
        hypre_F90_NAME(fhypre_sstructgridsetvariables, FHYPRE_SSTRUCTGRIDSETVARIABLES)
extern void hypre_F90_NAME(fhypre_sstructgridsetvariables, FHYPRE_SSTRUCTGRIDSETVARIABLES)
                          (long int *, int *, int *, long int *);

#define HYPRE_SStructGridAddVariables \
        hypre_F90_NAME(fhypre_sstructgridaddvariables, FHYPRE_SSTRUCTGRIDADDVARIABLES)
extern void hypre_F90_NAME(fhypre_sstructgridaddvariables, FHYPRE_SSTRUCTGRIDADDVARIABLES)
                          (long int  *, int *, int *, int *, long int *);

#define HYPRE_SStructGridSetNeighborBox \
        hypre_F90_NAME(fhypre_sstructgridsetneighborbo, FHYPRE_SSTRUCTGRIDSETNEIGHBORBO)
extern void hypre_F90_NAME(fhypre_sstructgridsetneighborbo, FHYPRE_SSTRUCTGRIDSETNEIGHBORBO)
                          (long int *, int *, int *, int *, int *, int *, int *, int *);

#define HYPRE_SStructGridAssemble \
        hypre_F90_NAME(fhypre_sstructgridassemble, FHYPRE_SSTRUCTGRIDASSEMBLE)
extern void hypre_F90_NAME(fhypre_sstructgridassemble, FHYPRE_SSTRUCTGRIDASSEMBLE)
                          (long int *);

#define HYPRE_SStructGridSetPeriodic \
        hypre_F90_NAME(fhypre_sstructgridsetperiodic, FHYPRE_SSTRUCTGRIDSETPERIODIC)
extern void hypre_F90_NAME(fhypre_sstructgridsetperiodic, FHYPRE_SSTRUCTGRIDSETPERIODIC)
                          (long int *, int *, int *);

#define HYPRE_SStructGridSetNumGhost \
        hypre_F90_NAME(fhypre_sstructgridsetnumghost, FHYPRE_SSTRUCTGRIDSETNUMGHOST)
extern void hypre_F90_NAME(fhypre_sstructgridsetnumghost, FHYPRE_SSTRUCTGRIDSETNUMGHOST)
                          (long int *, int *);



#define HYPRE_SStructMatrixCreate \
        hypre_F90_NAME(fhypre_sstructmatrixcreate, FHYPRE_SSTRUCTMATRIXCREATE)
extern void hypre_F90_NAME(fhypre_sstructmatrixcreate, FHYPRE_SSTRUCTMATRIXCREATE)
                          (int *, long int *, long int *);

#define HYPRE_SStructMatrixDestroy \
        hypre_F90_NAME(fhypre_sstructmatrixdestroy, FHYPRE_SSTRUCTMATRIXDESTROY)
extern void hypre_F90_NAME(fhypre_sstructmatrixdestroy, FHYPRE_SSTRUCTMATRIXDESTROY)
                          (long int *);

#define HYPRE_SStructMatrixInitialize \
        hypre_F90_NAME(fhypre_sstructmatrixinitialize, FHYPRE_SSTRUCTMATRIXINITIALIZE)
extern void hypre_F90_NAME(fhypre_sstructmatrixinitialize, FHYPRE_SSTRUCTMATRIXINITIALIZE)
                          (long int *);

#define HYPRE_SStructMatrixSetValues \
        hypre_F90_NAME(fhypre_sstructmatrixsetvalues, FHYPRE_SSTRUCTMATRIXSETVALUES)
extern void hypre_F90_NAME(fhypre_sstructmatrixsetvalues, FHYPRE_SSTRUCTMATRIXSETVALUES)
                          (long int *, int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructMatrixSetBoxValues \
        hypre_F90_NAME(fhypre_sstructmatrixsetboxvalue, FHYPRE_SSTRUCTMATRIXSETBOXVALUE)
extern void hypre_F90_NAME(fhypre_sstructmatrixsetboxvalue, FHYPRE_SSTRUCTMATRIXSETBOXVALUE)
                          (long int *, int *, int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructMatrixGetValues \
        hypre_F90_NAME(fhypre_sstructmatrixgetvalues, FHYPRE_SSTRUCTMATRIXGETVALUES)
extern void hypre_F90_NAME(fhypre_sstructmatrixgetvalues, FHYPRE_SSTRUCTMATRIXGETVALUES)
                          (long int *, int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructMatrixGetBoxValues \
        hypre_F90_NAME(fhypre_sstructmatrixgetboxvalue, FHYPRE_SSTRUCTMATRIXGETBOXVALUE)
extern void hypre_F90_NAME(fhypre_sstructmatrixgetboxvalue, FHYPRE_SSTRUCTMATRIXGETBOXVALUE)
                          (long int *, int *, int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructMatrixAddToValues \
        hypre_F90_NAME(fhypre_sstructmatrixaddtovalues, FHYPRE_SSTRUCTMATRIXADDTOVALUES)
extern void hypre_F90_NAME(fhypre_sstructmatrixaddtovalues, FHYPRE_SSTRUCTMATRIXADDTOVALUES)
                          (long int *, int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructMatrixAddToBoxValues \
        hypre_F90_NAME(fhypre_sstructmatrixaddtoboxval, FHYPRE_SSTRUCTMATRIXADDTOBOXVAL)
extern void hypre_F90_NAME(fhypre_sstructmatrixaddtoboxval, FHYPRE_SSTRUCTMATRIXADDTOBOXVAL)
                          (long int *, int *, int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructMatrixAssemble \
        hypre_F90_NAME(fhypre_sstructmatrixassemble, FHYPRE_SSTRUCTMATRIXASSEMBLE)
extern void hypre_F90_NAME(fhypre_sstructmatrixassemble, FHYPRE_SSTRUCTMATRIXASSEMBLE)
                          (long int *);

#define HYPRE_SStructMatrixSetSymmetric \
        hypre_F90_NAME(fhypre_sstructmatrixsetsymmetri, FHYPRE_SSTRUCTMATRIXSETSYMMETRI)
extern void hypre_F90_NAME(fhypre_sstructmatrixsetsymmetri, FHYPRE_SSTRUCTMATRIXSETSYMMETRI)
                          (long int *, int *, int *, int *, int *);

#define HYPRE_SStructMatrixSetNSSymmetric \
        hypre_F90_NAME(fhypre_sstructmatrixsetnssymmet, FHYPRE_SSTRUCTMATRIXSETNSSYMMET)
extern void hypre_F90_NAME(fhypre_sstructmatrixsetnssymmet, FHYPRE_SSTRUCTMATRIXSETNSSYMMET)
                          (long int *, int *);

#define HYPRE_SStructMatrixSetObjectType \
        hypre_F90_NAME(fhypre_sstructmatrixsetobjectty, FHYPRE_SSTRUCTMATRIXSETOBJECTTY)
extern void hypre_F90_NAME(fhypre_sstructmatrixsetobjectty, FHYPRE_SSTRUCTMATRIXSETOBJECTTY)
                          (long int *, int *);

#define HYPRE_SStructMatrixGetObject \
        hypre_F90_NAME(fhypre_sstructmatrixgetobject, FHYPRE_SSTRUCTMATRIXGETOBJECT)
extern void hypre_F90_NAME(fhypre_sstructmatrixgetobject, FHYPRE_SSTRUCTMATRIXGETOBJECT)
                          (long int *, long int *);

#define HYPRE_SStructMatrixPrint \
        hypre_F90_NAME(fhypre_sstructmatrixprint, FHYPRE_SSTRUCTMATRIXPRINT)
extern void hypre_F90_NAME(fhypre_sstructmatrixprint, FHYPRE_SSTRUCTMATRIXPRINT)
                          (const char *, long int *, int *);



#define HYPRE_SStructStencilCreate \
        hypre_F90_NAME(fhypre_sstructstencilcreate, FHYPRE_SSTRUCTSTENCILCREATE)
extern void hypre_F90_NAME(fhypre_sstructstencilcreate, FHYPRE_SSTRUCTSTENCILCREATE)
                          (int *, int *, long int *);

#define HYPRE_SStructStencilDestroy \
        hypre_F90_NAME(fhypre_sstructstencildestroy, FHYPRE_SSTRUCTSTENCILDESTROY)
extern void hypre_F90_NAME(fhypre_sstructstencildestroy, FHYPRE_SSTRUCTSTENCILDESTROY)
                          (long int *);

#define HYPRE_SStructStencilSetEntry \
        hypre_F90_NAME(fhypre_sstructstencilsetentry, FHYPRE_SSTRUCTSTENCILSETENTRY)
extern void hypre_F90_NAME(fhypre_sstructstencilsetentry, FHYPRE_SSTRUCTSTENCILSETENTRY)
                          (long int *, int *, int *, int *);



#define HYPRE_SStructVectorCreate \
        hypre_F90_NAME(fhypre_sstructvectorcreate, FHYPRE_SSTRUCTVECTORCREATE)
extern void hypre_F90_NAME(fhypre_sstructvectorcreate, FHYPRE_SSTRUCTVECTORCREATE)
                          (int *, long int *, long int *);

#define HYPRE_SStructVectorDestroy \
        hypre_F90_NAME(fhypre_sstructvectordestroy, FHYPRE_SSTRUCTVECTORDESTROY)
extern void hypre_F90_NAME(fhypre_sstructvectordestroy, FHYPRE_SSTRUCTVECTORDESTROY)
                          (long int *);

#define HYPRE_SStructVectorInitialize \
        hypre_F90_NAME(fhypre_sstructvectorinitialize, FHYPRE_SSTRUCTVECTORINITIALIZE)
extern void hypre_F90_NAME(fhypre_sstructvectorinitialize, FHYPRE_SSTRUCTVECTORINITIALIZE)
                          (long int *);

#define HYPRE_SStructVectorSetValues \
        hypre_F90_NAME(fhypre_sstructvectorsetvalues, FHYPRE_SSTRUCTVECTORSETVALUES)
extern void hypre_F90_NAME(fhypre_sstructvectorsetvalues, FHYPRE_SSTRUCTVECTORSETVALUES)
                          (long int *, int *, int *, int *, double *);

#define HYPRE_SStructVectorSetBoxValues \
        hypre_F90_NAME(fhypre_sstructvectorsetboxvalue, FHYPRE_SSTRUCTVECTORSETBOXVALUE)
extern void hypre_F90_NAME(fhypre_sstructvectorsetboxvalue, FHYPRE_SSTRUCTVECTORSETBOXVALUE)
                          (long int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructVectorAddToValues \
        hypre_F90_NAME(fhypre_sstructvectoraddtovalues, FHYPRE_SSTRUCTVECTORADDTOVALUES)
extern void hypre_F90_NAME(fhypre_sstructvectoraddtovalues, FHYPRE_SSTRUCTVECTORADDTOVALUES)
                          (long int *, int *, int *, int *, double *);

#define HYPRE_SStructVectorAddToBoxValues \
        hypre_F90_NAME(fhypre_sstructvectoraddtoboxval, FHYPRE_SSTRUCTVECTORADDTOBOXVAL)
extern void hypre_F90_NAME(fhypre_sstructvectoraddtoboxval, FHYPRE_SSTRUCTVECTORADDTOBOXVAL)
                          (long int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructVectorAssemble \
        hypre_F90_NAME(fhypre_sstructvectorassemble, FHYPRE_SSTRUCTVECTORASSEMBLE)
extern void hypre_F90_NAME(fhypre_sstructvectorassemble, FHYPRE_SSTRUCTVECTORASSEMBLE)
                          (long int *);

#define HYPRE_SStructVectorGather \
        hypre_F90_NAME(fhypre_sstructvectorgather, FHYPRE_SSTRUCTVECTORGATHER)
extern void hypre_F90_NAME(fhypre_sstructvectorgather, FHYPRE_SSTRUCTVECTORGATHER)
                          (long int *);

#define HYPRE_SStructVectorGetValues \
        hypre_F90_NAME(fhypre_sstructvectorgetvalues, FHYPRE_SSTRUCTVECTORGETVALUES)
extern void hypre_F90_NAME(fhypre_sstructvectorgetvalues, FHYPRE_SSTRUCTVECTORGETVALUES)
                          (long int *, int *, int *, int *, double *);

#define HYPRE_SStructVectorGetBoxValues \
        hypre_F90_NAME(fhypre_sstructvectorgetboxvalue, FHYPRE_SSTRUCTVECTORGETBOXVALUE)
extern void hypre_F90_NAME(fhypre_sstructvectorgetboxvalue, FHYPRE_SSTRUCTVECTORGETBOXVALUE)
                          (long int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructVectorSetObjectType \
        hypre_F90_NAME(fhypre_sstructvectorsetobjectty, FHYPRE_SSTRUCTVECTORSETOBJECTTY)
extern void hypre_F90_NAME(fhypre_sstructvectorsetobjectty, FHYPRE_SSTRUCTVECTORSETOBJECTTY)
                          (long int *, int *);

#define HYPRE_SStructVectorGetObject \
        hypre_F90_NAME(fhypre_sstructvectorgetobject, FHYPRE_SSTRUCTVECTORGETOBJECT)
extern void hypre_F90_NAME(fhypre_sstructvectorgetobject, FHYPRE_SSTRUCTVECTORGETOBJECT)
                          (long int *, void *);

#define HYPRE_SStructVectorPrint \
        hypre_F90_NAME(fhypre_sstructvectorprint, FHYPRE_SSTRUCTVECTORPRINT)
extern void hypre_F90_NAME(fhypre_sstructvectorprint, FHYPRE_SSTRUCTVECTORPRINT)
                          (const char *, long int *, int *);



#define HYPRE_SStructBiCGSTABCreate \
        hypre_F90_NAME(fhypre_sstructbicgstabcreate, FHYPRE_SSTRUCTBICGSTABCREATE)
extern void hypre_F90_NAME(fhypre_sstructbicgstabcreate, FHYPRE_SSTRUCTBICGSTABCREATE)
                          (int *, long int *);

#define HYPRE_SStructBiCGSTABDestroy \
        hypre_F90_NAME(fhypre_sstructbicgstabdestroy, FHYPRE_SSTRUCTBICGSTABDESTROY)
extern void hypre_F90_NAME(fhypre_sstructbicgstabdestroy, FHYPRE_SSTRUCTBICGSTABDESTROY)
                          (long int *);

#define HYPRE_SStructBiCGSTABSetup \
        hypre_F90_NAME(fhypre_sstructbicgstabsetup, FHYPRE_SSTRUCTBICGSTABSETUP)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetup, FHYPRE_SSTRUCTBICGSTABSETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructBiCGSTABSolve \
        hypre_F90_NAME(fhypre_sstructbicgstabsolve, FHYPRE_SSTRUCTBICGSTABSOLVE)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsolve, FHYPRE_SSTRUCTBICGSTABSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructBiCGSTABSetTol \
        hypre_F90_NAME(fhypre_sstructbicgstabsettol, FHYPRE_SSTRUCTBICGSTABSETTOL)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsettol, FHYPRE_SSTRUCTBICGSTABSETTOL)
                          (long int *, double *);

#define HYPRE_SStructBiCGSTABSetMinIter \
        hypre_F90_NAME(fhypre_sstructbicgstabsetminite, FHYPRE_SSTRUCTBICGSTABSETMINITE)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetminite, FHYPRE_SSTRUCTBICGSTABSETMINITE)
                          (long int *, int *);

#define HYPRE_SStructBiCGSTABSetMaxIter \
        hypre_F90_NAME(fhypre_sstructbicgstabsetmaxite, FHYPRE_SSTRUCTBICGSTABSETMAXITE)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetmaxite, FHYPRE_SSTRUCTBICGSTABSETMAXITE)
                          (long int *, int *);

#define HYPRE_SStructBiCGSTABSetStopCrit \
        hypre_F90_NAME(fhypre_sstructbicgstabsetstopcr, FHYPRE_SSTRUCTBICGSTABSETSTOPCR)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetstopcr, FHYPRE_SSTRUCTBICGSTABSETSTOPCR)
                          (long int *, int *);

#define HYPRE_SStructBiCGSTABSetPrecond \
        hypre_F90_NAME(fhypre_sstructbicgstabsetprecon, FHYPRE_SSTRUCTBICGSTABSETPRECON)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetprecon, FHYPRE_SSTRUCTBICGSTABSETPRECON)
                          (long int *, int *, long int *);

#define HYPRE_SStructBiCGSTABSetLogging \
        hypre_F90_NAME(fhypre_sstructbicgstabsetloggin, FHYPRE_SSTRUCTBICGSTABSETLOGGIN)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetloggin, FHYPRE_SSTRUCTBICGSTABSETLOGGIN)
                          (long int *, int *);

#define HYPRE_SStructBiCGSTABSetPrintLevel \
        hypre_F90_NAME(fhypre_sstructbicgstabsetprintl, FHYPRE_SSTRUCTBICGSTABSETPRINTL)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetprintl, FHYPRE_SSTRUCTBICGSTABSETPRINTL)
                          (long int *, int *);

#define HYPRE_SStructBiCGSTABGetNumIterations \
        hypre_F90_NAME(fhypre_sstructbicgstabgetnumite, FHYPRE_SSTRUCTBICGSTABGETNUMITE)
extern void hypre_F90_NAME(fhypre_sstructbicgstabgetnumite, FHYPRE_SSTRUCTBICGSTABGETNUMITE)
                          (long int *, int *);

#define HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_sstructbicgstabgetfinalr, FHYPRE_SSTRUCTBICGSTABGETFINALR)
extern void hypre_F90_NAME(fhypre_sstructbicgstabgetfinalr, FHYPRE_SSTRUCTBICGSTABGETFINALR)
                          (long int *, double *);

#define HYPRE_SStructBiCGSTABGetResidual \
        hypre_F90_NAME(fhypre_sstructbicgstabgetresidu, FHYPRE_SSTRUCTBICGSTABGETRESIDU)
extern void hypre_F90_NAME(fhypre_sstructbicgstabgetresidu, FHYPRE_SSTRUCTBICGSTABGETRESIDU)
                          (long int *, long int *);



#define HYPRE_SStructGMRESCreate \
        hypre_F90_NAME(fhypre_sstructgmrescreate, FHYPRE_SSTRUCTGMRESCREATE)
extern void hypre_F90_NAME(fhypre_sstructgmrescreate, FHYPRE_SSTRUCTGMRESCREATE)
                          (long int *, long int *);

#define HYPRE_SStructGMRESDestroy \
        hypre_F90_NAME(fhypre_sstructgmresdestroy, FHYPRE_SSTRUCTGMRESDESTROY)
extern void hypre_F90_NAME(fhypre_sstructgmresdestroy, FHYPRE_SSTRUCTGMRESDESTROY)
                          (long int *);

#define HYPRE_SStructGMRESSetup \
        hypre_F90_NAME(fhypre_sstructgmressetup, FHYPRE_SSTRUCTGMRESSETUP)
extern void hypre_F90_NAME(fhypre_sstructgmressetup, FHYPRE_SSTRUCTGMRESSETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructGMRESSolve \
        hypre_F90_NAME(fhypre_sstructgmressolve, FHYPRE_SSTRUCTGMRESSOLVE)
extern void hypre_F90_NAME(fhypre_sstructgmressolve, FHYPRE_SSTRUCTGMRESSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructGMRESSetKDim \
        hypre_F90_NAME(fhypre_sstructgmressetkdim, FHYPRE_SSTRUCTGMRESSETKDIM)
extern void hypre_F90_NAME(fhypre_sstructgmressetkdim, FHYPRE_SSTRUCTGMRESSETKDIM)
                          (long int *, int *);

#define HYPRE_SStructGMRESSetTol \
        hypre_F90_NAME(fhypre_sstructgmressettol, FHYPRE_SSTRUCTGMRESSETTOL)
extern void hypre_F90_NAME(fhypre_sstructgmressettol, FHYPRE_SSTRUCTGMRESSETTOL)
                          (long int *, double *);

#define HYPRE_SStructGMRESSetMinIter \
        hypre_F90_NAME(fhypre_sstructgmressetminiter, FHYPRE_SSTRUCTGMRESSETMINITER)
extern void hypre_F90_NAME(fhypre_sstructgmressetminiter, FHYPRE_SSTRUCTGMRESSETMINITER)
                          (long int *, int *);

#define HYPRE_SStructGMRESSetMaxIter \
        hypre_F90_NAME(fhypre_sstructgmressetmaxiter, FHYPRE_SSTRUCTGMRESSETMAXITER)
extern void hypre_F90_NAME(fhypre_sstructgmressetmaxiter, FHYPRE_SSTRUCTGMRESSETMAXITER)
                          (long int *, int *);

#define HYPRE_SStructGMRESSetStopCrit \
        hypre_F90_NAME(fhypre_sstructgmressetstopcrit, FHYPRE_SSTRUCTGMRESSETSTOPCRIT)
extern void hypre_F90_NAME(fhypre_sstructgmressetstopcrit, FHYPRE_SSTRUCTGMRESSETSTOPCRIT)
                          (long int *, int *);

#define HYPRE_SStructGMRESSetPrecond \
        hypre_F90_NAME(fhypre_sstructgmressetprecond, FHYPRE_SSTRUCTGMRESSETPRECOND)
extern void hypre_F90_NAME(fhypre_sstructgmressetprecond, FHYPRE_SSTRUCTGMRESSETPRECOND)
                          (long int *, int *, long int *);


#define HYPRE_SStructGMRESSetLogging \
        hypre_F90_NAME(fhypre_sstructgmressetlogging, FHYPRE_SSTRUCTGMRESSETLOGGING)
extern void hypre_F90_NAME(fhypre_sstructgmressetlogging, FHYPRE_SSTRUCTGMRESSETLOGGING)
                          (long int *, int *);

#define HYPRE_SStructGMRESSetPrintLevel \
        hypre_F90_NAME(fhypre_sstructgmressetprintleve, FHYPRE_SSTRUCTGMRESSETPRINTLEVE)
extern void hypre_F90_NAME(fhypre_sstructgmressetprintleve, FHYPRE_SSTRUCTGMRESSETPRINTLEVE)
                          (long int *, int *);

#define HYPRE_SStructGMRESGetNumIterations \
      hypre_F90_NAME(fhypre_sstructgmresgetnumiterati, FHYPRE_SSTRUCTGMRESGETNUMITERATI)
extern void hypre_F90_NAME(fhypre_sstructgmresgetnumiterati, FHYPRE_SSTRUCTGMRESGETNUMITERATI)
                          (long int *, int *);

#define HYPRE_SStructGMRESGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_sstructgmresgetfinalrela, FHYPRE_SSTRUCTGMRESGETFINALRELA)
extern void hypre_F90_NAME(fhypre_sstructgmresgetfinalrela, FHYPRE_SSTRUCTGMRESGETFINALRELA)
                          (long int *, double  *);

#define HYPRE_SStructGMRESGetResidual \
        hypre_F90_NAME(fhypre_sstructgmresgetresidual, FHYPRE_SSTRUCTGMRESGETRESIDUAL)
extern void hypre_F90_NAME(fhypre_sstructgmresgetresidual, FHYPRE_SSTRUCTGMRESGETRESIDUAL)
                          (long int *, long int *);



#define HYPRE_SStructPCGCreate \
        hypre_F90_NAME(fhypre_sstructpcgcreate, FHYPRE_SSTRUCTPCGCREATE)
extern void hypre_F90_NAME(fhypre_sstructpcgcreate, FHYPRE_SSTRUCTPCGCREATE)
                          (long int *, long int *);

#define HYPRE_SStructPCGDestroy \
        hypre_F90_NAME(fhypre_sstructpcgdestroy, FHYPRE_SSTRUCTPCGDESTROY)
extern void hypre_F90_NAME(fhypre_sstructpcgdestroy, FHYPRE_SSTRUCTPCGDESTROY)
                          (long int *);

#define HYPRE_SStructPCGSetup \
        hypre_F90_NAME(fhypre_sstructpcgsetup, FHYPRE_SSTRUCTPCGDESTROY)
extern void hypre_F90_NAME(fhypre_sstructpcgsetup, FHYPRE_SSTRUCTPCGDESTROY)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructPCGSolve \
        hypre_F90_NAME(fhypre_sstructpcgsolve, FHYPRE_SSTRUCTPCGSOLVE)
extern void hypre_F90_NAME(fhypre_sstructpcgsolve, FHYPRE_SSTRUCTPCGSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructPCGSetTol \
        hypre_F90_NAME(fhypre_sstructpcgsettol, FHYPRE_SSTRUCTPCGSETTOL)
extern void hypre_F90_NAME(fhypre_sstructpcgsettol, FHYPRE_SSTRUCTPCGSETTOL)
                          (long int *, double *);

#define HYPRE_SStructPCGSetMaxIter \
        hypre_F90_NAME(fhypre_sstructpcgsetmaxiter, FHYPRE_SSTRUCTPCGSETMAXITER)
extern void hypre_F90_NAME(fhypre_sstructpcgsetmaxiter, FHYPRE_SSTRUCTPCGSETMAXITER)
                          (long int *, int  *);

#define HYPRE_SStructPCGSetTwoNorm \
        hypre_F90_NAME(fhypre_sstructpcgsettwonorm, FHYPRE_SSTRUCTPCGSETTWONORM)
extern void hypre_F90_NAME(fhypre_sstructpcgsettwonorm, FHYPRE_SSTRUCTPCGSETTWONORM)
                          (long int *, int  *);

#define HYPRE_SStructPCGSetRelChange \
        hypre_F90_NAME(fhypre_sstructpcgsetrelchange, FHYPRE_SSTRUCTPCGSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_sstructpcgsetrelchange, FHYPRE_SSTRUCTPCGSETRELCHANGE)
                          (long int *, int  *);

#define HYPRE_SStructPCGSetPrecond \
        hypre_F90_NAME(fhypre_sstructpcgsetprecond, FHYPRE_SSTRUCTPCGSETPRECOND)
extern void hypre_F90_NAME(fhypre_sstructpcgsetprecond, FHYPRE_SSTRUCTPCGSETPRECOND)
                          (long int *, int  *, long int *);


#define HYPRE_SStructPCGSetLogging \
        hypre_F90_NAME(fhypre_sstructpcgsetlogging, FHYPRE_SSTRUCTPCGSETLOGGING)
extern void hypre_F90_NAME(fhypre_sstructpcgsetlogging, FHYPRE_SSTRUCTPCGSETLOGGING)
                          (long int *, int  *);

#define HYPRE_SStructPCGSetPrintLevel \
        hypre_F90_NAME(fhypre_sstructpcgsetprintlevel, FHYPRE_SSTRUCTPCGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_sstructpcgsetprintlevel, FHYPRE_SSTRUCTPCGSETPRINTLEVEL)
                          (long int *, int  *);

#define HYPRE_SStructPCGGetNumIterations \
        hypre_F90_NAME(fhypre_sstructpcggetnumiteratio, FHYPRE_SSTRUCTPCGGETNUMITERATIO)
extern void hypre_F90_NAME(fhypre_sstructpcggetnumiteratio, FHYPRE_SSTRUCTPCGGETNUMITERATIO)
                          (long int *, int  *);

#define HYPRE_SStructPCGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_sstructpcggetfinalrelati, FHYPRE_SSTRUCTPCGGETFINALRELATI)
extern void hypre_F90_NAME(fhypre_sstructpcggetfinalrelati, FHYPRE_SSTRUCTPCGGETFINALRELATI)
                          (long int *, double *);

#define HYPRE_SStructPCGGetResidual \
        hypre_F90_NAME(fhypre_sstructpcggetresidual, FHYPRE_SSTRUCTPCGGETRESIDUAL)
extern void hypre_F90_NAME(fhypre_sstructpcggetresidual, FHYPRE_SSTRUCTPCGGETRESIDUAL)
                          (long int *, long int *);

#define HYPRE_SStructDiagScaleSetup \
        hypre_F90_NAME(fhypre_sstructdiagscalesetup, FHYPRE_SSTRUCTDIAGSCALESETUP)
extern void hypre_F90_NAME(fhypre_sstructdiagscalesetup, FHYPRE_SSTRUCTDIAGSCALESETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructDiagScale \
        hypre_F90_NAME(fhypre_sstructdiagscale, FHYPRE_SSTRUCTDIAGSCALE)
extern void hypre_F90_NAME(fhypre_sstructdiagscale, FHYPRE_SSTRUCTDIAGSCALE)
                          (long int *, long int *, long int *, long int *);


#define HYPRE_SStructSplitCreate \
        hypre_F90_NAME(fhypre_sstructsplitcreate, FHYPRE_SSTRUCTSPLITCREATE)
extern void hypre_F90_NAME(fhypre_sstructsplitcreate, FHYPRE_SSTRUCTSPLITCREATE)
                          (long int *, long int *);

#define HYPRE_SStructSplitDestroy \
        hypre_F90_NAME(fhypre_sstructsplitdestroy, FHYPRE_SSTRUCTSPLITDESTROY)
extern void hypre_F90_NAME(fhypre_sstructsplitdestroy, FHYPRE_SSTRUCTSPLITDESTROY)
                          (long int *);

#define HYPRE_SStructSplitSetup \
        hypre_F90_NAME(fhypre_sstructsplitsetup, FHYPRE_SSTRUCTSPLITSETUP)
extern void hypre_F90_NAME(fhypre_sstructsplitsetup, FHYPRE_SSTRUCTSPLITSETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructSplitSolve \
        hypre_F90_NAME(fhypre_sstructsplitsolve, FHYPRE_SSTRUCTSPLITSOLVE)
extern void hypre_F90_NAME(fhypre_sstructsplitsolve, FHYPRE_SSTRUCTSPLITSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructSplitSetTol \
        hypre_F90_NAME(fhypre_sstructsplitsettol, FHYPRE_SSTRUCTSPLITSETTOL)
extern void hypre_F90_NAME(fhypre_sstructsplitsettol, FHYPRE_SSTRUCTSPLITSETTOL)
                          (long int *, double *);

#define HYPRE_SStructSplitSetMaxIter \
        hypre_F90_NAME(fhypre_sstructsplitsetmaxiter, FHYPRE_SSTRUCTSPLITSETMAXITER)
extern void hypre_F90_NAME(fhypre_sstructsplitsetmaxiter, FHYPRE_SSTRUCTSPLITSETMAXITER)
                          (long int *, int  *);

#define HYPRE_SStructSplitSetZeroGuess \
        hypre_F90_NAME(fhypre_sstructsplitsetzeroguess, FHYPRE_SSTRUCTSPLITSETZEROGUESS)
extern void hypre_F90_NAME(fhypre_sstructsplitsetzeroguess, FHYPRE_SSTRUCTSPLITSETZEROGUESS)
                          (long int *);

#define HYPRE_SStructSplitSetNonZeroGuess \
        hypre_F90_NAME(fhypre_sstructsplitsetnonzerogu, FHYPRE_SSTRUCTSPLITSETNONZEROGU)
extern void hypre_F90_NAME(fhypre_sstructsplitsetnonzerogu, FHYPRE_SSTRUCTSPLITSETNONZEROGU)
                          (long int *);

#define HYPRE_SStructSplitSetStructSolver \
        hypre_F90_NAME(fhypre_sstructsplitsetstructsol, FHYPRE_SSTRUCTSPLITSETSTRUCTSOL)
extern void hypre_F90_NAME(fhypre_sstructsplitsetstructsol, FHYPRE_SSTRUCTSPLITSETSTRUCTSOL)
                          (long int *, int  *);

#define HYPRE_SStructSplitGetNumIterations \
        hypre_F90_NAME(fhypre_sstructsplitgetnumiterat, FHYPRE_SSTRUCTSPLITGETNUMITERAT)
extern void hypre_F90_NAME(fhypre_sstructsplitgetnumiterat, FHYPRE_SSTRUCTSPLITGETNUMITERAT)
                          (long int *, int  *);

#define HYPRE_SStructSplitGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_sstructsplitgetfinalrela, FHYPRE_SSTRUCTSPLITGETFINALRELA)
extern void hypre_F90_NAME(fhypre_sstructsplitgetfinalrela, FHYPRE_SSTRUCTSPLITGETFINALRELA)
                          (long int *, double *);



#define HYPRE_SStructSysPFMGCreate \
        hypre_F90_NAME(fhypre_sstructsyspfmgcreate, FHYPRE_SSTRUCTSYSPFMGCREATE)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgcreate, FHYPRE_SSTRUCTSYSPFMGCREATE)
                          (long int *, long int *);

#define HYPRE_SStructSysPFMGDestroy \
        hypre_F90_NAME(fhypre_sstructsyspfmgdestroy, FHYPRE_SSTRUCTSYSPFMGDESTROY)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgdestroy, FHYPRE_SSTRUCTSYSPFMGDESTROY)
                          (long int *);

#define HYPRE_SStructSysPFMGSetup \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetup, FHYPRE_SSTRUCTSYSPFMGSETUP)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetup, FHYPRE_SSTRUCTSYSPFMGSETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructSysPFMGSolve \
        hypre_F90_NAME(fhypre_sstructsyspfmgsolve, FHYPRE_SSTRUCTSYSPFMGSOLVE)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsolve, FHYPRE_SSTRUCTSYSPFMGSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructSysPFMGSetTol \
        hypre_F90_NAME(fhypre_sstructsyspfmgsettol, FHYPRE_SSTRUCTSYSPFMGSETTOL)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsettol, FHYPRE_SSTRUCTSYSPFMGSETTOL)
                          (long int *, double *);

#define HYPRE_SStructSysPFMGSetMaxIter \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetmaxiter, FHYPRE_SSTRUCTSYSPFMGSETMAXITER)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetmaxiter, FHYPRE_SSTRUCTSYSPFMGSETMAXITER)
                          (long int *, int *);

#define HYPRE_SStructSysPFMGSetRelChange \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetrelchan, FHYPRE_SSTRUCTSYSPFMGSETRELCHAN)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetrelchan, FHYPRE_SSTRUCTSYSPFMGSETRELCHAN)
                          (long int *, int *);

#define HYPRE_SStructSysPFMGSetZeroGuess \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetzerogue, FHYPRE_SSTRUCTSYSPFMGSETZEROGUE)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetzerogue, FHYPRE_SSTRUCTSYSPFMGSETZEROGUE)
                          (long int *);

#define HYPRE_SStructSysPFMGSetNonZeroGuess \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetnonzero, FHYPRE_SSTRUCTSYSPFMGSETNONZERO)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetnonzero, FHYPRE_SSTRUCTSYSPFMGSETNONZERO)
                          (long int *);

#define HYPRE_SStructSysPFMGSetRelaxType \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetrelaxty, FHYPRE_SSTRUCTSYSPFMGSETRELAXTY)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetrelaxty, FHYPRE_SSTRUCTSYSPFMGSETRELAXTY)
                          (long int *, int *);

#define HYPRE_SStructSysPFMGSetNumPreRelax \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetnumprer, FHYPRE_SSTRUCTSYSPFMGSETNUMPRER)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetnumprer, FHYPRE_SSTRUCTSYSPFMGSETNUMPRER)
                          (long int *, int *);

#define HYPRE_SStructSysPFMGSetNumPostRelax \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetnumpost, FHYPRE_SSTRUCTSYSPFMGSETNUMPOST)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetnumpost, FHYPRE_SSTRUCTSYSPFMGSETNUMPOST)
                          (long int *, int *);


#define HYPRE_SStructSysPFMGSetSkipRelax \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetskiprel, FHYPRE_SSTRUCTSYSPFMGSETSKIPREL)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetskiprel, FHYPRE_SSTRUCTSYSPFMGSETSKIPREL)
                          (long int *, int *);

#define HYPRE_SStructSysPFMGSetDxyz \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetdxyz, FHYPRE_SSTRUCTSYSPFMGSETDXYZ)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetdxyz, FHYPRE_SSTRUCTSYSPFMGSETDXYZ)
                          (long int *, double *);

#define HYPRE_SStructSysPFMGSetLogging \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetlogging, FHYPRE_SSTRUCTSYSPFMGSETLOGGING)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetlogging, FHYPRE_SSTRUCTSYSPFMGSETLOGGING)
                          (long int *, int *);

#define HYPRE_SStructSysPFMGSetPrintLevel \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetprintle, FHYPRE_SSTRUCTSYSPFMGSETPRINTLE)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetprintle, FHYPRE_SSTRUCTSYSPFMGSETPRINTLE)
                          (long int *, int *);

#define HYPRE_SStructSysPFMGGetNumIterations \
        hypre_F90_NAME(fhypre_sstructsyspfmggetnumiter, FHYPRE_SSTRUCTSYSPFMGGETNUMITER)
extern void hypre_F90_NAME(fhypre_sstructsyspfmggetnumiter, FHYPRE_SSTRUCTSYSPFMGGETNUMITER)
                          (long int *, int *);


#define HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_sstructsyspfmggetfinalre, FHYPRE_SSTRUCTSYSPFMGGETFINALRE)
extern void hypre_F90_NAME(fhypre_sstructsyspfmggetfinalre, FHYPRE_SSTRUCTSYSPFMGGETFINALRE)
                          (long int *, double *);



#define HYPRE_SStructMaxwellCreate \
        hypre_F90_NAME(fhypre_sstructmaxwellcreate, FHYPRE_SSTRUCTMAXWELLCREATE)
extern void hypre_F90_NAME(fhypre_sstructmaxwellcreate, FHYPRE_SSTRUCTMAXWELLCREATE)
                          (long int *, long int *);

#define HYPRE_SStructMaxwellDestroy \
        hypre_F90_NAME(fhypre_sstructmaxwelldestroy, FHYPRE_SSTRUCTMAXWELLDESTROY)
extern void hypre_F90_NAME(fhypre_sstructmaxwelldestroy, FHYPRE_SSTRUCTMAXWELLDESTROY)
                          (long int *);

#define HYPRE_SStructMaxwellSetup \
        hypre_F90_NAME(fhypre_sstructmaxwellsetup, FHYPRE_SSTRUCTMAXWELLSETUP)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetup, FHYPRE_SSTRUCTMAXWELLSETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructMaxwellSolve \
        hypre_F90_NAME(fhypre_sstructmaxwellsolve, FHYPRE_SSTRUCTMAXWELLSOLVE)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsolve, FHYPRE_SSTRUCTMAXWELLSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructMaxwellSolve2 \
        hypre_F90_NAME(fhypre_sstructmaxwellsolve2, FHYPRE_SSTRUCTMAXWELLSOLVE2)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsolve2, FHYPRE_SSTRUCTMAXWELLSOLVE2)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_MaxwellGrad \
        hypre_F90_NAME(fhypre_maxwellgrad, FHYPRE_MAXWELLGRAD)
extern void hypre_F90_NAME(fhypre_maxwellgrad, FHYPRE_MAXWELLGRAD)
                          (long int *, long int *);

#define HYPRE_SStructMaxwellSetGrad \
        hypre_F90_NAME(fhypre_sstructmaxwellsetgrad, FHYPRE_SSTRUCTMAXWELLSETGRAD)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetgrad, FHYPRE_SSTRUCTMAXWELLSETGRAD)
                          (long int *, long int *);

#define HYPRE_SStructMaxwellSetRfactors \
        hypre_F90_NAME(fhypre_sstructmaxwellsetrfactor, FHYPRE_SSTRUCTMAXWELLSETRFACTOR)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetrfactor, FHYPRE_SSTRUCTMAXWELLSETRFACTOR)
                          (long int *, int *);

#define HYPRE_SStructMaxwellSetTol \
        hypre_F90_NAME(fhypre_sstructmaxwellsettol, FHYPRE_SSTRUCTMAXWELLSETTOL)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsettol, FHYPRE_SSTRUCTMAXWELLSETTOL)
                          (long int *, double *);

#define HYPRE_SStructMaxwellSetConstantCoef \
        hypre_F90_NAME(fhypre_sstructmaxwellsetconstan, FHYPRE_SSTRUCTMAXWELLSETCONSTAN)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetconstan, FHYPRE_SSTRUCTMAXWELLSETCONSTAN)
                          (long int *, int *);

#define HYPRE_SStructMaxwellSetMaxIter \
        hypre_F90_NAME(fhypre_sstructmaxwellsetmaxiter, FHYPRE_SSTRUCTMAXWELLSETMAXITER)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetmaxiter, FHYPRE_SSTRUCTMAXWELLSETMAXITER)
                          (long int *, int *);

#define HYPRE_SStructMaxwellSetRelChange \
        hypre_F90_NAME(fhypre_sstructmaxwellsetrelchan, FHYPRE_SSTRUCTMAXWELLSETRELCHAN)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetrelchan, FHYPRE_SSTRUCTMAXWELLSETRELCHAN)
                          (long int *, int *);

#define HYPRE_SStructMaxwellSetNumPreRelax \
        hypre_F90_NAME(fhypre_sstructmaxwellsetnumprer, FHYPRE_SSTRUCTMAXWELLSETNUMPRER)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetnumprer, FHYPRE_SSTRUCTMAXWELLSETNUMPRER)
                          (long int *, int *);

#define HYPRE_SStructMaxwellSetNumPostRelax \
        hypre_F90_NAME(fhypre_sstructmaxwellsetnumpost, FHYPRE_SSTRUCTMAXWELLSETNUMPOST)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetnumpost, FHYPRE_SSTRUCTMAXWELLSETNUMPOST)
                          (long int *, int *);

#define HYPRE_SStructMaxwellSetLogging \
        hypre_F90_NAME(fhypre_sstructmaxwellsetlogging, FHYPRE_SSTRUCTMAXWELLSETLOGGING)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetlogging, FHYPRE_SSTRUCTMAXWELLSETLOGGING)
                          (long int *, int *);

#define HYPRE_SStructMaxwellSetPrintLevel \
        hypre_F90_NAME(fhypre_sstructmaxwellsetprintle, FHYPRE_SSTRUCTMAXWELLSETPRINTLE)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetprintle, FHYPRE_SSTRUCTMAXWELLSETPRINTLE)
                          (long int *, int *);

#define HYPRE_SStructMaxwellPrintLogging \
        hypre_F90_NAME(fhypre_sstructmaxwellprintloggi, FHYPRE_SSTRUCTMAXWELLPRINTLOGGI)
extern void hypre_F90_NAME(fhypre_sstructmaxwellprintloggi, FHYPRE_SSTRUCTMAXWELLPRINTLOGGI)
                          (long int *, int *);

#define HYPRE_SStructMaxwellGetNumIterations \
        hypre_F90_NAME(fhypre_sstructmaxwellgetnumiter, FHYPRE_SSTRUCTMAXWELLGETNUMITER) 
extern void hypre_F90_NAME(fhypre_sstructmaxwellgetnumiter, FHYPRE_SSTRUCTMAXWELLGETNUMITER) 
                          (long int *, int *);

#define HYPRE_SStructMaxwellGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_sstructmaxwellgetfinalre, FHYPRE_SSTRUCTMAXWELLGETFINALRE) 
extern void hypre_F90_NAME(fhypre_sstructmaxwellgetfinalre, FHYPRE_SSTRUCTMAXWELLGETFINALRE) 
                          (long int *, double *);

#define HYPRE_SStructMaxwellPhysBdy \
        hypre_F90_NAME(fhypre_sstructmaxwellphysbdy, FHYPRE_SSTRUCTMAXWELLPHYSBDY) 
extern void hypre_F90_NAME(fhypre_sstructmaxwellphysbdy, FHYPRE_SSTRUCTMAXWELLPHYSBDY) 
                          (long int *, int *, int *, int *, int *);

#define HYPRE_SStructMaxwellEliminateRowsCols \
        hypre_F90_NAME(fhypre_sstructmaxwelleliminater, FHYPRE_SSTRUCTMAXWELLELIMINATER) 
extern void hypre_F90_NAME(fhypre_sstructmaxwelleliminater, FHYPRE_SSTRUCTMAXWELLELIMINATER) 
                          (long int *, int *, int *);

#define HYPRE_SStructMaxwellZeroVector \
        hypre_F90_NAME(fhypre_sstructmaxwellzerovector, FHYPRE_SSTRUCTMAXWELLZEROVECTOR) 
extern void hypre_F90_NAME(fhypre_sstructmaxwellzerovector, FHYPRE_SSTRUCTMAXWELLZEROVECTOR) 
                          (long int *, int *, int *);
