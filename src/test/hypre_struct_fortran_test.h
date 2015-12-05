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

/**************************************************
*  Definitions of struct fortran interface routines
**************************************************/

#define HYPRE_StructStencilCreate \
        hypre_F90_NAME(fhypre_structstencilcreate, FHYPRE_STRUCTSTENCILCREATE)
extern void hypre_F90_NAME(fhypre_structstencilcreate, FHYPRE_STRUCTSTENCILCREATE)
                          (HYPRE_Int *, HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_StructStencilDestroy \
        hypre_F90_NAME(fhypre_structstencildestroy, FHYPRE_STRUCTSTENCILDESTROY)
extern void hypre_F90_NAME(fhypre_structstencildestroy, FHYPRE_STRUCTSTENCILDESTROY)
                          (hypre_F90_Obj *);

#define HYPRE_StructStencilSetElement \
        hypre_F90_NAME(fhypre_structstencilsetelement, FHYPRE_STRUCTSTENCILSETELEMENT)
extern void hypre_F90_NAME(fhypre_structstencilsetelement, FHYPRE_STRUCTSTENCILSETELEMENT)
                          (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *);



#define HYPRE_StructGridCreate \
        hypre_F90_NAME(fhypre_structgridcreate, FHYPRE_STRUCTGRIDCREATE)
extern void hypre_F90_NAME(fhypre_structgridcreate, FHYPRE_STRUCTGRIDCREATE)
                          (HYPRE_Int *, HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_StructGridDestroy \
        hypre_F90_NAME(fhypre_structgriddestroy, FHYPRE_STRUCTGRIDDESTROY)
extern void hypre_F90_NAME(fhypre_structgriddestroy, FHYPRE_STRUCTGRIDDESTROY)
                          (hypre_F90_Obj *);

#define HYPRE_StructGridSetExtents \
        hypre_F90_NAME(fhypre_structgridsetextents, FHYPRE_STRUCTGRIDSETEXTENTS)
extern void hypre_F90_NAME(fhypre_structgridsetextents, FHYPRE_STRUCTGRIDSETEXTENTS)
                          (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *);

#define HYPRE_StructGridSetPeriodic \
        hypre_F90_NAME(fhypre_structgridsetperiodic, FHYPRE_STRUCTGRIDSETPERIODIC)
extern void hypre_F90_NAME(fhypre_structgridsetperiodic, fhypre_structsetgridperiodic)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructGridAssemble \
        hypre_F90_NAME(fhypre_structgridassemble, FHYPRE_STRUCTGRIDASSEMBLE)
extern void hypre_F90_NAME(fhypre_structgridassemble, FHYPRE_STRUCTGRIDASSEMBLE)
                          (hypre_F90_Obj *);

#define HYPRE_StructGridSetNumGhost \
        hypre_F90_NAME(fhypre_structgridsetnumghost, FHYPRE_STRUCTGRIDSETNUMGHOST)
extern void hypre_F90_NAME(fhypre_structgridsetnumghost, fhypre_structsetgridnumghost)
                          (hypre_F90_Obj *, HYPRE_Int *);
        


#define HYPRE_StructMatrixCreate \
        hypre_F90_NAME(fhypre_structmatrixcreate, FHYPRE_STRUCTMATRIXCREATE)
extern void hypre_F90_NAME(fhypre_structmatrixcreate, FHYPRE_STRUCTMATRIXCREATE)
                          (HYPRE_Int *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructMatrixDestroy \
        hypre_F90_NAME(fhypre_structmatrixdestroy, FHYPRE_STRUCTMATRIXDESTROY)
extern void hypre_F90_NAME(fhypre_structmatrixdestroy, FHYPRE_STRUCTMATRIXDESTROY)
                          (hypre_F90_Obj *);

#define HYPRE_StructMatrixInitialize \
        hypre_F90_NAME(fhypre_structmatrixinitialize, FHYPRE_STRUCTMATRIXINITIALIZE)
extern void hypre_F90_NAME(fhypre_structmatrixinitialize, FHYPRE_STRUCTMATRIXINITIALIZE)
                          (hypre_F90_Obj *);

#define HYPRE_StructMatrixSetValues \
        hypre_F90_NAME(fhypre_structmatrixsetvalues, FHYPRE_STRUCTMATRIXSETVALUES)
extern void hypre_F90_NAME(fhypre_structmatrixsetvalues, FHYPRE_STRUCTMATRIXSETVALUES)
                          (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *);

#define HYPRE_StructMatrixSetBoxValues \
        hypre_F90_NAME(fhypre_structmatrixsetboxvalues, FHYPRE_STRUCTMATRIXSETBOXVALUES)
extern void hypre_F90_NAME(fhypre_structmatrixsetboxvalues, FHYPRE_STRUCTMATRIXSETBOXVALUES)
                          (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, double *);

#define HYPRE_StructMatrixGetBoxValues \
        hypre_F90_NAME(fhypre_structmatrixgetboxvalues, FHYPRE_STRUCTMATRIXGETBOXVALUES)
extern void hypre_F90_NAME(fhypre_structmatrixgetboxvalues, FHYPRE_STRUCTMATRIXGETBOXVALUES)
                          (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, double *);

#define HYPRE_StructMatrixSetConstantEntries \
        hypre_F90_NAME(fhypre_structmatrixsetconstante, FHYPRE_STRUCTMATRIXSETCONSTANTE)
extern void hypre_F90_NAME(fhypre_structmatrixsetconstante, FHYPRE_STRUCTMATRIXSETCONSTANTE)
                          (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *);

#define HYPRE_StructMatrixSetConstantValues \
        hypre_F90_NAME(fhypre_structmatrixsetconstantv, FHYPRE_STRUCTMATRIXSETCONSTANTV)
extern void hypre_F90_NAME(fhypre_structmatrixsetconstantv, FHYPRE_STRUCTMATRIXSETCONSTANTV)
                          (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *, double *);

#define HYPRE_StructMatrixAddToValues \
        hypre_F90_NAME(fhypre_structmatrixaddtovalues, FHYPRE_STRUCTMATRIXADDTOVALUES)
extern void hypre_F90_NAME(fhypre_structmatrixaddtovalues, FHYPRE_STRUCTMATRIXADDTOVALUES)
                          (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, double *);

#define HYPRE_StructMatrixAddToBoxValues \
        hypre_F90_NAME(fhypre_structmatrixaddtoboxvalues, FHYPRE_STRUCTMATRIXADDTOBOXVALUES)
extern void hypre_F90_NAME(fhypre_structmatrixaddtoboxvalues, FHYPRE_STRUCTMATRIXADDTOBOXVALUES)
                          (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, double *);

#define HYPRE_StructMatrixAddToConstantValues \
        hypre_F90_NAME(fhypre_structmatrixaddtoconstant, FHYPRE_STRUCTMATRIXADDTOCONSTANT)
extern void hypre_F90_NAME(fhypre_structmatrixaddtoconstant, FHYPRE_STRUCTMATRIXADDTOCONSTANT)
                          (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *, double *);

#define HYPRE_StructMatrixAssemble \
        hypre_F90_NAME(fhypre_structmatrixassemble, FHYPRE_STRUCTMATRIXASSEMBLE)
extern void hypre_F90_NAME(fhypre_structmatrixassemble, FHYPRE_STRUCTMATRIXASSEMBLE)
                          (hypre_F90_Obj *);

#define HYPRE_StructMatrixSetNumGhost \
        hypre_F90_NAME(fhypre_structmatrixsetnumghost, FHYPRE_STRUCTMATRIXSETNUMGHOST)
extern void hypre_F90_NAME(fhypre_structmatrixsetnumghost, FHYPRE_STRUCTMATRIXSETNUMGHOST)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructMatrixGetGrid \
        hypre_F90_NAME(fhypre_structmatrixgetgrid, FHYPRE_STRUCTMATRIXGETGRID)
extern void hypre_F90_NAME(fhypre_structmatrixgetgrid, FHYPRE_STRUCTMATRIXGETGRID)
                          (hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructMatrixSetSymmetric \
        hypre_F90_NAME(fhypre_structmatrixsetsymmetric, FHYPRE_STRUCTMATRIXSETSYMMETRIC)
extern void hypre_F90_NAME(fhypre_structmatrixsetsymmetric, FHYPRE_STRUCTMATRIXSETSYMMETRIC)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructMatrixPrint \
hypre_F90_NAME(fhypre_structmatrixprint, FHYPRE_STRUCTMATRIXPRINT)
extern void hypre_F90_NAME(fhypre_structmatrixprint, FHYPRE_STRUCTMATRIXPRINT)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructMatrixMatvec \
hypre_F90_NAME(fhypre_structmatrixmatvec, FHYPRE_STRUCTMATRIXMATVEC)
extern void hypre_F90_NAME(fhypre_structmatrixmatvec, FHYPRE_STRUCTMATRIXMATVEC)
                          (HYPRE_Int *, hypre_F90_Obj *, hypre_F90_Obj *, HYPRE_Int *, hypre_F90_Obj *);
 


#define HYPRE_StructVectorCreate \
        hypre_F90_NAME(fhypre_structvectorcreate, FHYPRE_STRUCTVECTORCREATE)
extern void hypre_F90_NAME(fhypre_structvectorcreate, FHYPRE_STRUCTVECTORCREATE)
                          (HYPRE_Int *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructVectorDestroy \
        hypre_F90_NAME(fhypre_structvectordestroy, FHYPRE_STRUCTVECTORDESTROY)
extern void hypre_F90_NAME(fhypre_structvectordestroy, FHYPRE_STRUCTVECTORDESTROY) 
                          (hypre_F90_Obj *);

#define HYPRE_StructVectorInitialize \
        hypre_F90_NAME(fhypre_structvectorinitialize, FHYPRE_STRUCTVECTORINITIALIZE)
extern void hypre_F90_NAME(fhypre_structvectorinitialize, FHYPRE_STRUCTVECTORINITIALIZE)
                          (hypre_F90_Obj *);

#define HYPRE_StructVectorSetValues \
        hypre_F90_NAME(fhypre_structvectorsetvalues, FHYPRE_STRUCTVECTORSETVALUES)
extern void hypre_F90_NAME(fhypre_structvectorsetvalues, FHYPRE_STRUCTVECTORSETVALUES)
                          (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *);

#define HYPRE_StructVectorSetBoxValues \
        hypre_F90_NAME(fhypre_structvectorsetboxvalues, FHYPRE_STRUCTVECTORSETBOXVALUES)
extern void hypre_F90_NAME(fhypre_structvectorsetboxvalues, FHYPRE_STRUCTVECTORSETBOXVALUES)
                          (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *, double *);

#define HYPRE_StructVectorSetConstantValues \
        hypre_F90_NAME(fhypre_structvectorsetconstantv, FHYPRE_STRUCTVECTORSETCONTANTV)
extern void hypre_F90_NAME(fhypre_structvectorsetconstantv, FHYPRE_STRUCTVECTORSETCONTANTV)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructVectorAddToValues \
        hypre_F90_NAME(fhypre_structvectoraddtovalues, FHYPRE_STRUCTVECTORADDTOVALUES)
extern void hypre_F90_NAME(fhypre_structvectoraddtovalues, FHYPRE_STRUCTVECTORADDTOVALUES)
                          (hypre_F90_Obj *, HYPRE_Int *, double *);

#define HYPRE_StructVectorAddToBoxValues \
        hypre_F90_NAME(fhypre_structvectoraddtoboxvalu, FHYPRE_STRUCTVECTORADDTOBOXVALU)
extern void hypre_F90_NAME(fhypre_structvectoraddtoboxvalu, FHYPRE_STRUCTVECTORADDTOBOXVALU)
                          (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *, double *);

#define HYPRE_StructVectorScaleValues \
        hypre_F90_NAME(fhypre_structvectorscalevalues, FHYPRE_STRUCTVECTORSCALEVALUES)
extern void hypre_F90_NAME(fhypre_structvectorscalevalues, FHYPRE_STRUCTVECTORSCALEVALUES)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructVectorGetValues \
        hypre_F90_NAME(fhypre_structvectorgetvalues, FHYPRE_STRUCTVECTORGETVALUES)
extern void hypre_F90_NAME(fhypre_structvectorgetvalues, FHYPRE_STRUCTVECTORGETVALUES)
                          (hypre_F90_Obj *, HYPRE_Int *, double *);

#define HYPRE_StructVectorGetBoxValues \
        hypre_F90_NAME(fhypre_structvectorgetboxvalues, FHYPRE_STRUCTVECTORGETBOXVALUES)
extern void hypre_F90_NAME(fhypre_structvectorgetboxvalues, FHYPRE_STRUCTVECTORGETBOXVALUES)
                          (hypre_F90_Obj *, HYPRE_Int *, double *);

#define HYPRE_StructVectorAssemble \
        hypre_F90_NAME(fhypre_structvectorassemble, FHYPRE_STRUCTVECTORASSEMBLE)
extern void hypre_F90_NAME(fhypre_structvectorassemble, FHYPRE_STRUCTVECTORASSEMBLE)
                          (hypre_F90_Obj *);

#define HYPRE_StructVectorSetNumGhost \
        hypre_F90_NAME(fhypre_structvectorsetnumghost, FHYPRE_STRUCTVECTORSETNUMGHOST)
extern void hypre_F90_NAME(fhypre_structvectorsetnumghost, FHYPRE_STRUCTVECTORSETNUMGHOST)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructVectorCopy \
        hypre_F90_NAME(fhypre_structvectorcopy, FHYPRE_STRUCTVECTORCOPY)
extern void hypre_F90_NAME(fhypre_structvectorcopy, FHYPRE_STRUCTVECTORCOPY)
                          (hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructVectorGetMigrateCommPkg \
        hypre_F90_NAME(fhypre_structvectorgetmigrateco, FHYPRE_STRUCTVECTORGETMIGRATECO)
extern void hypre_F90_NAME(fhypre_structvectorgetmigrateco, FHYPRE_STRUCTVECTORGETMIGRATECO)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructVectorMigrate \
        hypre_F90_NAME(fhypre_structvectormigrate, FHYPRE_STRUCTVECTORMIGRATE)
extern void hypre_F90_NAME(fhypre_structvectormigrate, FHYPRE_STRUCTVECTORMIGRATE)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_CommPkgDestroy \
        hypre_F90_NAME(fhypre_commpkgdestroy, FHYPRE_COMMPKGDESTROY)
extern void hypre_F90_NAME(fhypre_commpkgdestroy, FHYPRE_COMMPKGDESTROY)
                          (hypre_F90_Obj *);

#define HYPRE_StructVectorPrint \
        hypre_F90_NAME(fhypre_structvectorprint, FHYPRE_STRUCTVECTORPRINT)
extern void hypre_F90_NAME(fhypre_structvectorprint, FHYPRE_STRUCTVECTORPRINT)
                          (hypre_F90_Obj *, HYPRE_Int *);
 

#define HYPRE_StructBiCGSTABCreate \
        hypre_F90_NAME(fhypre_structbicgstabcreate, FHYPRE_STRUCTBICGSTABCREATE)
extern void hypre_F90_NAME(fhypre_structbicgstabcreate, FHYPRE_STRUCTBICGSTABCREATE)
                          (HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_StructBiCGSTABDestroy \
        hypre_F90_NAME(fhypre_structbicgstabdestroy, FHYPRE_STRUCTBICGSTABDESTROY)
extern void hypre_F90_NAME(fhypre_structbicgstabdestroy, FHYPRE_STRUCTBICGSTABDESTROY)
                          (hypre_F90_Obj *);

#define HYPRE_StructBiCGSTABSetup \
        hypre_F90_NAME(fhypre_structbicgstabsetup, FHYPRE_STRUCTBICGSTABSETUP)
extern void hypre_F90_NAME(fhypre_structbicgstabsetup, FHYPRE_STRUCTBICGSTABSETUP)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructBiCGSTABSolve \
        hypre_F90_NAME(fhypre_structbicgstabsolve, FHYPRE_STRUCTBICGSTABSOLVE)
extern void hypre_F90_NAME(fhypre_structbicgstabsolve, FHYPRE_STRUCTBICGSTABSOLVE)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructBiCGSTABSetTol \
        hypre_F90_NAME(fhypre_structbicgstabsettol, FHYPRE_STRUCTBICGSTABSETTOL)
extern void hypre_F90_NAME(fhypre_structbicgstabsettol, FHYPRE_STRUCTBICGSTABSETTOL)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructBiCGSTABSetMaxIter \
        hypre_F90_NAME(fhypre_structbicgstabsetmaxiter, FHYPRE_STRUCTBICGSTABSETMAXITER)
extern void hypre_F90_NAME(fhypre_structbicgstabsetmaxiter, FHYPRE_STRUCTBICGSTABSETMAXITER)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructBiCGSTABSetPrecond \
        hypre_F90_NAME(fhypre_structbicgstabsetprecond, FHYPRE_STRUCTBICGSTABSETPRECOND)
extern void hypre_F90_NAME(fhypre_structbicgstabsetprecond, FHYPRE_STRUCTBICGSTABSETPRECOND)
                          (hypre_F90_Obj *, HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_StructBiCGSTABSetLogging \
        hypre_F90_NAME(fhypre_structbicgstabsetlogging, FHYPRE_STRUCTBICGSTABSETLOGGING)
extern void hypre_F90_NAME(fhypre_structbicgstabsetlogging, FHYPRE_STRUCTBICGSTABSETLOGGING)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructBiCGSTABSetPrintLevel \
        hypre_F90_NAME(fhypre_structbicgstabsetprintle, FHYPRE_STRUCTBICGSTABPRINTLE)
extern void hypre_F90_NAME(fhypre_structbicgstabsetprintle, FHYPRE_STRUCTBICGSTABPRINTLE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructBiCGSTABGetNumIterations \
        hypre_F90_NAME(fhypre_structbicgstabgetnumiter, FHYPRE_STRUCTBICGSTABGETNUMITER)
extern void hypre_F90_NAME(fhypre_structbicgstabgetnumiter, FHYPRE_STRUCTBICGSTABGETNUMITER)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructBiCGSTABGetResidual \
        hypre_F90_NAME(fhypre_structbicgstabgetresidua, FHYPRE_STRUCTBICGSTABGETRESIDUA)
extern void hypre_F90_NAME(fhypre_structbicgstabgetresidua, FHYPRE_STRUCTBICGSTABGETRESIDUA)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structbicgstabgetfinalre, FHYPRE_STRUCTBICGSTABGETFINALRE)
extern void hypre_F90_NAME(fhypre_structbicgstabgetfinalre, FHYPRE_STRUCTBICGSTABGETFINALRE)
                          (hypre_F90_Obj *, double *);



#define HYPRE_StructGMRESCreate \
        hypre_F90_NAME(fhypre_structgmrescreate, FHYPRE_STRUCTGMRESCREATE)
extern void hypre_F90_NAME(fhypre_structgmrescreate, FHYPRE_STRUCTGMRESCREATE)
                          (HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_StructGMRESDestroy \
        hypre_F90_NAME(fhypre_structgmresdestroy, FHYPRE_STRUCTGMRESDESTROY)
extern void hypre_F90_NAME(fhypre_structgmresdestroy, FHYPRE_STRUCTGMRESDESTROY)
                          (hypre_F90_Obj *);

#define HYPRE_StructGMRESSetup \
        hypre_F90_NAME(fhypre_structgmressetup, FHYPRE_STRUCTGMRESSETUP)
extern void hypre_F90_NAME(fhypre_structgmressetup, FHYPRE_STRUCTGMRESSETUP)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructGMRESSolve \
        hypre_F90_NAME(fhypre_structgmressolve, FHYPRE_STRUCTGMRESSOLVE)
extern void hypre_F90_NAME(fhypre_structgmressolve, FHYPRE_STRUCTGMRESSOLVE)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructGMRESSetTol \
        hypre_F90_NAME(fhypre_structgmressettol, FHYPRE_STRUCTGMRESSETTOL)
extern void hypre_F90_NAME(fhypre_structgmressettol, FHYPRE_STRUCTGMRESSETTOL)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructGMRESSetMaxIter \
        hypre_F90_NAME(fhypre_structgmressetmaxiter, FHYPRE_STRUCTGMRESSETMAXITER)
extern void hypre_F90_NAME(fhypre_structgmressetmaxiter, FHYPRE_STRUCTGMRESSETMAXITER)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructGMRESSetPrecond \
        hypre_F90_NAME(fhypre_structgmressetprecond, FHYPRE_STRUCTGMRESSETPRECOND)
extern void hypre_F90_NAME(fhypre_structgmressetprecond, FHYPRE_STRUCTGMRESSETPRECOND)
                          (hypre_F90_Obj *, HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_StructGMRESSetLogging \
        hypre_F90_NAME(fhypre_structgmressetlogging, FHYPRE_STRUCTGMRESSETLOGGING)
extern void hypre_F90_NAME(fhypre_structgmressetlogging, FHYPRE_STRUCTGMRESSETLOGGING)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructGMRESSetPrintLevel \
        hypre_F90_NAME(fhypre_structgmressetprintlevel, FHYPRE_STRUCTGMRESPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structgmressetprintlevel, FHYPRE_STRUCTGMRESPRINTLEVEL)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructGMRESGetNumIterations \
        hypre_F90_NAME(fhypre_structgmresgetnumiterati, FHYPRE_STRUCTGMRESGETNUMITERATI)
extern void hypre_F90_NAME(fhypre_structgmresgetnumiterati, FHYPRE_STRUCTGMRESGETNUMITERATI)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructGMRESGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structgmresgetfinalrelat, FHYPRE_STRUCTGMRESGETFINALRELAT)
extern void hypre_F90_NAME(fhypre_structgmresgetfinalrelat, FHYPRE_STRUCTGMRESGETFINALRELAT)
                          (hypre_F90_Obj *, double *);



#define HYPRE_StructHybridCreate \
        hypre_F90_NAME(fhypre_structhybridcreate, FHYPRE_STRUCTHYBRIDCREATE)
extern void hypre_F90_NAME(fhypre_structhybridcreate, FHYPRE_STRUCTHYBRIDCREATE)
                          (HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_StructHybridDestroy \
        hypre_F90_NAME(fhypre_structhybriddestroy, FHYPRE_STRUCTHYBRIDDESTROY)
extern void hypre_F90_NAME(fhypre_structhybriddestroy, FHYPRE_STRUCTHYBRIDDESTROY)
                          (hypre_F90_Obj *);

#define HYPRE_StructHybridSetup \
        hypre_F90_NAME(fhypre_structhybridsetup, FHYPRE_STRUCTHYBRIDSETUP)
extern void hypre_F90_NAME(fhypre_structhybridsetup, FHYPRE_STRUCTHYBRIDSETUP)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructHybridSolve \
        hypre_F90_NAME(fhypre_structhybridsolve, FHYPRE_STRUCTHYBRIDSOLVE)
extern void hypre_F90_NAME(fhypre_structhybridsolve, FHYPRE_STRUCTHYBRIDSOLVE)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructHybridSetSolverType \
        hypre_F90_NAME(fhypre_structhybridsetsolvertyp, FHYPRE_STRUCTHYBRIDSETSOLVERTYP)
extern void hypre_F90_NAME(fhypre_structhybridsetsolvertyp, FHYPRE_STRUCTHYBRIDSETSOLVERTYP)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructHybridSetStopCrit \
        hypre_F90_NAME(fhypre_structhybridsetstopcrit, FHYPRE_STRUCTHYBRIDSETSTOPCRIT)
extern void hypre_F90_NAME(fhypre_structhybridsetstopcrit, FHYPRE_STRUCTHYBRIDSETSTOPCRIT)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructHybridSetKDim \
        hypre_F90_NAME(fhypre_structhybridsetkdim, FHYPRE_STRUCTHYBRIDSETKDIM)
extern void hypre_F90_NAME(fhypre_structhybridsetkdim, FHYPRE_STRUCTHYBRIDSETKDIM)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructHybridSetTol \
        hypre_F90_NAME(fhypre_structhybridsettol, FHYPRE_STRUCTHYBRIDSETTOL)
extern void hypre_F90_NAME(fhypre_structhybridsettol, FHYPRE_STRUCTHYBRIDSETTOL)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructHybridSetConvergenceTol \
        hypre_F90_NAME(fhypre_structhybridsetconvergen, FHYPRE_STRUCTHYBRIDSETCONVERGEN)
extern void hypre_F90_NAME(fhypre_structhybridsetconvergen, FHYPRE_STRUCTHYBRIDSETCONVERGEN)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructHybridSetPCGAbsoluteTolFactor \
        hypre_F90_NAME(fhypre_structhybridsetpcgabsolu, FHYPRE_STRUCTHYBRIDSETABSOLU)
extern void hypre_F90_NAME(fhypre_structhybridsetpcgabsolu, FHYPRE_STRUCTHYBRIDSETABSOLU)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructHybridSetMaxIter \
        hypre_F90_NAME(fhypre_structhybridsetmaxiter, FHYPRE_STRUCTHYBRIDSETMAXITER)
extern void hypre_F90_NAME(fhypre_structhybridsetmaxiter, FHYPRE_STRUCTHYBRIDSETMAXITER)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructHybridSetDSCGMaxIter \
        hypre_F90_NAME(fhypre_structhybridsetdscgmaxit, FHYPRE_STRUCTHYBRIDSETDSCGMAXIT)
extern void hypre_F90_NAME(fhypre_structhybridsetdscgmaxit, FHYPRE_STRUCTHYBRIDSETDSCGMAXIT)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructHybridSetPCGMaxIter \
        hypre_F90_NAME(fhypre_structhybridsetpcgmaxite, FHYPRE_STRUCTHYBRIDSETPCGMAXITE)
extern void hypre_F90_NAME(fhypre_structhybridsetpcgmaxite, FHYPRE_STRUCTHYBRIDSETPCGMAXITE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructHybridSetTwoNorm \
        hypre_F90_NAME(fhypre_structhybridsettwonorm, FHYPRE_STRUCTHYBRIDSETTWONORM)
extern void hypre_F90_NAME(fhypre_structhybridsettwonorm, FHYPRE_STRUCTHYBRIDSETTWONORM)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructHybridSetRelChange \
        hypre_F90_NAME(fhypre_structhybridsetrelchange, FHYPRE_STRUCTHYBRIDSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structhybridsetrelchange, FHYPRE_STRUCTHYBRIDSETRELCHANGE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructHybridSetPrecond \
        hypre_F90_NAME(fhypre_structhybridsetprecond, FHYPRE_STRUCTHYBRIDSETPRECOND)
extern void hypre_F90_NAME(fhypre_structhybridsetprecond, FHYPRE_STRUCTHYBRIDSETPRECOND) 
                          (hypre_F90_Obj *, HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_StructHybridSetLogging \
        hypre_F90_NAME(fhypre_structhybridsetlogging, FHYPRE_STRUCTHYBRIDSETLOGGING)
extern void hypre_F90_NAME(fhypre_structhybridsetlogging, FHYPRE_STRUCTHYBRIDSETLOGGING)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructHybridSetPrintLevel \
        hypre_F90_NAME(fhypre_structhybridsetprintleve, FHYPRE_STRUCTHYBRIDSETPRINTLEVE)
extern void hypre_F90_NAME(fhypre_structhybridsetprintleve, FHYPRE_STRUCTHYBRIDSETPRINTLEVE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructHybridGetNumIterations \
        hypre_F90_NAME(fhypre_structhybridgetnumiterat, FHYPRE_STRUCTHYBRIDGETNUMITERAT)
extern void hypre_F90_NAME(fhypre_structhybridgetnumiterat, FHYPRE_STRUCTHYBRIDGETNUMITERAT)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructHybridGetDSCGNumIterations \
        hypre_F90_NAME(fhypre_structhybridgetdscgnumit, FHYPRE_STRUCTHYBRIDGETDSCGNUMIT)
extern void hypre_F90_NAME(fhypre_structhybridgetdscgnumit, FHYPRE_STRUCTHYBRIDGETDSCGNUMIT)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructHybridGetPCGNumIterations \
        hypre_F90_NAME(fhypre_structhybridgetpcgnumite, FHYPRE_STRUCTHYBRIDGETPCGNUMITE)
extern void hypre_F90_NAME(fhypre_structhybridgetpcgnumite, FHYPRE_STRUCTHYBRIDGETPCGNUMITE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructHybridGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structhybridgetfinalrela, FHYPRE_STRUCTHYBRIDGETFINALRELA)
extern void hypre_F90_NAME(fhypre_structhybridgetfinalrela, FHYPRE_STRUCTHYBRIDGETFINALRELA)
                          (hypre_F90_Obj *, double *);



#define HYPRE_StructVectorSetRandomValues \
        hypre_F90_NAME(fhypre_structvectorsetrandomvalu, FHYPRE_STRUCTVECTORSETRANDOMVALU)
extern void hypre_F90_NAME(fhypre_structvectorsetrandomvalu, FHYPRE_STRUCTVECTORSETRANDOMVALU)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSetRandomValues \
        hypre_F90_NAME(fhypre_structsetrandomvalues, FHYPRE_STRUCTSETRANDOMVALUES)
extern void hypre_F90_NAME(fhypre_structsetrandomvalues, FHYPRE_STRUCTSETRANDOMVALUES)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSetupInterpreter \
        hypre_F90_NAME(fhypre_structsetupinterpreter, FHYPRE_STRUCTSETUPINTERPRETER)
extern void hypre_F90_NAME(fhypre_structsetupinterpreter, FHYPRE_STRUCTSETUPINTERPRETER)
                          (hypre_F90_Obj *);

#define HYPRE_StructSetupMatvec \
        hypre_F90_NAME(fhypre_structsetupmatvec, FHYPRE_STRUCTSETUPMATVEC)
extern void hypre_F90_NAME(fhypre_structsetupmatvec, FHYPRE_STRUCTSETUPMATVEC)
                          (hypre_F90_Obj *);



#define HYPRE_StructJacobiCreate \
        hypre_F90_NAME(fhypre_structjacobicreate, FHYPRE_STRUCTJACOBICREATE)
extern void hypre_F90_NAME(fhypre_structjacobicreate, FHYPRE_STRUCTJACOBICREATE)
                          (HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_StructJacobiDestroy \
        hypre_F90_NAME(fhypre_structjacobidestroy, FHYPRE_STRUCTJACOBIDESTROY)
extern void hypre_F90_NAME(fhypre_structjacobidestroy, FHYPRE_STRUCTJACOBIDESTROY)
                          (hypre_F90_Obj *);

#define HYPRE_StructJacobiSetup \
        hypre_F90_NAME(fhypre_structjacobisetup, FHYPRE_STRUCTJACOBISETUP)
extern void hypre_F90_NAME(fhypre_structjacobisetup, FHYPRE_STRUCTJACOBISETUP)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructJacobiSolve \
        hypre_F90_NAME(fhypre_structjacobisolve, FHYPRE_STRUCTJACOBISOLVE)
extern void hypre_F90_NAME(fhypre_structjacobisolve, FHYPRE_STRUCTJACOBISOLVE)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructJacobiSetTol \
        hypre_F90_NAME(fhypre_structjacobisettol, FHYPRE_STRUCTJACOBISETTOL)
extern void hypre_F90_NAME(fhypre_structjacobisettol, FHYPRE_STRUCTJACOBISETTOL)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructJacobiGetTol \
        hypre_F90_NAME(fhypre_structjacobigettol, FHYPRE_STRUCTJACOBIGETTOL)
extern void hypre_F90_NAME(fhypre_structjacobigettol, FHYPRE_STRUCTJACOBIGETTOL)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructJacobiSetMaxIter \
        hypre_F90_NAME(fhypre_structjacobisetmaxiter, FHYPRE_STRUCTJACOBISETTOL)
extern void hypre_F90_NAME(fhypre_structjacobisetmaxiter, FHYPRE_STRUCTJACOBISETTOL)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructJacobiGetMaxIter \
        hypre_F90_NAME(fhypre_structjacobigetmaxiter, FHYPRE_STRUCTJACOBIGETTOL)
extern void hypre_F90_NAME(fhypre_structjacobigetmaxiter, FHYPRE_STRUCTJACOBIGETTOL)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructJacobiSetZeroGuess \
        hypre_F90_NAME(fhypre_structjacobisetzeroguess, FHYPRE_STRUCTJACOBISETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structjacobisetzeroguess, FHYPRE_STRUCTJACOBISETZEROGUESS)
                          (hypre_F90_Obj *);

#define HYPRE_StructJacobiGetZeroGuess \
        hypre_F90_NAME(fhypre_structjacobigetzeroguess, FHYPRE_STRUCTJACOBIGETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structjacobigetzeroguess, FHYPRE_STRUCTJACOBIGETZEROGUESS)
                          (hypre_F90_Obj *);

#define HYPRE_StructJacobiSetNonZeroGuess \
        hypre_F90_NAME(fhypre_structjacobisetnonzerogu, FHYPRE_STRUCTJACOBISETNONZEROGU)
extern void hypre_F90_NAME(fhypre_structjacobisetnonzerogu, FHYPRE_STRUCTJACOBISETNONZEROGU)
                          (hypre_F90_Obj *);

#define HYPRE_StructJacobiGetNumIterations \
        hypre_F90_NAME(fhypre_structjacobigetnumiterati, FHYPRE_STRUCTJACOBIGETNUMITERATI)
extern void hypre_F90_NAME(fhypre_structjacobigetnumiterati, FHYPRE_STRUCTJACOBIGETNUMITERATI)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructJacobiGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structjacobigetfinalrela, FHYPRE_STRUCTJACOBIGETFINALRELA)
extern void hypre_F90_NAME(fhypre_structjacobigetfinalrela, FHYPRE_STRUCTJACOBIGETFINALRELA)
                          (hypre_F90_Obj *, double *);



#define HYPRE_StructPCGCreate \
        hypre_F90_NAME(fhypre_structpcgcreate, FHYPRE_STRUCTPCGCREATE)
extern void hypre_F90_NAME(fhypre_structpcgcreate, FHYPRE_STRUCTPCGCREATE)
                          (HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_StructPCGDestroy \
        hypre_F90_NAME(fhypre_structpcgdestroy, FHYPRE_STRUCTPCGDESTROY)
extern void hypre_F90_NAME(fhypre_structpcgdestroy, FHYPRE_STRUCTPCGDESTROY)
                          (hypre_F90_Obj *);

#define HYPRE_StructPCGSetup \
        hypre_F90_NAME(fhypre_structpcgsetup, FHYPRE_STRUCTPCGSETUP)
extern void hypre_F90_NAME(fhypre_structpcgsetup, FHYPRE_STRUCTPCGSETUP) 
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructPCGSolve \
        hypre_F90_NAME(fhypre_structpcgsolve, FHYPRE_STRUCTPCGSOLVE)
extern void hypre_F90_NAME(fhypre_structpcgsolve, FHYPRE_STRUCTPCGSOLVE)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructPCGSetTol \
        hypre_F90_NAME(fhypre_structpcgsettol, FHYPRE_STRUCTPCGSETTOL)
extern void hypre_F90_NAME(fhypre_structpcgsettol, FHYPRE_STRUCTPCGSETTOL)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructPCGSetMaxIter \
        hypre_F90_NAME(fhypre_structpcgsetmaxiter, FHYPRE_STRUCTPCGSETMAXITER)
extern void hypre_F90_NAME(fhypre_structpcgsetmaxiter, FHYPRE_STRUCTPCGSETMAXITER)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPCGSetTwoNorm \
        hypre_F90_NAME(fhypre_structpcgsettwonorm, FHYRPE_STRUCTPCGSETTWONORM)
extern void hypre_F90_NAME(fhypre_structpcgsettwonorm, FHYRPE_STRUCTPCGSETTWONORM)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPCGSetRelChange \
        hypre_F90_NAME(fhypre_structpcgsetrelchange, FHYPRE_STRUCTPCGSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structpcgsetrelchange, FHYPRE_STRUCTPCGSETRELCHANGE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPCGSetPrecond \
        hypre_F90_NAME(fhypre_structpcgsetprecond, FHYPRE_STRUCTPCGSETPRECOND)
extern void hypre_F90_NAME(fhypre_structpcgsetprecond, FHYPRE_STRUCTPCGSETPRECOND)
                          (hypre_F90_Obj *, HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_StructPCGSetLogging \
        hypre_F90_NAME(fhypre_structpcgsetlogging, FHYPRES_TRUCTPCFSETLOGGING)
extern void hypre_F90_NAME(fhypre_structpcgsetlogging, FHYPRES_TRUCTPCFSETLOGGING)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPCGSetPrintLevel \
        hypre_F90_NAME(fhypre_structpcgsetprintlevel, FHYPRE_STRUCTPCGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structpcgsetprintlevel, FHYPRE_STRUCTPCGSETPRINTLEVEL)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPCGGetNumIterations \
        hypre_F90_NAME(fhypre_structpcggetnumiteration, FHYPRE_STRUCTPCGGETNUMITERATION)
extern void hypre_F90_NAME(fhypre_structpcggetnumiteration, FHYPRE_STRUCTPCGGETNUMITERATION)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPCGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structpcggetfinalrelativ, FHYPRE_STRUCTPCGGETFINALRELATIV)
extern void hypre_F90_NAME(fhypre_structpcggetfinalrelativ, FHYPRE_STRUCTPCGGETFINALRELATIV)
                          (hypre_F90_Obj *, double *);



#define HYPRE_StructDiagScaleSetup \
        hypre_F90_NAME(fhypre_structdiagscalesetup, FHYPRE_STRUCTDIAGSCALESETUP)
extern void hypre_F90_NAME(fhypre_structdiagscalesetup, FHYPRE_STRUCTDIAGSCALESETUP)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructDiagScaleSolve \
        hypre_F90_NAME(fhypre_structdiagscalesolve, FHYPRE_STRUCTDIAGSCALESOLVE)
extern void hypre_F90_NAME(fhypre_structdiagscalesolve, FHYPRE_STRUCTDIAGSCALESOLVE)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);



#define HYPRE_StructPFMGCreate \
        hypre_F90_NAME(fhypre_structpfmgcreate, FHYPRE_STRUCTPFMGCREATE)
extern void hypre_F90_NAME(fhypre_structpfmgcreate, FHYPRE_STRUCTPFMGCREATE)
                          (HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_StructPFMGDestroy \
        hypre_F90_NAME(fhypre_structpfmgdestroy, FHYPRE_STRUCTPFMGDESTROY)
extern void hypre_F90_NAME(fhypre_structpfmgdestroy, FHYPRE_STRUCTPFMGDESTROY)
                          (hypre_F90_Obj *);

#define HYPRE_StructPFMGSetup \
        hypre_F90_NAME(fhypre_structpfmgsetup, FHYPRE_STRUCTPFMGSETUP)
extern void hypre_F90_NAME(fhypre_structpfmgsetup, FHYPRE_STRUCTPFMGSETUP)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructPFMGSolve \
        hypre_F90_NAME(fhypre_structpfmgsolve, FHYPRE_STRUCTPFMGSOLVE)
extern void hypre_F90_NAME(fhypre_structpfmgsolve, FHYPRE_STRUCTPFMGSOLVE)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructPFMGSetTol \
        hypre_F90_NAME(fhypre_structpfmgsettol, FHYPRE_STRUCTPFMGSETTOL)
extern void hypre_F90_NAME(fhypre_structpfmgsettol, FHYPRE_STRUCTPFMGSETTOL)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructPFMGGetTol \
        hypre_F90_NAME(fhypre_structpfmggettol, FHYPRE_STRUCTPFMGGETTOL)
extern void hypre_F90_NAME(fhypre_structpfmggettol, FHYPRE_STRUCTPFMGGETTOL)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructPFMGSetMaxIter \
        hypre_F90_NAME(fhypre_structpfmgsetmaxiter, FHYPRE_STRUCTPFMGSETMAXITER)
extern void hypre_F90_NAME(fhypre_structpfmgsetmaxiter, FHYPRE_STRUCTPFMGSETMAXITER)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGGetMaxIter \
        hypre_F90_NAME(fhypre_structpfmggetmaxiter, FHYPRE_STRUCTPFMGGETMAXITER)
extern void hypre_F90_NAME(fhypre_structpfmggetmaxiter, FHYPRE_STRUCTPFMGGETMAXITER)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGSetMaxLevels \
        hypre_F90_NAME(fhypre_structpfmgsetmaxlevels, FHYPRE_STRUCTPFMGSETMAXLEVELS)
extern void hypre_F90_NAME(fhypre_structpfmgsetmaxlevels, FHYPRE_STRUCTPFMGSETMAXLEVELS)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGGetMaxLevels \
        hypre_F90_NAME(fhypre_structpfmggetmaxlevels, FHYPRE_STRUCTPFMGGETMAXLEVELS)
extern void hypre_F90_NAME(fhypre_structpfmggetmaxlevels, FHYPRE_STRUCTPFMGGETMAXLEVELS)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGSetRelChange \
        hypre_F90_NAME(fhypre_structpfmgsetrelchange, FHYPRE_STRUCTPFMGSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structpfmgsetrelchange, FHYPRE_STRUCTPFMGSETRELCHANGE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGGetRelChange \
        hypre_F90_NAME(fhypre_structpfmggetrelchange, FHYPRE_STRUCTPFMGGETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structpfmggetrelchange, FHYPRE_STRUCTPFMGGETRELCHANGE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGSetZeroGuess \
        hypre_F90_NAME(fhypre_structpfmgsetzeroguess, FHYPRE_STRUCTPFMGSETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structpfmgsetzeroguess, FHYPRE_STRUCTPFMGSETZEROGUESS)
                          (hypre_F90_Obj *);

#define HYPRE_StructPFMGGetZeroGuess \
        hypre_F90_NAME(fhypre_structpfmggetzeroguess, FHYPRE_STRUCTPFMGGETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structpfmggetzeroguess, FHYPRE_STRUCTPFMGGETZEROGUESS)
                          (hypre_F90_Obj *);

#define HYPRE_StructPFMGSetNonZeroGuess \
        hypre_F90_NAME(fhypre_structpfmgsetnonzerogues, FHYPRES_TRUCTPFMGSETNONZEROGUES)
extern void hypre_F90_NAME(fhypre_structpfmgsetnonzerogues, FHYPRES_TRUCTPFMGSETNONZEROGUES)
                          (hypre_F90_Obj *);

#define HYPRE_StructPFMGSetSkipRelax \
        hypre_F90_NAME(fhypre_structpfmgsetskiprelax, FHYPRE_STRUCTPFMGSETSKIPRELAX)
extern void hypre_F90_NAME(fhypre_structpfmgsetskiprelax, FHYPRE_STRUCTPFMGSETSKIPRELAX)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGGetSkipRelax \
        hypre_F90_NAME(fhypre_structpfmggetskiprelax, FHYPRE_STRUCTPFMGGETSKIPRELAX)
extern void hypre_F90_NAME(fhypre_structpfmggetskiprelax, FHYPRE_STRUCTPFMGGETSKIPRELAX)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGSetRelaxType \
        hypre_F90_NAME(fhypre_structpfmgsetrelaxtype, FHYPRE_STRUCTPFMGSETRELAXTYPE)
extern void hypre_F90_NAME(fhypre_structpfmgsetrelaxtype, FHYPRE_STRUCTPFMGSETRELAXTYPE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGGetRelaxType \
        hypre_F90_NAME(fhypre_structpfmggetrelaxtype, FHYPRE_STRUCTPFMGGETRELAXTYPE)
extern void hypre_F90_NAME(fhypre_structpfmggetrelaxtype, FHYPRE_STRUCTPFMGGETRELAXTYPE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGSetRAPType \
        hypre_F90_NAME(fhypre_structpfmgsetraptype, FHYPRE_STRUCTPFMGSETRAPTYPE)
extern void hypre_F90_NAME(fhypre_structpfmgsetraptype, FHYPRE_STRUCTPFMGSETRAPTYPE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGGetRAPType \
        hypre_F90_NAME(fhypre_structpfmggetraptype, FHYPRE_STRUCTPFMGGETRAPTYPE)
extern void hypre_F90_NAME(fhypre_structpfmggetraptype, FHYPRE_STRUCTPFMGGETRAPTYPE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGSetNumPreRelax \
        hypre_F90_NAME(fhypre_structpfmgsetnumprerelax, FHYPRE_STRUCTPFMGSETNUMPRERELAX)
extern void hypre_F90_NAME(fhypre_structpfmgsetnumprerelax, FHYPRE_STRUCTPFMGSETNUMPRERELAX)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGGetNumPreRelax \
        hypre_F90_NAME(fhypre_structpfmggetnumprerelax, FHYPRE_STRUCTPFMGGETNUMPRERELAX)
extern void hypre_F90_NAME(fhypre_structpfmggetnumprerelax, FHYPRE_STRUCTPFMGGETNUMPRERELAX)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGSetNumPostRelax \
        hypre_F90_NAME(fhypre_structpfmgsetnumpostrela, FHYPRE_STRUCTPFMGSETNUMPOSTRELA)
extern void hypre_F90_NAME(fhypre_structpfmgsetnumpostrela, FHYPRE_STRUCTPFMGSETNUMPOSTRELA)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGGetNumPostRelax \
        hypre_F90_NAME(fhypre_structpfmggetnumpostrela, FHYPRE_STRUCTPFMGGETNUMPOSTRELA)
extern void hypre_F90_NAME(fhypre_structpfmggetnumpostrela, FHYPRE_STRUCTPFMGGETNUMPOSTRELA)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGSetDxyz \
        hypre_F90_NAME(fhypre_structpfmgsetdxyz, FHYPRE_STRUCTPFMGSETDXYZ)
extern void hypre_F90_NAME(fhypre_structpfmgsetdxyz, FHYPRE_STRUCTPFMGSETDXYZ)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructPFMGSetLogging \
        hypre_F90_NAME(fhypre_structpfmgsetlogging, FHYPRE_STRUCTPFMGSETLOGGING)
extern void hypre_F90_NAME(fhypre_structpfmgsetlogging, FHYPRE_STRUCTPFMGSETLOGGING)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGGetLogging \
        hypre_F90_NAME(fhypre_structpfmggetlogging, FHYPRE_STRUCTPFMGGETLOGGING)
extern void hypre_F90_NAME(fhypre_structpfmggetlogging, FHYPRE_STRUCTPFMGGETLOGGING)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGSetPrintLevel \
        hypre_F90_NAME(fhypre_structpfmgsetprintlevel, FHYPRE_STRUCTPFMGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structpfmgsetprintlevel, FHYPRE_STRUCTPFMGSETPRINTLEVEL)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGGetPrintLevel \
        hypre_F90_NAME(fhypre_structpfmggetprintlevel, FHYPRE_STRUCTPFMGGETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structpfmggetprintlevel, FHYPRE_STRUCTPFMGGETPRINTLEVEL)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGGetNumIterations \
        hypre_F90_NAME(fhypre_structpfmggetnumiteratio, FHYPRE_STRUCTPFMGGETNUMITERATIO)
extern void hypre_F90_NAME(fhypre_structpfmggetnumiteratio, FHYPRE_STRUCTPFMGGETNUMITERATIO)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructPFMGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structpfmggetfinalrelati, FHYPRE_STRUCTPFMGGETFINALRELATI)
extern void hypre_F90_NAME(fhypre_structpfmggetfinalrelati, FHYPRE_STRUCTPFMGGETFINALRELATI)
                          (hypre_F90_Obj *, double *);



#define HYPRE_StructSMGCreate \
        hypre_F90_NAME(fhypre_structsmgcreate, FHYPRE_STRUCTSMGCREATE)
extern void hypre_F90_NAME(fhypre_structsmgcreate, FHYPRE_STRUCTSMGCREATE)
                          (HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_StructSMGDestroy \
        hypre_F90_NAME(fhypre_structsmgdestroy, FHYPRE_STRUCTSMGDESTROY)
extern void hypre_F90_NAME(fhypre_structsmgdestroy, FHYPRE_STRUCTSMGDESTROY)
                          (hypre_F90_Obj *);

#define HYPRE_StructSMGSetup \
        hypre_F90_NAME(fhypre_structsmgsetup, FHYPRE_STRUCTSMGSETUP)
extern void hypre_F90_NAME(fhypre_structsmgsetup, FHYPRE_STRUCTSMGSETUP)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructSMGSolve \
        hypre_F90_NAME(fhypre_structsmgsolve, FHYPRE_STRUCTSMGSOLVE)
extern void hypre_F90_NAME(fhypre_structsmgsolve, FHYPRE_STRUCTSMGSOLVE)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructSMGSetMemoryUse \
        hypre_F90_NAME(fhypre_structsmgsetmemoryuse, FHYPRE_STRUCTSMGSETMEMORYUSE)
extern void hypre_F90_NAME(fhypre_structsmgsetmemoryuse, FHYPRE_STRUCTSMGSETMEMORYUSE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSMGGetMemoryUse \
        hypre_F90_NAME(fhypre_structsmggetmemoryuse, FHYPRE_STRUCTSMGGETMEMORYUSE)
extern void hypre_F90_NAME(fhypre_structsmggetmemoryuse, FHYPRE_STRUCTSMGGETMEMORYUSE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSMGSetTol \
        hypre_F90_NAME(fhypre_structsmgsettol, FHYPRE_STRUCTSMGSETTOL)
extern void hypre_F90_NAME(fhypre_structsmgsettol, FHYPRE_STRUCTSMGSETTOL)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructSMGGetTol \
        hypre_F90_NAME(fhypre_structsmggettol, FHYPRE_STRUCTSMGGETTOL)
extern void hypre_F90_NAME(fhypre_structsmggettol, FHYPRE_STRUCTSMGGETTOL)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructSMGSetMaxIter \
        hypre_F90_NAME(fhypre_structsmgsetmaxiter, FHYPRE_STRUCTSMGSETMAXTITER)
extern void hypre_F90_NAME(fhypre_structsmgsetmaxiter, FHYPRE_STRUCTSMGSETMAXTITER)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSMGGetMaxIter \
        hypre_F90_NAME(fhypre_structsmggetmaxiter, FHYPRE_STRUCTSMGGETMAXTITER)
extern void hypre_F90_NAME(fhypre_structsmggetmaxiter, FHYPRE_STRUCTSMGGETMAXTITER)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSMGSetRelChange \
        hypre_F90_NAME(fhypre_structsmgsetrelchange, FHYPRE_STRUCTSMGSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structsmgsetrelchange, FHYPRE_STRUCTSMGSETRELCHANGE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSMGGetRelChange \
        hypre_F90_NAME(fhypre_structsmggetrelchange, FHYPRE_STRUCTSMGGETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structsmggetrelchange, FHYPRE_STRUCTSMGGETRELCHANGE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSMGSetZeroGuess \
        hypre_F90_NAME(fhypre_structsmgsetzeroguess, FHYPRE_STRUCTSMGSETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structsmgsetzeroguess, FHYPRE_STRUCTSMGSETZEROGUESS)
                          (hypre_F90_Obj *);

#define HYPRE_StructSMGGetZeroGuess \
        hypre_F90_NAME(fhypre_structsmggetzeroguess, FHYPRE_STRUCTSMGGETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structsmggetzeroguess, FHYPRE_STRUCTSMGGETZEROGUESS)
                          (hypre_F90_Obj *);

#define HYPRE_StructSMGSetNonZeroGuess \
        hypre_F90_NAME(fhypre_structsmgsetnonzerogues, FHYPRE_STRUCTSMGSETNONZEROGUES)
extern void hypre_F90_NAME(fhypre_structsmgsetnonzerogues, FHYPRE_STRUCTSMGSETNONZEROGUES)
                          (hypre_F90_Obj *);

#define HYPRE_StructSMGGetNumIterations \
        hypre_F90_NAME(fhypre_structsmggetnumiteration, FHYPRE_STRUCTSMGGETNUMITERATION)
extern void hypre_F90_NAME(fhypre_structsmggetnumiteration, FHYPRE_STRUCTSMGGETNUMITERATION)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSMGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structsmggetfinalrelativ, FHYPRE_STRUCTSMGGETFINALRELATIV)
extern void hypre_F90_NAME(fhypre_structsmggetfinalrelativ, FHYPRE_STRUCTSMGGETFINALRELATIV)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructSMGSetNumPreRelax \
        hypre_F90_NAME(fhypre_structsmgsetnumprerelax, FHYPRE_STRUCTSMGSETNUMPRERELAX)
extern void hypre_F90_NAME(fhypre_structsmgsetnumprerelax, FHYPRE_STRUCTSMGSETNUMPRERELAX)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSMGGetNumPreRelax \
        hypre_F90_NAME(fhypre_structsmggetnumprerelax, FHYPRE_STRUCTSMGGETNUMPRERELAX)
extern void hypre_F90_NAME(fhypre_structsmggetnumprerelax, FHYPRE_STRUCTSMGGETNUMPRERELAX)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSMGSetNumPostRelax \
        hypre_F90_NAME(fhypre_structsmgsetnumpostrelax, FHYPRE_STRUCTSMGSETNUMPOSTRELAX)
extern void hypre_F90_NAME(fhypre_structsmgsetnumpostrelax, FHYPRE_STRUCTSMGSETNUMPOSTRELAX)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSMGGetNumPostRelax \
        hypre_F90_NAME(fhypre_structsmggetnumpostrelax, FHYPRE_STRUCTSMGGETNUMPOSTRELAX)
extern void hypre_F90_NAME(fhypre_structsmggetnumpostrelax, FHYPRE_STRUCTSMGGETNUMPOSTRELAX)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSMGSetLogging \
        hypre_F90_NAME(fhypre_structsmgsetlogging, FHYPRE_STRUCTSMGSETLOGGING)
extern void hypre_F90_NAME(fhypre_structsmgsetlogging, FHYPRE_STRUCTSMGSETLOGGING)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSMGGetLogging \
        hypre_F90_NAME(fhypre_structsmggetlogging, FHYPRE_STRUCTSMGGETLOGGING)
extern void hypre_F90_NAME(fhypre_structsmggetlogging, FHYPRE_STRUCTSMGGETLOGGING)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSMGSetPrintLevel \
        hypre_F90_NAME(fhypre_structsmgsetprintlevel, FHYPRE_STRUCTSMGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structsmgsetprintlevel, FHYPRE_STRUCTSMGSETPRINTLEVEL)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSMGGetPrintLevel \
        hypre_F90_NAME(fhypre_structsmggetprintlevel, FHYPRE_STRUCTSMGGETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structsmggetprintlevel, FHYPRE_STRUCTSMGGETPRINTLEVEL)
                          (hypre_F90_Obj *, HYPRE_Int *);



#define HYPRE_StructSparseMSGCreate \
        hypre_F90_NAME(fhypre_structsparsemsgcreate, FHYPRE_STRUCTSPARSEMSGCREATE)
extern void hypre_F90_NAME(fhypre_structsparsemsgcreate, FHYPRE_STRUCTSPARSEMSGCREATE)
                          (HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_StructSparseMSGDestroy \
        hypre_F90_NAME(fhypre_structsparsemsgdestroy, FHYPRE_STRUCTSPARSEMSGDESTROY)
extern void hypre_F90_NAME(fhypre_structsparsemsgdestroy, FHYPRE_STRUCTSPARSEMSGDESTROY)
                          (hypre_F90_Obj *);

#define HYPRE_StructSparseMSGSetup \
        hypre_F90_NAME(fhypre_structsparsemsgsetup, FHYRPE_STRUCTSPARSEMSGSETUP)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetup, FHYRPE_STRUCTSPARSEMSGSETUP)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructSparseMSGSolve \
        hypre_F90_NAME(fhypre_structsparsemsgsolve, FHYPRE_STRUCTSPARSEMSGSOLVE)
extern void hypre_F90_NAME(fhypre_structsparsemsgsolve, FHYPRE_STRUCTSPARSEMSGSOLVE)
                          (hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_StructSparseMSGSetJump \
        hypre_F90_NAME(fhypre_structsparsemsgsetjump, FHYPRE_STRUCTSPARSEMSGSETJUMP)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetjump, FHYPRE_STRUCTSPARSEMSGSETJUMP)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSparseMSGSetTol \
        hypre_F90_NAME(fhypre_structsparsemsgsettol, FHYPRE_STRUCTSPARSEMSGSETTOL)
extern void hypre_F90_NAME(fhypre_structsparsemsgsettol, FHYPRE_STRUCTSPARSEMSGSETTOL)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructSparseMSGSetMaxIter \
        hypre_F90_NAME(fhypre_structsparsemsgsetmaxite, FHYPRE_STRUCTSPARSEMSGSETMAXITE)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetmaxite, FHYPRE_STRUCTSPARSEMSGSETMAXITE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSparseMSGSetRelChange \
        hypre_F90_NAME(fhypre_structsparsemsgsetrelcha, FHYPRE_STRUCTSPARSEMSGSETRELCHA)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetrelcha, FHYPRE_STRUCTSPARSEMSGSETRELCHA)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSparseMSGSetZeroGuess \
        hypre_F90_NAME(fhypre_structsparsemsgsetzerogu, FHYPRE_STRUCTSPARSEMSGSETZEROGU)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetzerogu, FHYPRE_STRUCTSPARSEMSGSETZEROGU)
                          (hypre_F90_Obj *);

#define HYPRE_StructSparseMSGSetNonZeroGuess \
        hypre_F90_NAME(fhypre_structsparsemsgsetnonzer, FHYPRE_STRUCTSPARSEMSGSETNONZER)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetnonzer, FHYPRE_STRUCTSPARSEMSGSETNONZER)
                          (hypre_F90_Obj *);

#define HYPRE_StructSparseMSGGetNumIterations \
        hypre_F90_NAME(fhypre_structsparsemsggetnumite, FHYPRE_STRUCTSPARSEMSGGETNUMITE)
extern void hypre_F90_NAME(fhypre_structsparsemsggetnumite, FHYPRE_STRUCTSPARSEMSGGETNUMITE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSparseMSGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structsparsemsggetfinalr, FHYRPE_STRUCTSPARSEMSGGETFINALR)
extern void hypre_F90_NAME(fhypre_structsparsemsggetfinalr, FHYRPE_STRUCTSPARSEMSGGETFINALR)
                          (hypre_F90_Obj *, double *);

#define HYPRE_StructSparseMSGSetRelaxType \
        hypre_F90_NAME(fhypre_structsparsemsgsetrelaxt, FHYPRE_STRUCTSPARSEMSGSETRELAXT)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetrelaxt, FHYPRE_STRUCTSPARSEMSGSETRELAXT)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSparseMSGSetNumPreRelax \
        hypre_F90_NAME(fhypre_structsparsemsgsetnumpre, FHYPRE_STRUCTSPARSEMSGSETNUMPRE)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetnumpre, FHYPRE_STRUCTSPARSEMSGSETNUMPRE)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSparseMSGSetNumPostRelax \
        hypre_F90_NAME(fhypre_structsparsemsgsetnumpos, FHYPRE_STRUCTSPARSEMSGSETNUMPOS)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetnumpos, FHYPRE_STRUCTSPARSEMSGSETNUMPOS)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSparseMSGSetNumFineRelax \
        hypre_F90_NAME(fhypre_structsparsemsgsetnumfin, FHYPRE_STRUCTSPARSEMSGSETNUMFIN)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetnumfin, FHYPRE_STRUCTSPARSEMSGSETNUMFIN)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSparseMSGSetLogging \
        hypre_F90_NAME(fhypre_structsparsemsgsetloggin, FHYPRE_STRUCTSPARSEMSGSETLOGGIN)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetloggin, FHYPRE_STRUCTSPARSEMSGSETLOGGIN)
                          (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_StructSparseMSGSetPrintLevel \
        hypre_F90_NAME(fhypre_structsparsemsgsetprintl, FHYPRE_STRUCTSPARSEMSGSETPRINTL)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetprintl, FHYPRE_STRUCTSPARSEMSGSETPRINTL)
                          (hypre_F90_Obj *, HYPRE_Int *);
