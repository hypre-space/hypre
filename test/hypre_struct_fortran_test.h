/*BHEADER**********************************************************************
 * Copyright (c) 2007,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/**************************************************
*  Definitions of struct fortran interface routines
**************************************************/

#define HYPRE_StructStencilCreate \
        hypre_F90_NAME(fhypre_structstencilcreate, FHYPRE_STRUCTSTENCILCREATE)
extern void hypre_F90_NAME(fhypre_structstencilcreate, FHYPRE_STRUCTSTENCILCREATE)
                          (int *, int *, long int *);

#define HYPRE_StructStencilDestroy \
        hypre_F90_NAME(fhypre_structstencildestroy, FHYPRE_STRUCTSTENCILDESTROY)
extern void hypre_F90_NAME(fhypre_structstencildestroy, FHYPRE_STRUCTSTENCILDESTROY)
                          (long int *);

#define HYPRE_StructStencilSetElement \
        hypre_F90_NAME(fhypre_structstencilsetelement, FHYPRE_STRUCTSTENCILSETELEMENT)
extern void hypre_F90_NAME(fhypre_structstencilsetelement, FHYPRE_STRUCTSTENCILSETELEMENT)
                          (long int *, int *, int *);



#define HYPRE_StructGridCreate \
        hypre_F90_NAME(fhypre_structgridcreate, FHYPRE_STRUCTGRIDCREATE)
extern void hypre_F90_NAME(fhypre_structgridcreate, FHYPRE_STRUCTGRIDCREATE)
                          (int *, int *, long int *);

#define HYPRE_StructGridDestroy \
        hypre_F90_NAME(fhypre_structgriddestroy, FHYPRE_STRUCTGRIDDESTROY)
extern void hypre_F90_NAME(fhypre_structgriddestroy, FHYPRE_STRUCTGRIDDESTROY)
                          (long int *);

#define HYPRE_StructGridSetExtents \
        hypre_F90_NAME(fhypre_structgridsetextents, FHYPRE_STRUCTGRIDSETEXTENTS)
extern void hypre_F90_NAME(fhypre_structgridsetextents, FHYPRE_STRUCTGRIDSETEXTENTS)
                          (long int *, int *, int *);

#define HYPRE_StructGridSetPeriodic \
        hypre_F90_NAME(fhypre_structgridsetperiodic, FHYPRE_STRUCTGRIDSETPERIODIC)
extern void hypre_F90_NAME(fhypre_structgridsetperiodic, fhypre_structsetgridperiodic)
                          (long int *, int *);

#define HYPRE_StructGridAssemble \
        hypre_F90_NAME(fhypre_structgridassemble, FHYPRE_STRUCTGRIDASSEMBLE)
extern void hypre_F90_NAME(fhypre_structgridassemble, FHYPRE_STRUCTGRIDASSEMBLE)
                          (long int *);

#define HYPRE_StructGridSetNumGhost \
        hypre_F90_NAME(fhypre_structgridsetnumghost, FHYPRE_STRUCTGRIDSETNUMGHOST)
extern void hypre_F90_NAME(fhypre_structgridsetnumghost, fhypre_structsetgridnumghost)
                          (long int *, int *);
        


#define HYPRE_StructMatrixCreate \
        hypre_F90_NAME(fhypre_structmatrixcreate, FHYPRE_STRUCTMATRIXCREATE)
extern void hypre_F90_NAME(fhypre_structmatrixcreate, FHYPRE_STRUCTMATRIXCREATE)
                          (int *, long int *, long int *, long int *);

#define HYPRE_StructMatrixDestroy \
        hypre_F90_NAME(fhypre_structmatrixdestroy, FHYPRE_STRUCTMATRIXDESTROY)
extern void hypre_F90_NAME(fhypre_structmatrixdestroy, FHYPRE_STRUCTMATRIXDESTROY)
                          (long int *);

#define HYPRE_StructMatrixInitialize \
        hypre_F90_NAME(fhypre_structmatrixinitialize, FHYPRE_STRUCTMATRIXINITIALIZE)
extern void hypre_F90_NAME(fhypre_structmatrixinitialize, FHYPRE_STRUCTMATRIXINITIALIZE)
                          (long int *);

#define HYPRE_StructMatrixSetValues \
        hypre_F90_NAME(fhypre_structmatrixsetvalues, FHYPRE_STRUCTMATRIXSETVALUES)
extern void hypre_F90_NAME(fhypre_structmatrixsetvalues, FHYPRE_STRUCTMATRIXSETVALUES)
                          (long int *, int *, int *);

#define HYPRE_StructMatrixSetBoxValues \
        hypre_F90_NAME(fhypre_structmatrixsetboxvalues, FHYPRE_STRUCTMATRIXSETBOXVALUES)
extern void hypre_F90_NAME(fhypre_structmatrixsetboxvalues, FHYPRE_STRUCTMATRIXSETBOXVALUES)
                          (long int *, int *, int *, int *, int *, double *);

#define HYPRE_StructMatrixGetBoxValues \
        hypre_F90_NAME(fhypre_structmatrixgetboxvalues, FHYPRE_STRUCTMATRIXGETBOXVALUES)
extern void hypre_F90_NAME(fhypre_structmatrixgetboxvalues, FHYPRE_STRUCTMATRIXGETBOXVALUES)
                          (long int *, int *, int *, int *, int *, double *);

#define HYPRE_StructMatrixSetConstantEntries \
        hypre_F90_NAME(fhypre_structmatrixsetconstante, FHYPRE_STRUCTMATRIXSETCONSTANTE)
extern void hypre_F90_NAME(fhypre_structmatrixsetconstante, FHYPRE_STRUCTMATRIXSETCONSTANTE)
                          (long int *, int *, int *);

#define HYPRE_StructMatrixSetConstantValues \
        hypre_F90_NAME(fhypre_structmatrixsetconstantv, FHYPRE_STRUCTMATRIXSETCONSTANTV)
extern void hypre_F90_NAME(fhypre_structmatrixsetconstantv, FHYPRE_STRUCTMATRIXSETCONSTANTV)
                          (long int *, int *, int *, double *);

#define HYPRE_StructMatrixAddToValues \
        hypre_F90_NAME(fhypre_structmatrixaddtovalues, FHYPRE_STRUCTMATRIXADDTOVALUES)
extern void hypre_F90_NAME(fhypre_structmatrixaddtovalues, FHYPRE_STRUCTMATRIXADDTOVALUES)
                          (long int *, int *, int *, int *, double *);

#define HYPRE_StructMatrixAddToBoxValues \
        hypre_F90_NAME(fhypre_structmatrixaddtoboxvalues, FHYPRE_STRUCTMATRIXADDTOBOXVALUES)
extern void hypre_F90_NAME(fhypre_structmatrixaddtoboxvalues, FHYPRE_STRUCTMATRIXADDTOBOXVALUES)
                          (long int *, int *, int *, int *, int *, double *);

#define HYPRE_StructMatrixAddToConstantValues \
        hypre_F90_NAME(fhypre_structmatrixaddtoconstant, FHYPRE_STRUCTMATRIXADDTOCONSTANT)
extern void hypre_F90_NAME(fhypre_structmatrixaddtoconstant, FHYPRE_STRUCTMATRIXADDTOCONSTANT)
                          (long int *, int *, int *, double *);

#define HYPRE_StructMatrixAssemble \
        hypre_F90_NAME(fhypre_structmatrixassemble, FHYPRE_STRUCTMATRIXASSEMBLE)
extern void hypre_F90_NAME(fhypre_structmatrixassemble, FHYPRE_STRUCTMATRIXASSEMBLE)
                          (long int *);

#define HYPRE_StructMatrixSetNumGhost \
        hypre_F90_NAME(fhypre_structmatrixsetnumghost, FHYPRE_STRUCTMATRIXSETNUMGHOST)
extern void hypre_F90_NAME(fhypre_structmatrixsetnumghost, FHYPRE_STRUCTMATRIXSETNUMGHOST)
                          (long int *, int *);

#define HYPRE_StructMatrixGetGrid \
        hypre_F90_NAME(fhypre_structmatrixgetgrid, FHYPRE_STRUCTMATRIXGETGRID)
extern void hypre_F90_NAME(fhypre_structmatrixgetgrid, FHYPRE_STRUCTMATRIXGETGRID)
                          (long int *, long int *);

#define HYPRE_StructMatrixSetSymmetric \
        hypre_F90_NAME(fhypre_structmatrixsetsymmetric, FHYPRE_STRUCTMATRIXSETSYMMETRIC)
extern void hypre_F90_NAME(fhypre_structmatrixsetsymmetric, FHYPRE_STRUCTMATRIXSETSYMMETRIC)
                          (long int *, int *);

#define HYPRE_StructMatrixPrint \
hypre_F90_NAME(fhypre_structmatrixprint, FHYPRE_STRUCTMATRIXPRINT)
extern void hypre_F90_NAME(fhypre_structmatrixprint, FHYPRE_STRUCTMATRIXPRINT)
                          (long int *, int *);

#define HYPRE_StructMatrixMatvec \
hypre_F90_NAME(fhypre_structmatrixmatvec, FHYPRE_STRUCTMATRIXMATVEC)
extern void hypre_F90_NAME(fhypre_structmatrixmatvec, FHYPRE_STRUCTMATRIXMATVEC)
                          (int *, long int *, long int *, int *, long int *);
 


#define HYPRE_StructVectorCreate \
        hypre_F90_NAME(fhypre_structvectorcreate, FHYPRE_STRUCTVECTORCREATE)
extern void hypre_F90_NAME(fhypre_structvectorcreate, FHYPRE_STRUCTVECTORCREATE)
                          (int *, long int *, long int *);

#define HYPRE_StructVectorDestroy \
        hypre_F90_NAME(fhypre_structvectordestroy, FHYPRE_STRUCTVECTORDESTROY)
extern void hypre_F90_NAME(fhypre_structvectordestroy, FHYPRE_STRUCTVECTORDESTROY) 
                          (long int *);

#define HYPRE_StructVectorInitialize \
        hypre_F90_NAME(fhypre_structvectorinitialize, FHYPRE_STRUCTVECTORINITIALIZE)
extern void hypre_F90_NAME(fhypre_structvectorinitialize, FHYPRE_STRUCTVECTORINITIALIZE)
                          (long int *);

#define HYPRE_StructVectorSetValues \
        hypre_F90_NAME(fhypre_structvectorsetvalues, FHYPRE_STRUCTVECTORSETVALUES)
extern void hypre_F90_NAME(fhypre_structvectorsetvalues, FHYPRE_STRUCTVECTORSETVALUES)
                          (long int *, int *, int *);

#define HYPRE_StructVectorSetBoxValues \
        hypre_F90_NAME(fhypre_structvectorsetboxvalues, FHYPRE_STRUCTVECTORSETBOXVALUES)
extern void hypre_F90_NAME(fhypre_structvectorsetboxvalues, FHYPRE_STRUCTVECTORSETBOXVALUES)
                          (long int *, int *, int *, double *);

#define HYPRE_StructVectorSetConstantValues \
        hypre_F90_NAME(fhypre_structvectorsetconstantv, FHYPRE_STRUCTVECTORSETCONTANTV)
extern void hypre_F90_NAME(fhypre_structvectorsetconstantv, FHYPRE_STRUCTVECTORSETCONTANTV)
                          (long int *, double *);

#define HYPRE_StructVectorAddToValues \
        hypre_F90_NAME(fhypre_structvectoraddtovalues, FHYPRE_STRUCTVECTORADDTOVALUES)
extern void hypre_F90_NAME(fhypre_structvectoraddtovalues, FHYPRE_STRUCTVECTORADDTOVALUES)
                          (long int *, int *, double *);

#define HYPRE_StructVectorAddToBoxValues \
        hypre_F90_NAME(fhypre_structvectoraddtoboxvalu, FHYPRE_STRUCTVECTORADDTOBOXVALU)
extern void hypre_F90_NAME(fhypre_structvectoraddtoboxvalu, FHYPRE_STRUCTVECTORADDTOBOXVALU)
                          (long int *, int *, int *, double *);

#define HYPRE_StructVectorScaleValues \
        hypre_F90_NAME(fhypre_structvectorscalevalues, FHYPRE_STRUCTVECTORSCALEVALUES)
extern void hypre_F90_NAME(fhypre_structvectorscalevalues, FHYPRE_STRUCTVECTORSCALEVALUES)
                          (long int *, double *);

#define HYPRE_StructVectorGetValues \
        hypre_F90_NAME(fhypre_structvectorgetvalues, FHYPRE_STRUCTVECTORGETVALUES)
extern void hypre_F90_NAME(fhypre_structvectorgetvalues, FHYPRE_STRUCTVECTORGETVALUES)
                          (long int *, int *, double *);

#define HYPRE_StructVectorGetBoxValues \
        hypre_F90_NAME(fhypre_structvectorgetboxvalues, FHYPRE_STRUCTVECTORGETBOXVALUES)
extern void hypre_F90_NAME(fhypre_structvectorgetboxvalues, FHYPRE_STRUCTVECTORGETBOXVALUES)
                          (long int *, int *, double *);

#define HYPRE_StructVectorAssemble \
        hypre_F90_NAME(fhypre_structvectorassemble, FHYPRE_STRUCTVECTORASSEMBLE)
extern void hypre_F90_NAME(fhypre_structvectorassemble, FHYPRE_STRUCTVECTORASSEMBLE)
                          (long int *);

#define HYPRE_StructVectorSetNumGhost \
        hypre_F90_NAME(fhypre_structvectorsetnumghost, FHYPRE_STRUCTVECTORSETNUMGHOST)
extern void hypre_F90_NAME(fhypre_structvectorsetnumghost, FHYPRE_STRUCTVECTORSETNUMGHOST)
                          (long int *, int *);

#define HYPRE_StructVectorCopy \
        hypre_F90_NAME(fhypre_structvectorcopy, FHYPRE_STRUCTVECTORCOPY)
extern void hypre_F90_NAME(fhypre_structvectorcopy, FHYPRE_STRUCTVECTORCOPY)
                          (long int *, long int *);

#define HYPRE_StructVectorGetMigrateCommPkg \
        hypre_F90_NAME(fhypre_structvectorgetmigrateco, FHYPRE_STRUCTVECTORGETMIGRATECO)
extern void hypre_F90_NAME(fhypre_structvectorgetmigrateco, FHYPRE_STRUCTVECTORGETMIGRATECO)
                          (long int *, long int *, long int *);

#define HYPRE_StructVectorMigrate \
        hypre_F90_NAME(fhypre_structvectormigrate, FHYPRE_STRUCTVECTORMIGRATE)
extern void hypre_F90_NAME(fhypre_structvectormigrate, FHYPRE_STRUCTVECTORMIGRATE)
                          (long int *, long int *, long int *);

#define HYPRE_CommPkgDestroy \
        hypre_F90_NAME(fhypre_commpkgdestroy, FHYPRE_COMMPKGDESTROY)
extern void hypre_F90_NAME(fhypre_commpkgdestroy, FHYPRE_COMMPKGDESTROY)
                          (long int *);

#define HYPRE_StructVectorPrint \
        hypre_F90_NAME(fhypre_structvectorprint, FHYPRE_STRUCTVECTORPRINT)
extern void hypre_F90_NAME(fhypre_structvectorprint, FHYPRE_STRUCTVECTORPRINT)
                          (long int *, int *);
 

#define HYPRE_StructBiCGSTABCreate \
        hypre_F90_NAME(fhypre_structbicgstabcreate, FHYPRE_STRUCTBICGSTABCREATE)
extern void hypre_F90_NAME(fhypre_structbicgstabcreate, FHYPRE_STRUCTBICGSTABCREATE)
                          (int *, long int *);

#define HYPRE_StructBiCGSTABDestroy \
        hypre_F90_NAME(fhypre_structbicgstabdestroy, FHYPRE_STRUCTBICGSTABDESTROY)
extern void hypre_F90_NAME(fhypre_structbicgstabdestroy, FHYPRE_STRUCTBICGSTABDESTROY)
                          (long int *);

#define HYPRE_StructBiCGSTABSetup \
        hypre_F90_NAME(fhypre_structbicgstabsetup, FHYPRE_STRUCTBICGSTABSETUP)
extern void hypre_F90_NAME(fhypre_structbicgstabsetup, FHYPRE_STRUCTBICGSTABSETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_StructBiCGSTABSolve \
        hypre_F90_NAME(fhypre_structbicgstabsolve, FHYPRE_STRUCTBICGSTABSOLVE)
extern void hypre_F90_NAME(fhypre_structbicgstabsolve, FHYPRE_STRUCTBICGSTABSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_StructBiCGSTABSetTol \
        hypre_F90_NAME(fhypre_structbicgstabsettol, FHYPRE_STRUCTBICGSTABSETTOL)
extern void hypre_F90_NAME(fhypre_structbicgstabsettol, FHYPRE_STRUCTBICGSTABSETTOL)
                          (long int *, double *);

#define HYPRE_StructBiCGSTABSetMaxIter \
        hypre_F90_NAME(fhypre_structbicgstabsetmaxiter, FHYPRE_STRUCTBICGSTABSETMAXITER)
extern void hypre_F90_NAME(fhypre_structbicgstabsetmaxiter, FHYPRE_STRUCTBICGSTABSETMAXITER)
                          (long int *, int *);

#define HYPRE_StructBiCGSTABSetPrecond \
        hypre_F90_NAME(fhypre_structbicgstabsetprecond, FHYPRE_STRUCTBICGSTABSETPRECOND)
extern void hypre_F90_NAME(fhypre_structbicgstabsetprecond, FHYPRE_STRUCTBICGSTABSETPRECOND)
                          (long int *, int *, long int *);

#define HYPRE_StructBiCGSTABSetLogging \
        hypre_F90_NAME(fhypre_structbicgstabsetlogging, FHYPRE_STRUCTBICGSTABSETLOGGING)
extern void hypre_F90_NAME(fhypre_structbicgstabsetlogging, FHYPRE_STRUCTBICGSTABSETLOGGING)
                          (long int *, int *);

#define HYPRE_StructBiCGSTABSetPrintLevel \
        hypre_F90_NAME(fhypre_structbicgstabsetprintle, FHYPRE_STRUCTBICGSTABPRINTLE)
extern void hypre_F90_NAME(fhypre_structbicgstabsetprintle, FHYPRE_STRUCTBICGSTABPRINTLE)
                          (long int *, int *);

#define HYPRE_StructBiCGSTABGetNumIterations \
        hypre_F90_NAME(fhypre_structbicgstabgetnumiter, FHYPRE_STRUCTBICGSTABGETNUMITER)
extern void hypre_F90_NAME(fhypre_structbicgstabgetnumiter, FHYPRE_STRUCTBICGSTABGETNUMITER)
                          (long int *, int *);

#define HYPRE_StructBiCGSTABGetResidual \
        hypre_F90_NAME(fhypre_structbicgstabgetresidua, FHYPRE_STRUCTBICGSTABGETRESIDUA)
extern void hypre_F90_NAME(fhypre_structbicgstabgetresidua, FHYPRE_STRUCTBICGSTABGETRESIDUA)
                          (long int *, double *);

#define HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structbicgstabgetfinalre, FHYPRE_STRUCTBICGSTABGETFINALRE)
extern void hypre_F90_NAME(fhypre_structbicgstabgetfinalre, FHYPRE_STRUCTBICGSTABGETFINALRE)
                          (long int *, double *);



#define HYPRE_StructGMRESCreate \
        hypre_F90_NAME(fhypre_structgmrescreate, FHYPRE_STRUCTGMRESCREATE)
extern void hypre_F90_NAME(fhypre_structgmrescreate, FHYPRE_STRUCTGMRESCREATE)
                          (int *, long int *);

#define HYPRE_StructGMRESDestroy \
        hypre_F90_NAME(fhypre_structgmresdestroy, FHYPRE_STRUCTGMRESDESTROY)
extern void hypre_F90_NAME(fhypre_structgmresdestroy, FHYPRE_STRUCTGMRESDESTROY)
                          (long int *);

#define HYPRE_StructGMRESSetup \
        hypre_F90_NAME(fhypre_structgmressetup, FHYPRE_STRUCTGMRESSETUP)
extern void hypre_F90_NAME(fhypre_structgmressetup, FHYPRE_STRUCTGMRESSETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_StructGMRESSolve \
        hypre_F90_NAME(fhypre_structgmressolve, FHYPRE_STRUCTGMRESSOLVE)
extern void hypre_F90_NAME(fhypre_structgmressolve, FHYPRE_STRUCTGMRESSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_StructGMRESSetTol \
        hypre_F90_NAME(fhypre_structgmressettol, FHYPRE_STRUCTGMRESSETTOL)
extern void hypre_F90_NAME(fhypre_structgmressettol, FHYPRE_STRUCTGMRESSETTOL)
                          (long int *, double *);

#define HYPRE_StructGMRESSetMaxIter \
        hypre_F90_NAME(fhypre_structgmressetmaxiter, FHYPRE_STRUCTGMRESSETMAXITER)
extern void hypre_F90_NAME(fhypre_structgmressetmaxiter, FHYPRE_STRUCTGMRESSETMAXITER)
                          (long int *, int *);

#define HYPRE_StructGMRESSetPrecond \
        hypre_F90_NAME(fhypre_structgmressetprecond, FHYPRE_STRUCTGMRESSETPRECOND)
extern void hypre_F90_NAME(fhypre_structgmressetprecond, FHYPRE_STRUCTGMRESSETPRECOND)
                          (long int *, int *, long int *);

#define HYPRE_StructGMRESSetLogging \
        hypre_F90_NAME(fhypre_structgmressetlogging, FHYPRE_STRUCTGMRESSETLOGGING)
extern void hypre_F90_NAME(fhypre_structgmressetlogging, FHYPRE_STRUCTGMRESSETLOGGING)
                          (long int *, int *);

#define HYPRE_StructGMRESSetPrintLevel \
        hypre_F90_NAME(fhypre_structgmressetprintlevel, FHYPRE_STRUCTGMRESPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structgmressetprintlevel, FHYPRE_STRUCTGMRESPRINTLEVEL)
                          (long int *, int *);

#define HYPRE_StructGMRESGetNumIterations \
        hypre_F90_NAME(fhypre_structgmresgetnumiterati, FHYPRE_STRUCTGMRESGETNUMITERATI)
extern void hypre_F90_NAME(fhypre_structgmresgetnumiterati, FHYPRE_STRUCTGMRESGETNUMITERATI)
                          (long int *, int *);

#define HYPRE_StructGMRESGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structgmresgetfinalrelat, FHYPRE_STRUCTGMRESGETFINALRELAT)
extern void hypre_F90_NAME(fhypre_structgmresgetfinalrelat, FHYPRE_STRUCTGMRESGETFINALRELAT)
                          (long int *, double *);



#define HYPRE_StructHybridCreate \
        hypre_F90_NAME(fhypre_structhybridcreate, FHYPRE_STRUCTHYBRIDCREATE)
extern void hypre_F90_NAME(fhypre_structhybridcreate, FHYPRE_STRUCTHYBRIDCREATE)
                          (int *, long int *);

#define HYPRE_StructHybridDestroy \
        hypre_F90_NAME(fhypre_structhybriddestroy, FHYPRE_STRUCTHYBRIDDESTROY)
extern void hypre_F90_NAME(fhypre_structhybriddestroy, FHYPRE_STRUCTHYBRIDDESTROY)
                          (long int *);

#define HYPRE_StructHybridSetup \
        hypre_F90_NAME(fhypre_structhybridsetup, FHYPRE_STRUCTHYBRIDSETUP)
extern void hypre_F90_NAME(fhypre_structhybridsetup, FHYPRE_STRUCTHYBRIDSETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_StructHybridSolve \
        hypre_F90_NAME(fhypre_structhybridsolve, FHYPRE_STRUCTHYBRIDSOLVE)
extern void hypre_F90_NAME(fhypre_structhybridsolve, FHYPRE_STRUCTHYBRIDSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_StructHybridSetSolverType \
        hypre_F90_NAME(fhypre_structhybridsetsolvertyp, FHYPRE_STRUCTHYBRIDSETSOLVERTYP)
extern void hypre_F90_NAME(fhypre_structhybridsetsolvertyp, FHYPRE_STRUCTHYBRIDSETSOLVERTYP)
                          (long int *, int *);

#define HYPRE_StructHybridSetStopCrit \
        hypre_F90_NAME(fhypre_structhybridsetstopcrit, FHYPRE_STRUCTHYBRIDSETSTOPCRIT)
extern void hypre_F90_NAME(fhypre_structhybridsetstopcrit, FHYPRE_STRUCTHYBRIDSETSTOPCRIT)
                          (long int *, int *);

#define HYPRE_StructHybridSetKDim \
        hypre_F90_NAME(fhypre_structhybridsetkdim, FHYPRE_STRUCTHYBRIDSETKDIM)
extern void hypre_F90_NAME(fhypre_structhybridsetkdim, FHYPRE_STRUCTHYBRIDSETKDIM)
                          (long int *, int *);

#define HYPRE_StructHybridSetTol \
        hypre_F90_NAME(fhypre_structhybridsettol, FHYPRE_STRUCTHYBRIDSETTOL)
extern void hypre_F90_NAME(fhypre_structhybridsettol, FHYPRE_STRUCTHYBRIDSETTOL)
                          (long int *, double *);

#define HYPRE_StructHybridSetConvergenceTol \
        hypre_F90_NAME(fhypre_structhybridsetconvergen, FHYPRE_STRUCTHYBRIDSETCONVERGEN)
extern void hypre_F90_NAME(fhypre_structhybridsetconvergen, FHYPRE_STRUCTHYBRIDSETCONVERGEN)
                          (long int *, double *);

#define HYPRE_StructHybridSetPCGAbsoluteTolFactor \
        hypre_F90_NAME(fhypre_structhybridsetpcgabsolu, FHYPRE_STRUCTHYBRIDSETABSOLU)
extern void hypre_F90_NAME(fhypre_structhybridsetpcgabsolu, FHYPRE_STRUCTHYBRIDSETABSOLU)
                          (long int *, double *);

#define HYPRE_StructHybridSetMaxIter \
        hypre_F90_NAME(fhypre_structhybridsetmaxiter, FHYPRE_STRUCTHYBRIDSETMAXITER)
extern void hypre_F90_NAME(fhypre_structhybridsetmaxiter, FHYPRE_STRUCTHYBRIDSETMAXITER)
                          (long int *, int *);

#define HYPRE_StructHybridSetDSCGMaxIter \
        hypre_F90_NAME(fhypre_structhybridsetdscgmaxit, FHYPRE_STRUCTHYBRIDSETDSCGMAXIT)
extern void hypre_F90_NAME(fhypre_structhybridsetdscgmaxit, FHYPRE_STRUCTHYBRIDSETDSCGMAXIT)
                          (long int *, int *);

#define HYPRE_StructHybridSetPCGMaxIter \
        hypre_F90_NAME(fhypre_structhybridsetpcgmaxite, FHYPRE_STRUCTHYBRIDSETPCGMAXITE)
extern void hypre_F90_NAME(fhypre_structhybridsetpcgmaxite, FHYPRE_STRUCTHYBRIDSETPCGMAXITE)
                          (long int *, int *);

#define HYPRE_StructHybridSetTwoNorm \
        hypre_F90_NAME(fhypre_structhybridsettwonorm, FHYPRE_STRUCTHYBRIDSETTWONORM)
extern void hypre_F90_NAME(fhypre_structhybridsettwonorm, FHYPRE_STRUCTHYBRIDSETTWONORM)
                          (long int *, int *);

#define HYPRE_StructHybridSetRelChange \
        hypre_F90_NAME(fhypre_structhybridsetrelchange, FHYPRE_STRUCTHYBRIDSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structhybridsetrelchange, FHYPRE_STRUCTHYBRIDSETRELCHANGE)
                          (long int *, int *);

#define HYPRE_StructHybridSetPrecond \
        hypre_F90_NAME(fhypre_structhybridsetprecond, FHYPRE_STRUCTHYBRIDSETPRECOND)
extern void hypre_F90_NAME(fhypre_structhybridsetprecond, FHYPRE_STRUCTHYBRIDSETPRECOND) 
                          (long int *, int *, long int *);

#define HYPRE_StructHybridSetLogging \
        hypre_F90_NAME(fhypre_structhybridsetlogging, FHYPRE_STRUCTHYBRIDSETLOGGING)
extern void hypre_F90_NAME(fhypre_structhybridsetlogging, FHYPRE_STRUCTHYBRIDSETLOGGING)
                          (long int *, int *);

#define HYPRE_StructHybridSetPrintLevel \
        hypre_F90_NAME(fhypre_structhybridsetprintleve, FHYPRE_STRUCTHYBRIDSETPRINTLEVE)
extern void hypre_F90_NAME(fhypre_structhybridsetprintleve, FHYPRE_STRUCTHYBRIDSETPRINTLEVE)
                          (long int *, int *);

#define HYPRE_StructHybridGetNumIterations \
        hypre_F90_NAME(fhypre_structhybridgetnumiterat, FHYPRE_STRUCTHYBRIDGETNUMITERAT)
extern void hypre_F90_NAME(fhypre_structhybridgetnumiterat, FHYPRE_STRUCTHYBRIDGETNUMITERAT)
                          (long int *, int *);

#define HYPRE_StructHybridGetDSCGNumIterations \
        hypre_F90_NAME(fhypre_structhybridgetdscgnumit, FHYPRE_STRUCTHYBRIDGETDSCGNUMIT)
extern void hypre_F90_NAME(fhypre_structhybridgetdscgnumit, FHYPRE_STRUCTHYBRIDGETDSCGNUMIT)
                          (long int *, int *);

#define HYPRE_StructHybridGetPCGNumIterations \
        hypre_F90_NAME(fhypre_structhybridgetpcgnumite, FHYPRE_STRUCTHYBRIDGETPCGNUMITE)
extern void hypre_F90_NAME(fhypre_structhybridgetpcgnumite, FHYPRE_STRUCTHYBRIDGETPCGNUMITE)
                          (long int *, int *);

#define HYPRE_StructHybridGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structhybridgetfinalrela, FHYPRE_STRUCTHYBRIDGETFINALRELA)
extern void hypre_F90_NAME(fhypre_structhybridgetfinalrela, FHYPRE_STRUCTHYBRIDGETFINALRELA)
                          (long int *, double *);



#define HYPRE_StructVectorSetRandomValues \
        hypre_F90_NAME(fhypre_structvectorsetrandomvalu, FHYPRE_STRUCTVECTORSETRANDOMVALU)
extern void hypre_F90_NAME(fhypre_structvectorsetrandomvalu, FHYPRE_STRUCTVECTORSETRANDOMVALU)
                          (long int *, int *);

#define HYPRE_StructSetRandomValues \
        hypre_F90_NAME(fhypre_structsetrandomvalues, FHYPRE_STRUCTSETRANDOMVALUES)
extern void hypre_F90_NAME(fhypre_structsetrandomvalues, FHYPRE_STRUCTSETRANDOMVALUES)
                          (long int *, int *);

#define HYPRE_StructSetupInterpreter \
        hypre_F90_NAME(fhypre_structsetupinterpreter, FHYPRE_STRUCTSETUPINTERPRETER)
extern void hypre_F90_NAME(fhypre_structsetupinterpreter, FHYPRE_STRUCTSETUPINTERPRETER)
                          (long int *);

#define HYPRE_StructSetupMatvec \
        hypre_F90_NAME(fhypre_structsetupmatvec, FHYPRE_STRUCTSETUPMATVEC)
extern void hypre_F90_NAME(fhypre_structsetupmatvec, FHYPRE_STRUCTSETUPMATVEC)
                          (long int *);



#define HYPRE_StructJacobiCreate \
        hypre_F90_NAME(fhypre_structjacobicreate, FHYPRE_STRUCTJACOBICREATE)
extern void hypre_F90_NAME(fhypre_structjacobicreate, FHYPRE_STRUCTJACOBICREATE)
                          (int *, long int *);

#define HYPRE_StructJacobiDestroy \
        hypre_F90_NAME(fhypre_structjacobidestroy, FHYPRE_STRUCTJACOBIDESTROY)
extern void hypre_F90_NAME(fhypre_structjacobidestroy, FHYPRE_STRUCTJACOBIDESTROY)
                          (long int *);

#define HYPRE_StructJacobiSetup \
        hypre_F90_NAME(fhypre_structjacobisetup, FHYPRE_STRUCTJACOBISETUP)
extern void hypre_F90_NAME(fhypre_structjacobisetup, FHYPRE_STRUCTJACOBISETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_StructJacobiSolve \
        hypre_F90_NAME(fhypre_structjacobisolve, FHYPRE_STRUCTJACOBISOLVE)
extern void hypre_F90_NAME(fhypre_structjacobisolve, FHYPRE_STRUCTJACOBISOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_StructJacobiSetTol \
        hypre_F90_NAME(fhypre_structjacobisettol, FHYPRE_STRUCTJACOBISETTOL)
extern void hypre_F90_NAME(fhypre_structjacobisettol, FHYPRE_STRUCTJACOBISETTOL)
                          (long int *, double *);

#define HYPRE_StructJacobiGetTol \
        hypre_F90_NAME(fhypre_structjacobigettol, FHYPRE_STRUCTJACOBIGETTOL)
extern void hypre_F90_NAME(fhypre_structjacobigettol, FHYPRE_STRUCTJACOBIGETTOL)
                          (long int *, double *);

#define HYPRE_StructJacobiSetMaxIter \
        hypre_F90_NAME(fhypre_structjacobisetmaxiter, FHYPRE_STRUCTJACOBISETTOL)
extern void hypre_F90_NAME(fhypre_structjacobisetmaxiter, FHYPRE_STRUCTJACOBISETTOL)
                          (long int *, int *);

#define HYPRE_StructJacobiGetMaxIter \
        hypre_F90_NAME(fhypre_structjacobigetmaxiter, FHYPRE_STRUCTJACOBIGETTOL)
extern void hypre_F90_NAME(fhypre_structjacobigetmaxiter, FHYPRE_STRUCTJACOBIGETTOL)
                          (long int *, int *);

#define HYPRE_StructJacobiSetZeroGuess \
        hypre_F90_NAME(fhypre_structjacobisetzeroguess, FHYPRE_STRUCTJACOBISETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structjacobisetzeroguess, FHYPRE_STRUCTJACOBISETZEROGUESS)
                          (long int *);

#define HYPRE_StructJacobiGetZeroGuess \
        hypre_F90_NAME(fhypre_structjacobigetzeroguess, FHYPRE_STRUCTJACOBIGETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structjacobigetzeroguess, FHYPRE_STRUCTJACOBIGETZEROGUESS)
                          (long int *);

#define HYPRE_StructJacobiSetNonZeroGuess \
        hypre_F90_NAME(fhypre_structjacobisetnonzerogu, FHYPRE_STRUCTJACOBISETNONZEROGU)
extern void hypre_F90_NAME(fhypre_structjacobisetnonzerogu, FHYPRE_STRUCTJACOBISETNONZEROGU)
                          (long int *);

#define HYPRE_StructJacobiGetNumIterations \
        hypre_F90_NAME(fhypre_structjacobigetnumiterati, FHYPRE_STRUCTJACOBIGETNUMITERATI)
extern void hypre_F90_NAME(fhypre_structjacobigetnumiterati, FHYPRE_STRUCTJACOBIGETNUMITERATI)
                          (long int *, int *);

#define HYPRE_StructJacobiGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structjacobigetfinalrela, FHYPRE_STRUCTJACOBIGETFINALRELA)
extern void hypre_F90_NAME(fhypre_structjacobigetfinalrela, FHYPRE_STRUCTJACOBIGETFINALRELA)
                          (long int *, double *);



#define HYPRE_StructPCGCreate \
        hypre_F90_NAME(fhypre_structpcgcreate, FHYPRE_STRUCTPCGCREATE)
extern void hypre_F90_NAME(fhypre_structpcgcreate, FHYPRE_STRUCTPCGCREATE)
                          (int *, long int *);

#define HYPRE_StructPCGDestroy \
        hypre_F90_NAME(fhypre_structpcgdestroy, FHYPRE_STRUCTPCGDESTROY)
extern void hypre_F90_NAME(fhypre_structpcgdestroy, FHYPRE_STRUCTPCGDESTROY)
                          (long int *);

#define HYPRE_StructPCGSetup \
        hypre_F90_NAME(fhypre_structpcgsetup, FHYPRE_STRUCTPCGSETUP)
extern void hypre_F90_NAME(fhypre_structpcgsetup, FHYPRE_STRUCTPCGSETUP) 
                          (long int *, long int *, long int *, long int *);

#define HYPRE_StructPCGSolve \
        hypre_F90_NAME(fhypre_structpcgsolve, FHYPRE_STRUCTPCGSOLVE)
extern void hypre_F90_NAME(fhypre_structpcgsolve, FHYPRE_STRUCTPCGSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_StructPCGSetTol \
        hypre_F90_NAME(fhypre_structpcgsettol, FHYPRE_STRUCTPCGSETTOL)
extern void hypre_F90_NAME(fhypre_structpcgsettol, FHYPRE_STRUCTPCGSETTOL)
                          (long int *, double *);

#define HYPRE_StructPCGSetMaxIter \
        hypre_F90_NAME(fhypre_structpcgsetmaxiter, FHYPRE_STRUCTPCGSETMAXITER)
extern void hypre_F90_NAME(fhypre_structpcgsetmaxiter, FHYPRE_STRUCTPCGSETMAXITER)
                          (long int *, int *);

#define HYPRE_StructPCGSetTwoNorm \
        hypre_F90_NAME(fhypre_structpcgsettwonorm, FHYRPE_STRUCTPCGSETTWONORM)
extern void hypre_F90_NAME(fhypre_structpcgsettwonorm, FHYRPE_STRUCTPCGSETTWONORM)
                          (long int *, int *);

#define HYPRE_StructPCGSetRelChange \
        hypre_F90_NAME(fhypre_structpcgsetrelchange, FHYPRE_STRUCTPCGSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structpcgsetrelchange, FHYPRE_STRUCTPCGSETRELCHANGE)
                          (long int *, int *);

#define HYPRE_StructPCGSetPrecond \
        hypre_F90_NAME(fhypre_structpcgsetprecond, FHYPRE_STRUCTPCGSETPRECOND)
extern void hypre_F90_NAME(fhypre_structpcgsetprecond, FHYPRE_STRUCTPCGSETPRECOND)
                          (long int *, int *, long int *);

#define HYPRE_StructPCGSetLogging \
        hypre_F90_NAME(fhypre_structpcgsetlogging, FHYPRES_TRUCTPCFSETLOGGING)
extern void hypre_F90_NAME(fhypre_structpcgsetlogging, FHYPRES_TRUCTPCFSETLOGGING)
                          (long int *, int *);

#define HYPRE_StructPCGSetPrintLevel \
        hypre_F90_NAME(fhypre_structpcgsetprintlevel, FHYPRE_STRUCTPCGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structpcgsetprintlevel, FHYPRE_STRUCTPCGSETPRINTLEVEL)
                          (long int *, int *);

#define HYPRE_StructPCGGetNumIterations \
        hypre_F90_NAME(fhypre_structpcggetnumiteration, FHYPRE_STRUCTPCGGETNUMITERATION)
extern void hypre_F90_NAME(fhypre_structpcggetnumiteration, FHYPRE_STRUCTPCGGETNUMITERATION)
                          (long int *, int *);

#define HYPRE_StructPCGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structpcggetfinalrelativ, FHYPRE_STRUCTPCGGETFINALRELATIV)
extern void hypre_F90_NAME(fhypre_structpcggetfinalrelativ, FHYPRE_STRUCTPCGGETFINALRELATIV)
                          (long int *, double *);



#define HYPRE_StructDiagScaleSetup \
        hypre_F90_NAME(fhypre_structdiagscalesetup, FHYPRE_STRUCTDIAGSCALESETUP)
extern void hypre_F90_NAME(fhypre_structdiagscalesetup, FHYPRE_STRUCTDIAGSCALESETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_StructDiagScaleSolve \
        hypre_F90_NAME(fhypre_structdiagscalesolve, FHYPRE_STRUCTDIAGSCALESOLVE)
extern void hypre_F90_NAME(fhypre_structdiagscalesolve, FHYPRE_STRUCTDIAGSCALESOLVE)
                          (long int *, long int *, long int *, long int *);



#define HYPRE_StructPFMGCreate \
        hypre_F90_NAME(fhypre_structpfmgcreate, FHYPRE_STRUCTPFMGCREATE)
extern void hypre_F90_NAME(fhypre_structpfmgcreate, FHYPRE_STRUCTPFMGCREATE)
                          (int *, long int *);

#define HYPRE_StructPFMGDestroy \
        hypre_F90_NAME(fhypre_structpfmgdestroy, FHYPRE_STRUCTPFMGDESTROY)
extern void hypre_F90_NAME(fhypre_structpfmgdestroy, FHYPRE_STRUCTPFMGDESTROY)
                          (long int *);

#define HYPRE_StructPFMGSetup \
        hypre_F90_NAME(fhypre_structpfmgsetup, FHYPRE_STRUCTPFMGSETUP)
extern void hypre_F90_NAME(fhypre_structpfmgsetup, FHYPRE_STRUCTPFMGSETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_StructPFMGSolve \
        hypre_F90_NAME(fhypre_structpfmgsolve, FHYPRE_STRUCTPFMGSOLVE)
extern void hypre_F90_NAME(fhypre_structpfmgsolve, FHYPRE_STRUCTPFMGSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_StructPFMGSetTol \
        hypre_F90_NAME(fhypre_structpfmgsettol, FHYPRE_STRUCTPFMGSETTOL)
extern void hypre_F90_NAME(fhypre_structpfmgsettol, FHYPRE_STRUCTPFMGSETTOL)
                          (long int *, double *);

#define HYPRE_StructPFMGGetTol \
        hypre_F90_NAME(fhypre_structpfmggettol, FHYPRE_STRUCTPFMGGETTOL)
extern void hypre_F90_NAME(fhypre_structpfmggettol, FHYPRE_STRUCTPFMGGETTOL)
                          (long int *, double *);

#define HYPRE_StructPFMGSetMaxIter \
        hypre_F90_NAME(fhypre_structpfmgsetmaxiter, FHYPRE_STRUCTPFMGSETMAXITER)
extern void hypre_F90_NAME(fhypre_structpfmgsetmaxiter, FHYPRE_STRUCTPFMGSETMAXITER)
                          (long int *, int *);

#define HYPRE_StructPFMGGetMaxIter \
        hypre_F90_NAME(fhypre_structpfmggetmaxiter, FHYPRE_STRUCTPFMGGETMAXITER)
extern void hypre_F90_NAME(fhypre_structpfmggetmaxiter, FHYPRE_STRUCTPFMGGETMAXITER)
                          (long int *, int *);

#define HYPRE_StructPFMGSetMaxLevels \
        hypre_F90_NAME(fhypre_structpfmgsetmaxlevels, FHYPRE_STRUCTPFMGSETMAXLEVELS)
extern void hypre_F90_NAME(fhypre_structpfmgsetmaxlevels, FHYPRE_STRUCTPFMGSETMAXLEVELS)
                          (long int *, int *);

#define HYPRE_StructPFMGGetMaxLevels \
        hypre_F90_NAME(fhypre_structpfmggetmaxlevels, FHYPRE_STRUCTPFMGGETMAXLEVELS)
extern void hypre_F90_NAME(fhypre_structpfmggetmaxlevels, FHYPRE_STRUCTPFMGGETMAXLEVELS)
                          (long int *, int *);

#define HYPRE_StructPFMGSetRelChange \
        hypre_F90_NAME(fhypre_structpfmgsetrelchange, FHYPRE_STRUCTPFMGSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structpfmgsetrelchange, FHYPRE_STRUCTPFMGSETRELCHANGE)
                          (long int *, int *);

#define HYPRE_StructPFMGGetRelChange \
        hypre_F90_NAME(fhypre_structpfmggetrelchange, FHYPRE_STRUCTPFMGGETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structpfmggetrelchange, FHYPRE_STRUCTPFMGGETRELCHANGE)
                          (long int *, int *);

#define HYPRE_StructPFMGSetZeroGuess \
        hypre_F90_NAME(fhypre_structpfmgsetzeroguess, FHYPRE_STRUCTPFMGSETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structpfmgsetzeroguess, FHYPRE_STRUCTPFMGSETZEROGUESS)
                          (long int *);

#define HYPRE_StructPFMGGetZeroGuess \
        hypre_F90_NAME(fhypre_structpfmggetzeroguess, FHYPRE_STRUCTPFMGGETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structpfmggetzeroguess, FHYPRE_STRUCTPFMGGETZEROGUESS)
                          (long int *);

#define HYPRE_StructPFMGSetNonZeroGuess \
        hypre_F90_NAME(fhypre_structpfmgsetnonzerogues, FHYPRES_TRUCTPFMGSETNONZEROGUES)
extern void hypre_F90_NAME(fhypre_structpfmgsetnonzerogues, FHYPRES_TRUCTPFMGSETNONZEROGUES)
                          (long int *);

#define HYPRE_StructPFMGSetSkipRelax \
        hypre_F90_NAME(fhypre_structpfmgsetskiprelax, FHYPRE_STRUCTPFMGSETSKIPRELAX)
extern void hypre_F90_NAME(fhypre_structpfmgsetskiprelax, FHYPRE_STRUCTPFMGSETSKIPRELAX)
                          (long int *, int *);

#define HYPRE_StructPFMGGetSkipRelax \
        hypre_F90_NAME(fhypre_structpfmggetskiprelax, FHYPRE_STRUCTPFMGGETSKIPRELAX)
extern void hypre_F90_NAME(fhypre_structpfmggetskiprelax, FHYPRE_STRUCTPFMGGETSKIPRELAX)
                          (long int *, int *);

#define HYPRE_StructPFMGSetRelaxType \
        hypre_F90_NAME(fhypre_structpfmgsetrelaxtype, FHYPRE_STRUCTPFMGSETRELAXTYPE)
extern void hypre_F90_NAME(fhypre_structpfmgsetrelaxtype, FHYPRE_STRUCTPFMGSETRELAXTYPE)
                          (long int *, int *);

#define HYPRE_StructPFMGGetRelaxType \
        hypre_F90_NAME(fhypre_structpfmggetrelaxtype, FHYPRE_STRUCTPFMGGETRELAXTYPE)
extern void hypre_F90_NAME(fhypre_structpfmggetrelaxtype, FHYPRE_STRUCTPFMGGETRELAXTYPE)
                          (long int *, int *);

#define HYPRE_StructPFMGSetRAPType \
        hypre_F90_NAME(fhypre_structpfmgsetraptype, FHYPRE_STRUCTPFMGSETRAPTYPE)
extern void hypre_F90_NAME(fhypre_structpfmgsetraptype, FHYPRE_STRUCTPFMGSETRAPTYPE)
                          (long int *, int *);

#define HYPRE_StructPFMGGetRAPType \
        hypre_F90_NAME(fhypre_structpfmggetraptype, FHYPRE_STRUCTPFMGGETRAPTYPE)
extern void hypre_F90_NAME(fhypre_structpfmggetraptype, FHYPRE_STRUCTPFMGGETRAPTYPE)
                          (long int *, int *);

#define HYPRE_StructPFMGSetNumPreRelax \
        hypre_F90_NAME(fhypre_structpfmgsetnumprerelax, FHYPRE_STRUCTPFMGSETNUMPRERELAX)
extern void hypre_F90_NAME(fhypre_structpfmgsetnumprerelax, FHYPRE_STRUCTPFMGSETNUMPRERELAX)
                          (long int *, int *);

#define HYPRE_StructPFMGGetNumPreRelax \
        hypre_F90_NAME(fhypre_structpfmggetnumprerelax, FHYPRE_STRUCTPFMGGETNUMPRERELAX)
extern void hypre_F90_NAME(fhypre_structpfmggetnumprerelax, FHYPRE_STRUCTPFMGGETNUMPRERELAX)
                          (long int *, int *);

#define HYPRE_StructPFMGSetNumPostRelax \
        hypre_F90_NAME(fhypre_structpfmgsetnumpostrela, FHYPRE_STRUCTPFMGSETNUMPOSTRELA)
extern void hypre_F90_NAME(fhypre_structpfmgsetnumpostrela, FHYPRE_STRUCTPFMGSETNUMPOSTRELA)
                          (long int *, int *);

#define HYPRE_StructPFMGGetNumPostRelax \
        hypre_F90_NAME(fhypre_structpfmggetnumpostrela, FHYPRE_STRUCTPFMGGETNUMPOSTRELA)
extern void hypre_F90_NAME(fhypre_structpfmggetnumpostrela, FHYPRE_STRUCTPFMGGETNUMPOSTRELA)
                          (long int *, int *);

#define HYPRE_StructPFMGSetDxyz \
        hypre_F90_NAME(fhypre_structpfmgsetdxyz, FHYPRE_STRUCTPFMGSETDXYZ)
extern void hypre_F90_NAME(fhypre_structpfmgsetdxyz, FHYPRE_STRUCTPFMGSETDXYZ)
                          (long int *, double *);

#define HYPRE_StructPFMGSetLogging \
        hypre_F90_NAME(fhypre_structpfmgsetlogging, FHYPRE_STRUCTPFMGSETLOGGING)
extern void hypre_F90_NAME(fhypre_structpfmgsetlogging, FHYPRE_STRUCTPFMGSETLOGGING)
                          (long int *, int *);

#define HYPRE_StructPFMGGetLogging \
        hypre_F90_NAME(fhypre_structpfmggetlogging, FHYPRE_STRUCTPFMGGETLOGGING)
extern void hypre_F90_NAME(fhypre_structpfmggetlogging, FHYPRE_STRUCTPFMGGETLOGGING)
                          (long int *, int *);

#define HYPRE_StructPFMGSetPrintLevel \
        hypre_F90_NAME(fhypre_structpfmgsetprintlevel, FHYPRE_STRUCTPFMGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structpfmgsetprintlevel, FHYPRE_STRUCTPFMGSETPRINTLEVEL)
                          (long int *, int *);

#define HYPRE_StructPFMGGetPrintLevel \
        hypre_F90_NAME(fhypre_structpfmggetprintlevel, FHYPRE_STRUCTPFMGGETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structpfmggetprintlevel, FHYPRE_STRUCTPFMGGETPRINTLEVEL)
                          (long int *, int *);

#define HYPRE_StructPFMGGetNumIterations \
        hypre_F90_NAME(fhypre_structpfmggetnumiteratio, FHYPRE_STRUCTPFMGGETNUMITERATIO)
extern void hypre_F90_NAME(fhypre_structpfmggetnumiteratio, FHYPRE_STRUCTPFMGGETNUMITERATIO)
                          (long int *, int *);

#define HYPRE_StructPFMGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structpfmggetfinalrelati, FHYPRE_STRUCTPFMGGETFINALRELATI)
extern void hypre_F90_NAME(fhypre_structpfmggetfinalrelati, FHYPRE_STRUCTPFMGGETFINALRELATI)
                          (long int *, double *);



#define HYPRE_StructSMGCreate \
        hypre_F90_NAME(fhypre_structsmgcreate, FHYPRE_STRUCTSMGCREATE)
extern void hypre_F90_NAME(fhypre_structsmgcreate, FHYPRE_STRUCTSMGCREATE)
                          (int *, long int *);

#define HYPRE_StructSMGDestroy \
        hypre_F90_NAME(fhypre_structsmgdestroy, FHYPRE_STRUCTSMGDESTROY)
extern void hypre_F90_NAME(fhypre_structsmgdestroy, FHYPRE_STRUCTSMGDESTROY)
                          (long int *);

#define HYPRE_StructSMGSetup \
        hypre_F90_NAME(fhypre_structsmgsetup, FHYPRE_STRUCTSMGSETUP)
extern void hypre_F90_NAME(fhypre_structsmgsetup, FHYPRE_STRUCTSMGSETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_StructSMGSolve \
        hypre_F90_NAME(fhypre_structsmgsolve, FHYPRE_STRUCTSMGSOLVE)
extern void hypre_F90_NAME(fhypre_structsmgsolve, FHYPRE_STRUCTSMGSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_StructSMGSetMemoryUse \
        hypre_F90_NAME(fhypre_structsmgsetmemoryuse, FHYPRE_STRUCTSMGSETMEMORYUSE)
extern void hypre_F90_NAME(fhypre_structsmgsetmemoryuse, FHYPRE_STRUCTSMGSETMEMORYUSE)
                          (long int *, int *);

#define HYPRE_StructSMGGetMemoryUse \
        hypre_F90_NAME(fhypre_structsmggetmemoryuse, FHYPRE_STRUCTSMGGETMEMORYUSE)
extern void hypre_F90_NAME(fhypre_structsmggetmemoryuse, FHYPRE_STRUCTSMGGETMEMORYUSE)
                          (long int *, int *);

#define HYPRE_StructSMGSetTol \
        hypre_F90_NAME(fhypre_structsmgsettol, FHYPRE_STRUCTSMGSETTOL)
extern void hypre_F90_NAME(fhypre_structsmgsettol, FHYPRE_STRUCTSMGSETTOL)
                          (long int *, double *);

#define HYPRE_StructSMGGetTol \
        hypre_F90_NAME(fhypre_structsmggettol, FHYPRE_STRUCTSMGGETTOL)
extern void hypre_F90_NAME(fhypre_structsmggettol, FHYPRE_STRUCTSMGGETTOL)
                          (long int *, double *);

#define HYPRE_StructSMGSetMaxIter \
        hypre_F90_NAME(fhypre_structsmgsetmaxiter, FHYPRE_STRUCTSMGSETMAXTITER)
extern void hypre_F90_NAME(fhypre_structsmgsetmaxiter, FHYPRE_STRUCTSMGSETMAXTITER)
                          (long int *, int *);

#define HYPRE_StructSMGGetMaxIter \
        hypre_F90_NAME(fhypre_structsmggetmaxiter, FHYPRE_STRUCTSMGGETMAXTITER)
extern void hypre_F90_NAME(fhypre_structsmggetmaxiter, FHYPRE_STRUCTSMGGETMAXTITER)
                          (long int *, int *);

#define HYPRE_StructSMGSetRelChange \
        hypre_F90_NAME(fhypre_structsmgsetrelchange, FHYPRE_STRUCTSMGSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structsmgsetrelchange, FHYPRE_STRUCTSMGSETRELCHANGE)
                          (long int *, int *);

#define HYPRE_StructSMGGetRelChange \
        hypre_F90_NAME(fhypre_structsmggetrelchange, FHYPRE_STRUCTSMGGETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structsmggetrelchange, FHYPRE_STRUCTSMGGETRELCHANGE)
                          (long int *, int *);

#define HYPRE_StructSMGSetZeroGuess \
        hypre_F90_NAME(fhypre_structsmgsetzeroguess, FHYPRE_STRUCTSMGSETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structsmgsetzeroguess, FHYPRE_STRUCTSMGSETZEROGUESS)
                          (long int *);

#define HYPRE_StructSMGGetZeroGuess \
        hypre_F90_NAME(fhypre_structsmggetzeroguess, FHYPRE_STRUCTSMGGETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structsmggetzeroguess, FHYPRE_STRUCTSMGGETZEROGUESS)
                          (long int *);

#define HYPRE_StructSMGSetNonZeroGuess \
        hypre_F90_NAME(fhypre_structsmgsetnonzerogues, FHYPRE_STRUCTSMGSETNONZEROGUES)
extern void hypre_F90_NAME(fhypre_structsmgsetnonzerogues, FHYPRE_STRUCTSMGSETNONZEROGUES)
                          (long int *);

#define HYPRE_StructSMGGetNumIterations \
        hypre_F90_NAME(fhypre_structsmggetnumiteration, FHYPRE_STRUCTSMGGETNUMITERATION)
extern void hypre_F90_NAME(fhypre_structsmggetnumiteration, FHYPRE_STRUCTSMGGETNUMITERATION)
                          (long int *, int *);

#define HYPRE_StructSMGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structsmggetfinalrelativ, FHYPRE_STRUCTSMGGETFINALRELATIV)
extern void hypre_F90_NAME(fhypre_structsmggetfinalrelativ, FHYPRE_STRUCTSMGGETFINALRELATIV)
                          (long int *, double *);

#define HYPRE_StructSMGSetNumPreRelax \
        hypre_F90_NAME(fhypre_structsmgsetnumprerelax, FHYPRE_STRUCTSMGSETNUMPRERELAX)
extern void hypre_F90_NAME(fhypre_structsmgsetnumprerelax, FHYPRE_STRUCTSMGSETNUMPRERELAX)
                          (long int *, int *);

#define HYPRE_StructSMGGetNumPreRelax \
        hypre_F90_NAME(fhypre_structsmggetnumprerelax, FHYPRE_STRUCTSMGGETNUMPRERELAX)
extern void hypre_F90_NAME(fhypre_structsmggetnumprerelax, FHYPRE_STRUCTSMGGETNUMPRERELAX)
                          (long int *, int *);

#define HYPRE_StructSMGSetNumPostRelax \
        hypre_F90_NAME(fhypre_structsmgsetnumpostrelax, FHYPRE_STRUCTSMGSETNUMPOSTRELAX)
extern void hypre_F90_NAME(fhypre_structsmgsetnumpostrelax, FHYPRE_STRUCTSMGSETNUMPOSTRELAX)
                          (long int *, int *);

#define HYPRE_StructSMGGetNumPostRelax \
        hypre_F90_NAME(fhypre_structsmggetnumpostrelax, FHYPRE_STRUCTSMGGETNUMPOSTRELAX)
extern void hypre_F90_NAME(fhypre_structsmggetnumpostrelax, FHYPRE_STRUCTSMGGETNUMPOSTRELAX)
                          (long int *, int *);

#define HYPRE_StructSMGSetLogging \
        hypre_F90_NAME(fhypre_structsmgsetlogging, FHYPRE_STRUCTSMGSETLOGGING)
extern void hypre_F90_NAME(fhypre_structsmgsetlogging, FHYPRE_STRUCTSMGSETLOGGING)
                          (long int *, int *);

#define HYPRE_StructSMGGetLogging \
        hypre_F90_NAME(fhypre_structsmggetlogging, FHYPRE_STRUCTSMGGETLOGGING)
extern void hypre_F90_NAME(fhypre_structsmggetlogging, FHYPRE_STRUCTSMGGETLOGGING)
                          (long int *, int *);

#define HYPRE_StructSMGSetPrintLevel \
        hypre_F90_NAME(fhypre_structsmgsetprintlevel, FHYPRE_STRUCTSMGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structsmgsetprintlevel, FHYPRE_STRUCTSMGSETPRINTLEVEL)
                          (long int *, int *);

#define HYPRE_StructSMGGetPrintLevel \
        hypre_F90_NAME(fhypre_structsmggetprintlevel, FHYPRE_STRUCTSMGGETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structsmggetprintlevel, FHYPRE_STRUCTSMGGETPRINTLEVEL)
                          (long int *, int *);



#define HYPRE_StructSparseMSGCreate \
        hypre_F90_NAME(fhypre_structsparsemsgcreate, FHYPRE_STRUCTSPARSEMSGCREATE)
extern void hypre_F90_NAME(fhypre_structsparsemsgcreate, FHYPRE_STRUCTSPARSEMSGCREATE)
                          (int *, long int *);

#define HYPRE_StructSparseMSGDestroy \
        hypre_F90_NAME(fhypre_structsparsemsgdestroy, FHYPRE_STRUCTSPARSEMSGDESTROY)
extern void hypre_F90_NAME(fhypre_structsparsemsgdestroy, FHYPRE_STRUCTSPARSEMSGDESTROY)
                          (long int *);

#define HYPRE_StructSparseMSGSetup \
        hypre_F90_NAME(fhypre_structsparsemsgsetup, FHYRPE_STRUCTSPARSEMSGSETUP)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetup, FHYRPE_STRUCTSPARSEMSGSETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_StructSparseMSGSolve \
        hypre_F90_NAME(fhypre_structsparsemsgsolve, FHYPRE_STRUCTSPARSEMSGSOLVE)
extern void hypre_F90_NAME(fhypre_structsparsemsgsolve, FHYPRE_STRUCTSPARSEMSGSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_StructSparseMSGSetJump \
        hypre_F90_NAME(fhypre_structsparsemsgsetjump, FHYPRE_STRUCTSPARSEMSGSETJUMP)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetjump, FHYPRE_STRUCTSPARSEMSGSETJUMP)
                          (long int *, int *);

#define HYPRE_StructSparseMSGSetTol \
        hypre_F90_NAME(fhypre_structsparsemsgsettol, FHYPRE_STRUCTSPARSEMSGSETTOL)
extern void hypre_F90_NAME(fhypre_structsparsemsgsettol, FHYPRE_STRUCTSPARSEMSGSETTOL)
                          (long int *, double *);

#define HYPRE_StructSparseMSGSetMaxIter \
        hypre_F90_NAME(fhypre_structsparsemsgsetmaxite, FHYPRE_STRUCTSPARSEMSGSETMAXITE)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetmaxite, FHYPRE_STRUCTSPARSEMSGSETMAXITE)
                          (long int *, int *);

#define HYPRE_StructSparseMSGSetRelChange \
        hypre_F90_NAME(fhypre_structsparsemsgsetrelcha, FHYPRE_STRUCTSPARSEMSGSETRELCHA)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetrelcha, FHYPRE_STRUCTSPARSEMSGSETRELCHA)
                          (long int *, int *);

#define HYPRE_StructSparseMSGSetZeroGuess \
        hypre_F90_NAME(fhypre_structsparsemsgsetzerogu, FHYPRE_STRUCTSPARSEMSGSETZEROGU)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetzerogu, FHYPRE_STRUCTSPARSEMSGSETZEROGU)
                          (long int *);

#define HYPRE_StructSparseMSGSetNonZeroGuess \
        hypre_F90_NAME(fhypre_structsparsemsgsetnonzer, FHYPRE_STRUCTSPARSEMSGSETNONZER)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetnonzer, FHYPRE_STRUCTSPARSEMSGSETNONZER)
                          (long int *);

#define HYPRE_StructSparseMSGGetNumIterations \
        hypre_F90_NAME(fhypre_structsparsemsggetnumite, FHYPRE_STRUCTSPARSEMSGGETNUMITE)
extern void hypre_F90_NAME(fhypre_structsparsemsggetnumite, FHYPRE_STRUCTSPARSEMSGGETNUMITE)
                          (long int *, int *);

#define HYPRE_StructSparseMSGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structsparsemsggetfinalr, FHYRPE_STRUCTSPARSEMSGGETFINALR)
extern void hypre_F90_NAME(fhypre_structsparsemsggetfinalr, FHYRPE_STRUCTSPARSEMSGGETFINALR)
                          (long int *, double *);

#define HYPRE_StructSparseMSGSetRelaxType \
        hypre_F90_NAME(fhypre_structsparsemsgsetrelaxt, FHYPRE_STRUCTSPARSEMSGSETRELAXT)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetrelaxt, FHYPRE_STRUCTSPARSEMSGSETRELAXT)
                          (long int *, int *);

#define HYPRE_StructSparseMSGSetNumPreRelax \
        hypre_F90_NAME(fhypre_structsparsemsgsetnumpre, FHYPRE_STRUCTSPARSEMSGSETNUMPRE)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetnumpre, FHYPRE_STRUCTSPARSEMSGSETNUMPRE)
                          (long int *, int *);

#define HYPRE_StructSparseMSGSetNumPostRelax \
        hypre_F90_NAME(fhypre_structsparsemsgsetnumpos, FHYPRE_STRUCTSPARSEMSGSETNUMPOS)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetnumpos, FHYPRE_STRUCTSPARSEMSGSETNUMPOS)
                          (long int *, int *);

#define HYPRE_StructSparseMSGSetNumFineRelax \
        hypre_F90_NAME(fhypre_structsparsemsgsetnumfin, FHYPRE_STRUCTSPARSEMSGSETNUMFIN)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetnumfin, FHYPRE_STRUCTSPARSEMSGSETNUMFIN)
                          (long int *, int *);

#define HYPRE_StructSparseMSGSetLogging \
        hypre_F90_NAME(fhypre_structsparsemsgsetloggin, FHYPRE_STRUCTSPARSEMSGSETLOGGIN)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetloggin, FHYPRE_STRUCTSPARSEMSGSETLOGGIN)
                          (long int *, int *);

#define HYPRE_StructSparseMSGSetPrintLevel \
        hypre_F90_NAME(fhypre_structsparsemsgsetprintl, FHYPRE_STRUCTSPARSEMSGSETPRINTL)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetprintl, FHYPRE_STRUCTSPARSEMSGSETPRINTL)
                          (long int *, int *);
