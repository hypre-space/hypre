/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.4 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 * Definitions of IJMatrix Fortran interface routines
 *****************************************************************************/

#define HYPRE_IJMatrixCreate \
        hypre_F90_NAME(fhypre_ijmatrixcreate, FHYPRE_IJMATRIXCREATE)
extern void hypre_F90_NAME(fhypre_ijmatrixcreate, FHYPRE_IJMATRIXCREATE)
                      (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, hypre_F90_Obj*);

#define HYPRE_IJMatrixDestroy \
        hypre_F90_NAME(fhypre_ijmatrixdestroy, FHYPRE_IJMATRIXDESTROY)
extern void hypre_F90_NAME(fhypre_ijmatrixdestroy, FHYPRE_IJMATRIXDESTROY)
                      (hypre_F90_Obj *);

#define HYPRE_IJMatrixInitialize \
        hypre_F90_NAME(fhypre_ijmatrixinitialize, FHYPRE_IJMATRIXINITIALIZE)
extern void hypre_F90_NAME(fhypre_ijmatrixinitialize, FHYPRE_IJMATRIXINITIALIZE)
                      (hypre_F90_Obj *);

#define HYPRE_IJMatrixAssemble \
        hypre_F90_NAME(fhypre_ijmatrixassemble, FHYPRE_IJMATRIXASSEMBLE)
extern void hypre_F90_NAME(fhypre_ijmatrixassemble, FHYPRE_IJMATRIXASSEMBLE)
                      (hypre_F90_Obj *);

#define HYPRE_IJMatrixSetRowSizes \
        hypre_F90_NAME(fhypre_ijmatrixsetrowsizes, FHYPRE_IJMATRIXSETROWSIZES)
extern void hypre_F90_NAME(fhypre_ijmatrixsetrowsizes, FHYPRE_IJMATRIXSETROWSIZES)
                      (hypre_F90_Obj *, const HYPRE_Int *);

#define HYPRE_IJMatrixSetDiagOffdSizes \
        hypre_F90_NAME(fhypre_ijmatrixsetdiagoffdsizes, FHYPRE_IJMATRIXSETDIAGOFFDSIZES)
extern void hypre_F90_NAME(fhypre_ijmatrixsetdiagoffdsizes, FHYPRE_IJMATRIXSETDIAGOFFDSIZES)
                      (hypre_F90_Obj *, const HYPRE_Int *, const HYPRE_Int *);

#define HYPRE_IJMatrixSetValues \
        hypre_F90_NAME(fhypre_ijmatrixsetvalues, FHYPRE_IJMATRIXSETVALUES)
extern void hypre_F90_NAME(fhypre_ijmatrixsetvalues, FHYPRE_IJMATRIXSETVALUES)
                      (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *, const HYPRE_Int *, const HYPRE_Int *,
                       const double *);

#define HYPRE_IJMatrixAddToValues \
        hypre_F90_NAME(fhypre_ijmatrixaddtovalues, FHYPRE_IJMATRIXADDTOVALUES)
extern void hypre_F90_NAME(fhypre_ijmatrixaddtovalues, FHYPRE_IJMATRIXADDTOVALUES)
                      (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *, const HYPRE_Int *, const HYPRE_Int *,
                       const double *);

#define HYPRE_IJMatrixSetObjectType \
        hypre_F90_NAME(fhypre_ijmatrixsetobjecttype, FHYPRE_IJMATRIXSETOBJECTTYPE)
extern void hypre_F90_NAME(fhypre_ijmatrixsetobjecttype, FHYPRE_IJMATRIXSETOBJECTTYPE)
                      (hypre_F90_Obj *, const HYPRE_Int *);

#define HYPRE_IJMatrixGetObjectType \
        hypre_F90_NAME(fhypre_ijmatrixgetobjecttype, FHYPRE_IJMATRIXGETOBJECTTYPE)
extern void hypre_F90_NAME(fhypre_ijmatrixgetobjecttype, FHYPRE_IJMATRIXGETOBJECTTYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_IJMatrixGetObject \
        hypre_F90_NAME(fhypre_ijmatrixgetobject, FHYPRE_IJMATRIXGETOBJECT)
extern void hypre_F90_NAME(fhypre_ijmatrixgetobject, FHYPRE_IJMATRIXGETOBJECT)
                      (hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_IJMatrixRead \
        hypre_F90_NAME(fhypre_ijmatrixread, FHYPRE_IJMATRIXREAD)
extern void hypre_F90_NAME(fhypre_ijmatrixread, FHYPRE_IJMATRIXREAD)
                      (char *, hypre_F90_Obj *, HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_IJMatrixPrint \
        hypre_F90_NAME(fhypre_ijmatrixprint, FHYPRE_IJMATRIXPRINT)
extern void hypre_F90_NAME(fhypre_ijmatrixprint, FHYPRE_IJMATRIXPRINT)
                      (hypre_F90_Obj *, char *);



#define hypre_IJMatrixSetObject \
        hypre_F90_NAME(fhypre_ijmatrixsetobject, FHYPRE_IJMATRIXSETOBJECT)
extern void hypre_F90_NAME(fhypre_ijmatrixsetobject, FHYPRE_IJMATRIXSETOBJECT)
                      (hypre_F90_Obj *, hypre_F90_Obj *);



#define HYPRE_IJVectorCreate \
        hypre_F90_NAME(fhypre_ijvectorcreate, FHYPRE_IJVECTORCREATE)
extern void hypre_F90_NAME(fhypre_ijvectorcreate, FHYPRE_IJVECTORCREATE)
                      (HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_IJVectorDestroy \
        hypre_F90_NAME(fhypre_ijvectordestroy, FHYPRE_IJVECTORDESTROY)
extern void hypre_F90_NAME(fhypre_ijvectordestroy, FHYPRE_IJVECTORDESTROY)
                      (hypre_F90_Obj *);

#define HYPRE_IJVectorInitialize \
        hypre_F90_NAME(fhypre_ijvectorinitialize, FHYPRE_IJVECTORINITIALIZE)
extern void hypre_F90_NAME(fhypre_ijvectorinitialize, FHYPRE_IJVECTORINITIALIZE)
                      (hypre_F90_Obj *);

#define HYPRE_IJVectorSetValues \
        hypre_F90_NAME(fhypre_ijvectorsetvalues, FHYPRE_IJVECTORSETVALUES)
extern void hypre_F90_NAME(fhypre_ijvectorsetvalues, FHYPRE_IJVECTORSETVALUES)
                      (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *, double *);

#define HYPRE_IJVectorAddToValues \
        hypre_F90_NAME(fhypre_ijvectoraddtovalues, FHYPRE_IJVECTORADDTOVALUES)
extern void hypre_F90_NAME(fhypre_ijvectoraddtovalues, FHYPRE_IJVECTORADDTOVALUES)
                      (hypre_F90_Obj *, HYPRE_Int *, HYPRE_Int *, double *);

#define HYPRE_IJVectorAssemble \
        hypre_F90_NAME(fhypre_ijvectorassemble, FHYPRE_IJVECTORASSEMBLE)
extern void hypre_F90_NAME(fhypre_ijvectorassemble, FHYPRE_IJVECTORASSEMBLE)
                      (hypre_F90_Obj *);

#define HYPRE_IJVectorGetValues \
        hypre_F90_NAME(fhypre_ijvectorgetvalues, FHYPRE_IJVECTORGETVALUES)
extern void hypre_F90_NAME(fhypre_ijvectorgetvalues, FHYPRE_IJVECTORGETVALUES)
                      (hypre_F90_Obj *, const HYPRE_Int *, const HYPRE_Int *, double *);

#define HYPRE_IJVectorSetObjectType \
        hypre_F90_NAME(fhypre_ijvectorsetobjecttype, FHYPRE_IJVECTORSETOBJECTTYPE)
extern void hypre_F90_NAME(fhypre_ijvectorsetobjecttype, FHYPRE_IJVECTORSETOBJECTTYPE)
                      (hypre_F90_Obj *, const HYPRE_Int *);

#define HYPRE_IJVectorGetObjectType \
        hypre_F90_NAME(fhypre_ijvectorgetobjecttype, FHYPRE_IJVECTORGETOBJECTTYPE)
extern void hypre_F90_NAME(fhypre_ijvectorgetobjecttype, FHYPRE_IJVECTORGETOBJECTTYPE)
                      (hypre_F90_Obj *, HYPRE_Int *);

#define HYPRE_IJVectorGetObject \
        hypre_F90_NAME(fhypre_ijvectorgetobject, FHYPRE_IJVECTORGETOBJECT)
extern void hypre_F90_NAME(fhypre_ijvectorgetobject, FHYPRE_IJVECTORGETOBJECT)
                      (hypre_F90_Obj *, hypre_F90_Obj *);

#define HYPRE_IJVectorRead \
        hypre_F90_NAME(fhypre_ijvectorread, FHYPRE_IJVECTORREAD)
extern void hypre_F90_NAME(fhypre_ijvectorread, FHYPRE_IJVECTORREAD)
                      (char *, hypre_F90_Obj *, HYPRE_Int *, hypre_F90_Obj *);

#define HYPRE_IJVectorPrint \
        hypre_F90_NAME(fhypre_ijvectorprint, FHYPRE_IJVECTORPRINT)
extern void hypre_F90_NAME(fhypre_ijvectorprint, FHYPRE_IJVECTORPRINT)
                      (hypre_F90_Obj *, char *);
