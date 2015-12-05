/******************************************************************************
 * Definitions of IJMatrix Fortran interface routines
 *****************************************************************************/

#define HYPRE_IJMatrixCreate \
        hypre_F90_NAME(fhypre_ijmatrixcreate, FHYPRE_IJMATRIXCREATE)
extern void hypre_F90_NAME(fhypre_ijmatrixcreate, FHYPRE_IJMATRIXCREATE)
                      (long int *, int *, int *, int *, int *, long int*);

#define HYPRE_IJMatrixDestroy \
        hypre_F90_NAME(fhypre_ijmatrixdestroy, FHYPRE_IJMATRIXDESTROY)
extern void hypre_F90_NAME(fhypre_ijmatrixdestroy, FHYPRE_IJMATRIXDESTROY)
                      (long int *);

#define HYPRE_IJMatrixInitialize \
        hypre_F90_NAME(fhypre_ijmatrixinitialize, FHYPRE_IJMATRIXINITIALIZE)
extern void hypre_F90_NAME(fhypre_ijmatrixinitialize, FHYPRE_IJMATRIXINITIALIZE)
                      (long int *);

#define HYPRE_IJMatrixAssemble \
        hypre_F90_NAME(fhypre_ijmatrixassemble, FHYPRE_IJMATRIXASSEMBLE)
extern void hypre_F90_NAME(fhypre_ijmatrixassemble, FHYPRE_IJMATRIXASSEMBLE)
                      (long int *);

#define HYPRE_IJMatrixSetRowSizes \
        hypre_F90_NAME(fhypre_ijmatrixsetrowsizes, FHYPRE_IJMATRIXSETROWSIZES)
extern void hypre_F90_NAME(fhypre_ijmatrixsetrowsizes, FHYPRE_IJMATRIXSETROWSIZES)
                      (long int *, const int *);

#define HYPRE_IJMatrixSetDiagOffdSizes \
        hypre_F90_NAME(fhypre_ijmatrixsetdiagoffdsizes, FHYPRE_IJMATRIXSETDIAGOFFDSIZES)
extern void hypre_F90_NAME(fhypre_ijmatrixsetdiagoffdsizes, FHYPRE_IJMATRIXSETDIAGOFFDSIZES)
                      (long int *, const int *, const int *);

#define HYPRE_IJMatrixSetValues \
        hypre_F90_NAME(fhypre_ijmatrixsetvalues, FHYPRE_IJMATRIXSETVALUES)
extern void hypre_F90_NAME(fhypre_ijmatrixsetvalues, FHYPRE_IJMATRIXSETVALUES)
                      (long int *, int *, int *, const int *, const int *,
                       const double *);

#define HYPRE_IJMatrixAddToValues \
        hypre_F90_NAME(fhypre_ijmatrixaddtovalues, FHYPRE_IJMATRIXADDTOVALUES)
extern void hypre_F90_NAME(fhypre_ijmatrixaddtovalues, FHYPRE_IJMATRIXADDTOVALUES)
                      (long int *, int *, int *, const int *, const int *,
                       const double *);

#define HYPRE_IJMatrixSetObjectType \
        hypre_F90_NAME(fhypre_ijmatrixsetobjecttype, FHYPRE_IJMATRIXSETOBJECTTYPE)
extern void hypre_F90_NAME(fhypre_ijmatrixsetobjecttype, FHYPRE_IJMATRIXSETOBJECTTYPE)
                      (long int *, const int *);

#define HYPRE_IJMatrixGetObjectType \
        hypre_F90_NAME(fhypre_ijmatrixgetobjecttype, FHYPRE_IJMATRIXGETOBJECTTYPE)
extern void hypre_F90_NAME(fhypre_ijmatrixgetobjecttype, FHYPRE_IJMATRIXGETOBJECTTYPE)
                      (long int *, int *);

#define HYPRE_IJMatrixGetObject \
        hypre_F90_NAME(fhypre_ijmatrixgetobject, FHYPRE_IJMATRIXGETOBJECT)
extern void hypre_F90_NAME(fhypre_ijmatrixgetobject, FHYPRE_IJMATRIXGETOBJECT)
                      (long int *, long int *);

#define HYPRE_IJMatrixRead \
        hypre_F90_NAME(fhypre_ijmatrixread, FHYPRE_IJMATRIXREAD)
extern void hypre_F90_NAME(fhypre_ijmatrixread, FHYPRE_IJMATRIXREAD)
                      (char *, long int *, int *, long int *);

#define HYPRE_IJMatrixPrint \
        hypre_F90_NAME(fhypre_ijmatrixprint, FHYPRE_IJMATRIXPRINT)
extern void hypre_F90_NAME(fhypre_ijmatrixprint, FHYPRE_IJMATRIXPRINT)
                      (long int *, char *);



#define hypre_IJMatrixSetObject \
        hypre_F90_NAME(fhypre_ijmatrixsetobject, FHYPRE_IJMATRIXSETOBJECT)
extern void hypre_F90_NAME(fhypre_ijmatrixsetobject, FHYPRE_IJMATRIXSETOBJECT)
                      (long int *, long int *);



#define HYPRE_IJVectorCreate \
        hypre_F90_NAME(fhypre_ijvectorcreate, FHYPRE_IJVECTORCREATE)
extern void hypre_F90_NAME(fhypre_ijvectorcreate, FHYPRE_IJVECTORCREATE)
                      (int *, int *, int *, long int *);

#define HYPRE_IJVectorDestroy \
        hypre_F90_NAME(fhypre_ijvectordestroy, FHYPRE_IJVECTORDESTROY)
extern void hypre_F90_NAME(fhypre_ijvectordestroy, FHYPRE_IJVECTORDESTROY)
                      (long int *);

#define HYPRE_IJVectorInitialize \
        hypre_F90_NAME(fhypre_ijvectorinitialize, FHYPRE_IJVECTORINITIALIZE)
extern void hypre_F90_NAME(fhypre_ijvectorinitialize, FHYPRE_IJVECTORINITIALIZE)
                      (long int *);

#define HYPRE_IJVectorSetValues \
        hypre_F90_NAME(fhypre_ijvectorsetvalues, FHYPRE_IJVECTORSETVALUES)
extern void hypre_F90_NAME(fhypre_ijvectorsetvalues, FHYPRE_IJVECTORSETVALUES)
                      (long int *, int *, int *, double *);

#define HYPRE_IJVectorAddToValues \
        hypre_F90_NAME(fhypre_ijvectoraddtovalues, FHYPRE_IJVECTORADDTOVALUES)
extern void hypre_F90_NAME(fhypre_ijvectoraddtovalues, FHYPRE_IJVECTORADDTOVALUES)
                      (long int *, int *, int *, double *);

#define HYPRE_IJVectorAssemble \
        hypre_F90_NAME(fhypre_ijvectorassemble, FHYPRE_IJVECTORASSEMBLE)
extern void hypre_F90_NAME(fhypre_ijvectorassemble, FHYPRE_IJVECTORASSEMBLE)
                      (long int *);

#define HYPRE_IJVectorGetValues \
        hypre_F90_NAME(fhypre_ijvectorgetvalues, FHYPRE_IJVECTORGETVALUES)
extern void hypre_F90_NAME(fhypre_ijvectorgetvalues, FHYPRE_IJVECTORGETVALUES)
                      (long int *, const int *, const int *, double *);

#define HYPRE_IJVectorSetObjectType \
        hypre_F90_NAME(fhypre_ijvectorsetobjecttype, FHYPRE_IJVECTORSETOBJECTTYPE)
extern void hypre_F90_NAME(fhypre_ijvectorsetobjecttype, FHYPRE_IJVECTORSETOBJECTTYPE)
                      (long int *, const int *);

#define HYPRE_IJVectorGetObjectType \
        hypre_F90_NAME(fhypre_ijvectorgetobjecttype, FHYPRE_IJVECTORGETOBJECTTYPE)
extern void hypre_F90_NAME(fhypre_ijvectorgetobjecttype, FHYPRE_IJVECTORGETOBJECTTYPE)
                      (long int *, int *);

#define HYPRE_IJVectorGetObject \
        hypre_F90_NAME(fhypre_ijvectorgetobject, FHYPRE_IJVECTORGETOBJECT)
extern void hypre_F90_NAME(fhypre_ijvectorgetobject, FHYPRE_IJVECTORGETOBJECT)
                      (long int *, long int *);

#define HYPRE_IJVectorRead \
        hypre_F90_NAME(fhypre_ijvectorread, FHYPRE_IJVECTORREAD)
extern void hypre_F90_NAME(fhypre_ijvectorread, FHYPRE_IJVECTORREAD)
                      (char *, long int *, int *, long int *);

#define HYPRE_IJVectorPrint \
        hypre_F90_NAME(fhypre_ijvectorprint, FHYPRE_IJVECTORPRINT)
extern void hypre_F90_NAME(fhypre_ijvectorprint, FHYPRE_IJVECTORPRINT)
                      (long int *, char *);
