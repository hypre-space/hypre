/*--------------------------------------------------------------------------
 * Author:        Andy Cleary
 *                Centre for Applied Scientific Computing
 *                Lawrence Livermore Labs
 * Revision History:
 *  5-22-97: AJC. Original Version.
 *--------------------------------------------------------------------------*/


#define INCFACTFact

/*****************************************************************************
 * C to Fortran interfacing macros
 *****************************************************************************/

/*****************************************************************************
 * C routine would use this macro to call "incfactt_driver.f" :
 *
 *  CALL_INCFACTT_DRIVER(n, nnz,
 *                   ia, ja, a,
 *                   ipar, rpar,
 *                   lenpmx,
#ifdef ILUFact
 *                   plu, jlu, ju, perm, qperm, rscale, cscale,
#elif defined ICFact
 *                   plu, jlu, perm, qperm, scale,
#endif
 *                   iwork, liw, rwork, lrw,
 *                   ier_incfactt, ier_input);
 *
 *****************************************************************************/
#ifdef _CRAYMPP
#define INCFACTT_DRIVER INCFACTT_DRIVER
#else
#define INCFACTT_DRIVER incfactt_driver__
#endif

#ifdef ILUFact
#define CALL_INCFACTT_DRIVER(n, nnz,\
                         ia, ja, a, \
                         ipar, rpar,\
                         lenpmx,\
                         plu, jlu, ju, perm, qperm, rscale, cscale, \
                         iwork, liw, rwork, lrw,\
                         ier_incfactt, ier_input) \
             INCFACTT_DRIVER(&n, &nnz,\
                         ia, ja, a, \
                         ipar, rpar,\
                         &lenpmx,\
                         plu, jlu, ju, perm, qperm, rscale, cscale, \
                         iwork, &liw, rwork, &lrw,\
                         &ier_incfactt, &ier_input)
#elif defined ICFact
#define CALL_INCFACTT_DRIVER(n, nnz,\
                         ia, ja, a, \
                         ipar, rpar,\
                         lenpmx,\
                         plu, jlu, perm, qperm, scale, \
                         iwork, liw, rwork, lrw,\
                         ier_incfactt, ier_input) \
             INCFACTT_DRIVER(&n, &nnz,\
                         ia, ja, a, \
                         ipar, rpar,\
                         &lenpmx,\
                         plu, jlu, perm, qperm, scale, \
                         iwork, &liw, rwork, &lrw,\
                         &ier_incfactt, &ier_input)
#endif

void INCFACTT_DRIVER(HYPRE_Int *n, HYPRE_Int *nnz,
                 HYPRE_Int *ia, HYPRE_Int *ja, double *a,
                 HYPRE_Int *ipar, double *rpar,
                 HYPRE_Int *lenpmx,
                 double *plu, HYPRE_Int *jlu, 
#ifdef ILUFact
                 HYPRE_Int *ju, 
#endif
                 HYPRE_Int *perm, HYPRE_Int *qperm, 
#ifdef ILUFact
                 double *rscale, double *cscale, 
#endif
#ifdef ICFact
                 double *scale,
#endif
                 HYPRE_Int *iwork, HYPRE_Int *liw, double *rwork, HYPRE_Int *lrw,
                 HYPRE_Int *ier_incfactt, HYPRE_Int *ier_input);



/*****************************************************************************
 * C routine would use this macro to call "ksp_driver.f" :
 *
 *  CALL_KSP_DRIVER(n, nnz,
 *                   ia, ja, a, rhs, sol,
 *                   ipar, rpar,
 *                   lenpmx,
#ifdef ILUFact
 *                   plu, jlu, ju, perm, qperm, rscale, cscale,
#elif defined ICFact
 *                   plu, jlu, perm, qperm, scale,
#endif
 *                   rwork, lrw,
 *                   ier_ksp, ier_input);
 *
 *****************************************************************************/
#ifdef _CRAYMPP
#define KSP_DRIVER KSP_DRIVER
#else
#define KSP_DRIVER ksp_driver__
#endif

#ifdef ILUFact
#define CALL_KSP_DRIVER(n, nnz,\
                         ia, ja, a, rhs, sol,\
                         ipar, rpar,\
                         lenpmx,\
                         plu, jlu, ju, perm, qperm, rscale, cscale, \
                         rwork, lrw,\
                         ier_ksp, ier_input) \
             KSP_DRIVER(&n, &nnz,\
                         ia, ja, a, rhs, sol,\
                         ipar, rpar,\
                         &lenpmx,\
                         plu, jlu, ju, perm, qperm, rscale, cscale, \
                         rwork, &lrw,\
                         &ier_ksp, &ier_input)
#elif defined ICFact
#define CALL_KSP_DRIVER(n, nnz,\
                         ia, ja, a, rhs, sol,\
                         ipar, rpar,\
                         lenpmx,\
                         plu, jlu, perm, qperm, scale, \
                         rwork, lrw,\
                         ier_ksp, ier_input) \
             KSP_DRIVER(&n, &nnz,\
                         ia, ja, a, rhs, sol,\
                         ipar, rpar,\
                         &lenpmx,\
                         plu, jlu, perm, qperm, scale, \
                         rwork, &lrw,\
                         &ier_ksp, &ier_input)
#endif

void KSP_DRIVER(HYPRE_Int *n, HYPRE_Int *nnz,
                 HYPRE_Int *ia, HYPRE_Int *ja, double *a, double *rhs, double *sol,
                 HYPRE_Int *ipar, double *rpar,
                 HYPRE_Int *lenpmx,
                 double *plu, HYPRE_Int *jlu, 
#ifdef ILUFact
                 HYPRE_Int *ju,
#endif 
                 HYPRE_Int *perm, HYPRE_Int *qperm, 
#ifdef ILUFact
                 double *rscale, double *cscale,
#endif 
#ifdef ICFact
                 double *scale,
#endif 
                 double *rwork, HYPRE_Int *lrw,
                 HYPRE_Int *ier_ksp, HYPRE_Int *ier_input);


/*****************************************************************************
 * C routine would use this macro to call "dvperm" :
 *
 *  CALL_DVPERM(n, x, perm);
 *
 *****************************************************************************/
#ifdef _CRAYMPP
#define DVPERM DVPERM
#else
#define DVPERM dvperm_
#endif

#define CALL_DVPERM(n, x, perm) \
             DVPERM(&n, x, perm)

void DVPERM(HYPRE_Int *n, double *x, HYPRE_Int *perm);

/*****************************************************************************
 * C routine would use this macro to call "csrssr" :
 *
 *  CALL_CSRSSR(n, a, ja, ia, nzmax, ao, jao, iao, ierr);
 *
 *****************************************************************************/
#ifdef _CRAYMPP
#define CSRSSR CSRSSR
#else
#define CSRSSR csrssr_
#endif

#define CALL_CSRSSR(n, a, ja, ia, nzmax, ao, jao, iao, ierr ) \
             CSRSSR(&n, a, ja, ia, &nzmax, ao, jao, iao, &ierr )

void CSRSSR(HYPRE_Int *n, double *a, HYPRE_Int *ja, HYPRE_Int *ia, HYPRE_Int *nzmax,
            double *ao, HYPRE_Int *jao, HYPRE_Int *iao, HYPRE_Int *ierr );
