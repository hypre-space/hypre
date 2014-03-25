/*--------------------------------------------------------------------------
 * Author:        Andy Cleary
 *                Centre for Applied Scientific Computing
 *                Lawrence Livermore Labs
 * Revision History:
 *  5-22-97: AJC. Original Version.
 *--------------------------------------------------------------------------*/


#define ILUFact

/*****************************************************************************
 * C to Fortran interfacing macros
 *****************************************************************************/

/*****************************************************************************
 * C routine would use this macro to call "ilut_driver.f" :
 *
 *  CALL_ILUT_DRIVER(n, nnz,
 *                   ia, ja, a,
 *                   ipar, rpar,
 *                   lenpmx,
#ifdef ILUFact
 *                   plu, jlu, ju, perm, qperm, rscale, cscale,
#elif defined ICFact
 *                   plu, jlu, perm, qperm, scale,
#endif
 *                   iwork, liw, rwork, lrw,
 *                   ier_ilut, ier_input);
 *
 *****************************************************************************/
#ifdef _CRAYMPP
#define ILUT_DRIVER ILUT_DRIVER
#else
#define ILUT_DRIVER ilut_driver__
#endif

#ifdef ILUFact
#define CALL_ILUT_DRIVER(n, nnz,\
                         ia, ja, a, \
                         ipar, rpar,\
                         lenpmx,\
                         plu, jlu, ju, perm, qperm, rscale, cscale, \
                         iwork, liw, rwork, lrw,\
                         ier_ilut, ier_input) \
             ILUT_DRIVER(&n, &nnz,\
                         ia, ja, a, \
                         ipar, rpar,\
                         &lenpmx,\
                         plu, jlu, ju, perm, qperm, rscale, cscale, \
                         iwork, &liw, rwork, &lrw,\
                         &ier_ilut, &ier_input)
#elif defined ICFact
#define CALL_ILUT_DRIVER(n, nnz,\
                         ia, ja, a, \
                         ipar, rpar,\
                         lenpmx,\
                         plu, jlu, perm, qperm, scale, \
                         iwork, liw, rwork, lrw,\
                         ier_ilut, ier_input) \
             ILUT_DRIVER(&n, &nnz,\
                         ia, ja, a, \
                         ipar, rpar,\
                         &lenpmx,\
                         plu, jlu, perm, qperm, scale, \
                         iwork, &liw, rwork, &lrw,\
                         &ier_ilut, &ier_input)
#endif

void ILUT_DRIVER(HYPRE_Int *n, HYPRE_Int *nnz,
                 HYPRE_Int *ia, HYPRE_Int *ja, HYPRE_Real *a,
                 HYPRE_Int *ipar, HYPRE_Real *rpar,
                 HYPRE_Int *lenpmx,
                 HYPRE_Real *plu, HYPRE_Int *jlu, 
#ifdef ILUFact
                 HYPRE_Int *ju, 
#endif
                 HYPRE_Int *perm, HYPRE_Int *qperm, 
#ifdef ILUFact
                 HYPRE_Real *rscale, HYPRE_Real *cscale, 
#endif
#ifdef ICFact
                 HYPRE_Real *scale,
#endif
                 HYPRE_Int *iwork, HYPRE_Int *liw, HYPRE_Real *rwork, HYPRE_Int *lrw,
                 HYPRE_Int *ier_ilut, HYPRE_Int *ier_input);



/*****************************************************************************
 * C routine would use this macro to call "gmres_driver.f" :
 *
 *  CALL_GMRES_DRIVER(n, nnz,
 *                   ia, ja, a, rhs, sol,
 *                   ipar, rpar,
 *                   lenpmx,
#ifdef ILUFact
 *                   plu, jlu, ju, perm, qperm, rscale, cscale,
#elif defined ICFact
 *                   plu, jlu, perm, qperm, scale,
#endif
 *                   rwork, lrw,
 *                   ier_gmres, ier_input);
 *
 *****************************************************************************/
#ifdef _CRAYMPP
#define GMRES_DRIVER GMRES_DRIVER
#else
#define GMRES_DRIVER gmres_driver__
#endif

#ifdef ILUFact
#define CALL_GMRES_DRIVER(n, nnz,\
                         ia, ja, a, rhs, sol,\
                         ipar, rpar,\
                         lenpmx,\
                         plu, jlu, ju, perm, qperm, rscale, cscale, \
                         rwork, lrw,\
                         ier_gmres, ier_input) \
             GMRES_DRIVER(&n, &nnz,\
                         ia, ja, a, rhs, sol,\
                         ipar, rpar,\
                         &lenpmx,\
                         plu, jlu, ju, perm, qperm, rscale, cscale, \
                         rwork, &lrw,\
                         &ier_gmres, &ier_input)
#elif defined ICFact
#define CALL_GMRES_DRIVER(n, nnz,\
                         ia, ja, a, rhs, sol,\
                         ipar, rpar,\
                         lenpmx,\
                         plu, jlu, perm, qperm, scale, \
                         rwork, lrw,\
                         ier_gmres, ier_input) \
             GMRES_DRIVER(&n, &nnz,\
                         ia, ja, a, rhs, sol,\
                         ipar, rpar,\
                         &lenpmx,\
                         plu, jlu, perm, qperm, scale, \
                         rwork, &lrw,\
                         &ier_gmres, &ier_input)
#endif

void GMRES_DRIVER(HYPRE_Int *n, HYPRE_Int *nnz,
                 HYPRE_Int *ia, HYPRE_Int *ja, HYPRE_Real *a, HYPRE_Real *rhs, HYPRE_Real *sol,
                 HYPRE_Int *ipar, HYPRE_Real *rpar,
                 HYPRE_Int *lenpmx,
                 HYPRE_Real *plu, HYPRE_Int *jlu, 
#ifdef ILUFact
                 HYPRE_Int *ju,
#endif 
                 HYPRE_Int *perm, HYPRE_Int *qperm, 
#ifdef ILUFact
                 HYPRE_Real *rscale, HYPRE_Real *cscale,
#endif 
#ifdef ICFact
                 HYPRE_Real *scale,
#endif 
                 HYPRE_Real *rwork, HYPRE_Int *lrw,
                 HYPRE_Int *ier_gmres, HYPRE_Int *ier_input);


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

void DVPERM(HYPRE_Int *n, HYPRE_Real *x, HYPRE_Int *perm);

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

void CSRSSR(HYPRE_Int *n, HYPRE_Real *a, HYPRE_Int *ja, HYPRE_Int *ia, HYPRE_Int *nzmax,
            HYPRE_Real *ao, HYPRE_Int *jao, HYPRE_Int *iao, HYPRE_Int *ierr );
