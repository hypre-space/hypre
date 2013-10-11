/*--------------------------------------------------------------------------
 * Author:        Andy Cleary
 *                Centre for Applied Scientific Computing
 *                Lawrence Livermore Labs
 * Revision History:
 *  5-22-97: AJC. Original Version.
 *--------------------------------------------------------------------------*/


#define ICFact

/*****************************************************************************
 * C to Fortran interfacing macros
 *****************************************************************************/

/*****************************************************************************
 * C routine would use this macro to call "ict_driver.f" :
 *
 *  CALL_ICT_DRIVER(n, nnz,
 *                   ia, ja, a,
 *                   ipar, rpar,
 *                   lenpmx,
#ifdef ILUFact
 *                   plu, jlu, ju, perm, qperm, rscale, cscale,
#elif defined ICFact
 *                   plu, jlu, perm, qperm, scale,
#endif
 *                   iwork, liw, rwork, lrw,
 *                   ier_ict, ier_input);
 *
 *****************************************************************************/
#ifdef _CRAYMPP
#define ICT_DRIVER ICT_DRIVER
#else
#define ICT_DRIVER ict_driver__
#endif

#ifdef ILUFact
#define CALL_ICT_DRIVER(n, nnz,\
                         ia, ja, a, \
                         ipar, rpar,\
                         lenpmx,\
                         plu, jlu, ju, perm, qperm, rscale, cscale, \
                         iwork, liw, rwork, lrw,\
                         ier_ict, ier_input) \
             ICT_DRIVER(&n, &nnz,\
                         ia, ja, a, \
                         ipar, rpar,\
                         &lenpmx,\
                         plu, jlu, ju, perm, qperm, rscale, cscale, \
                         iwork, &liw, rwork, &lrw,\
                         &ier_ict, &ier_input)
#elif defined ICFact
#define CALL_ICT_DRIVER(n, nnz,\
                         ia, ja, a, \
                         ipar, rpar,\
                         lenpmx,\
                         plu, jlu, perm, qperm, scale, \
                         iwork, liw, rwork, lrw,\
                         ier_ict, ier_input) \
             ICT_DRIVER(&n, &nnz,\
                         ia, ja, a, \
                         ipar, rpar,\
                         &lenpmx,\
                         plu, jlu, perm, qperm, scale, \
                         iwork, &liw, rwork, &lrw,\
                         &ier_ict, &ier_input)
#endif

void ICT_DRIVER(HYPRE_Int *n, HYPRE_Int *nnz,
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
                 HYPRE_Int *ier_ict, HYPRE_Int *ier_input);



/*****************************************************************************
 * C routine would use this macro to call "cg_driver.f" :
 *
 *  CALL_CG_DRIVER(n, nnz,
 *                   ia, ja, a, rhs, sol,
 *                   ipar, rpar,
 *                   lenpmx,
#ifdef ILUFact
 *                   plu, jlu, ju, perm, qperm, rscale, cscale,
#elif defined ICFact
 *                   plu, jlu, perm, qperm, scale,
#endif
 *                   rwork, lrw,
 *                   ier_cg, ier_input);
 *
 *****************************************************************************/
#ifdef _CRAYMPP
#define CG_DRIVER CG_DRIVER
#else
#define CG_DRIVER cg_driver__
#endif

#ifdef ILUFact
#define CALL_CG_DRIVER(n, nnz,\
                         ia, ja, a, rhs, sol,\
                         ipar, rpar,\
                         lenpmx,\
                         plu, jlu, ju, perm, qperm, rscale, cscale, \
                         rwork, lrw,\
                         ier_cg, ier_input) \
             CG_DRIVER(&n, &nnz,\
                         ia, ja, a, rhs, sol,\
                         ipar, rpar,\
                         &lenpmx,\
                         plu, jlu, ju, perm, qperm, rscale, cscale, \
                         rwork, &lrw,\
                         &ier_cg, &ier_input)
#elif defined ICFact
#define CALL_CG_DRIVER(n, nnz,\
                         ia, ja, a, rhs, sol,\
                         ipar, rpar,\
                         lenpmx,\
                         plu, jlu, perm, qperm, scale, \
                         rwork, lrw,\
                         ier_cg, ier_input) \
             CG_DRIVER(&n, &nnz,\
                         ia, ja, a, rhs, sol,\
                         ipar, rpar,\
                         &lenpmx,\
                         plu, jlu, perm, qperm, scale, \
                         rwork, &lrw,\
                         &ier_cg, &ier_input)
#endif

void CG_DRIVER(HYPRE_Int *n, HYPRE_Int *nnz,
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
                 HYPRE_Int *ier_cg, HYPRE_Int *ier_input);


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
