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

void ILUT_DRIVER(int *n, int *nnz,
                 int *ia, int *ja, double *a,
                 int *ipar, double *rpar,
                 int *lenpmx,
                 double *plu, int *jlu, 
#ifdef ILUFact
                 int *ju, 
#endif
                 int *perm, int *qperm, 
#ifdef ILUFact
                 double *rscale, double *cscale, 
#endif
#ifdef ICFact
                 double *scale,
#endif
                 int *iwork, int *liw, double *rwork, int *lrw,
                 int *ier_ilut, int *ier_input);



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

void GMRES_DRIVER(int *n, int *nnz,
                 int *ia, int *ja, double *a, double *rhs, double *sol,
                 int *ipar, double *rpar,
                 int *lenpmx,
                 double *plu, int *jlu, 
#ifdef ILUFact
                 int *ju,
#endif 
                 int *perm, int *qperm, 
#ifdef ILUFact
                 double *rscale, double *cscale,
#endif 
#ifdef ICFact
                 double *scale,
#endif 
                 double *rwork, int *lrw,
                 int *ier_gmres, int *ier_input);


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

void DVPERM(int *n, double *x, int *perm);

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

void CSRSSR(int *n, double *a, int *ja, int *ia, int *nzmax,
            double *ao, int *jao, int *iao, int *ierr );
