/*--------------------------------------------------------------------------
 * Purpose:       Define data structure and access macros for incomplete
 *                factorization data, including the matrix, the factors of
 *                the matrix, and parameters used in the factorization and
 *                solve.

 * Author:        Andy Cleary
 *                Centre for Applied Scientific Computing
 *                Lawrence Livermore Labs
 * Revision History:
 *  5-22-97: AJC. Original Version.
 *--------------------------------------------------------------------------*/

#ifndef _ICDATA_HEADER
#define _ICDATA_HEADER

#define ICFact

#ifdef ICFact
  #define rscale scale
#endif

#include "matrix.h"
/*--------------------------------------------------------------------------
 * ic_data
 *--------------------------------------------------------------------------*/

typedef struct
{

  int             mode;
  Matrix         *A;
  int             ipar[20];
  double          rpar[20];
  Matrix         *preconditioner;
  int             lenpmx;
  int            *perm;
  int            *inverse_perm;
  double         *rscale;
#ifdef ILUFact
  double         *cscale;
#endif
  int            *iwork;
  int             l_iwork;
  double         *rwork;
  int             l_rwork;
  
} ICData;

/*--------------------------------------------------------------------------
 * Accessor functions for the ic_data structure
 *--------------------------------------------------------------------------*/
#define ICDataMode(ic_data)               ((ic_data) -> mode)
#define ICDataA(ic_data)                  ((ic_data) -> A)
#define ICDataIpar(ic_data)               ((ic_data) -> ipar)
#define ICDataRpar(ic_data)               ((ic_data) -> rpar)
#define ICDataPreconditioner(ic_data)     ((ic_data) -> preconditioner)
#define ICDataLenpmx(ic_data)             ((ic_data) -> lenpmx)
#define ICDataPerm(ic_data)               ((ic_data) -> perm)
#define ICDataInversePerm(ic_data)        ((ic_data) -> inverse_perm)
#ifdef ILUFact
  #define ICDataRscale(ic_data)             ((ic_data) -> rscale)
  #define ICDataCscale(ic_data)             ((ic_data) -> cscale)
#else
  #define ICDataScale(ic_data)              ((ic_data) -> scale)
  #define ICDataRscale(ic_data)             ICDataScale(ic_data)
  #define ICDataCscale(ic_data)             ICDataScale(ic_data)
#endif
#define ICDataIwork(ic_data)              ((ic_data) -> iwork)
#define ICDataLIwork(ic_data)             ((ic_data) -> l_iwork)
#define ICDataRwork(ic_data)              ((ic_data) -> rwork)
#define ICDataLRwork(ic_data)             ((ic_data) -> l_rwork)


#endif
