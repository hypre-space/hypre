#include "zsp_defs.h"
#include "util.h"

/* global variable */

typedef struct Znode {
   SuperMatrix A;
   SuperMatrix AC;
   int *perm_c;
   int *perm_r;
   int *etree;
   double diag_pivot_thresh;
   double drop_tol;
   int relax;
   int panel_size;
   SuperMatrix L;
   SuperMatrix U;
   int info;
} Znode_type;

#define maxm 5

Znode_type Ztoken_list[maxm];

extern SuperLUStat_t SuperLUStat;
extern int zsparse_preprocess_(int *, int *, int *, int *, doublecomplex *, int *);
extern int zsparse_factor_(int *);
extern int zsparse_solve_(int *, doublecomplex *, doublecomplex *);
extern int zsparse_destroy(int *);

/* ------------------ Matrix ordering ----------------------------*/
int zsparse_preprocess_(int *token,   int *n,                int *pointers,
                        int *indices, doublecomplex *values, int *order_type)
{
   int  nnz,        info;
   char refac[1];

   *refac='N';
   if (*token < 0 || *token > maxm) {
      fprintf(stderr," Invalid token number! token = %d\n", *token);
      info = -1;
   }
   else {
      nnz = pointers[*n];

      zCreate_CompCol_Matrix(&(Ztoken_list[*token].A), *n,     *n, 
                             nnz,                      values, indices, 
                             pointers,                 NC,     _Z,
                             GE);
      if (!(Ztoken_list[*token].etree = intMalloc((Ztoken_list[*token].A).ncol)))
         ABORT("Malloc fails for etree[]."); 
      if ( !(Ztoken_list[*token].perm_c = (int*)intMalloc(*n)) ) 
         ABORT("Malloc fails for perm_c.");
      if ( !(Ztoken_list[*token].perm_r = (int*)intMalloc(*n)) ) 
         ABORT("Malloc fails for perm_r.");

      Ztoken_list[*token].diag_pivot_thresh = 1.0;
      Ztoken_list[*token].drop_tol = 0.0;
      Ztoken_list[*token].panel_size = sp_ienv(1);
      Ztoken_list[*token].relax = sp_ienv(2);

      StatInit(Ztoken_list[*token].panel_size, Ztoken_list[*token].relax);

      get_perm_c(*order_type,                &(Ztoken_list[*token].A), 
                 Ztoken_list[*token].perm_c);

      sp_preorder(refac,                      &(Ztoken_list[*token].A), 
                  Ztoken_list[*token].perm_c, Ztoken_list[*token].etree, 
                  &(Ztoken_list[*token].AC));

      info = 0;
   }
   return info;
}


/*------------------- Numerical factorization ----------------------*/

int zsparse_factor_(int *token)
{
   char  refac[1];
   int   lwork = 0, info;

   *refac = 'N';
   if (*token < 0 || *token > maxm) {
      fprintf(stderr," Invalid token number! token = %d\n", *token);
      info = -1;
   }
   else {

      zgstrf(refac, &(Ztoken_list[*token].AC), 
             Ztoken_list[*token].diag_pivot_thresh,
             Ztoken_list[*token].drop_tol, 
             Ztoken_list[*token].relax, 
             Ztoken_list[*token].panel_size,
             Ztoken_list[*token].etree,
             NULL, lwork, Ztoken_list[*token].perm_r, 
             Ztoken_list[*token].perm_c,
             &(Ztoken_list[*token].L), &(Ztoken_list[*token].U), 
             &(Ztoken_list[*token].info));
      if (Ztoken_list[*token].info != 0) {
         fprintf(stderr, " Factorization failed, info = %d\n",
                 Ztoken_list[*token].info);
         info = -1;
      }
      else {
         info = 0;
      }
   }
   return info;      
}

/* --------------------- Triangular Solves ---------------------*/

int zsparse_solve_(int *token, doublecomplex *x, doublecomplex *rhs)
{
   char*       trans="N";
   int         info, n, i;
   SuperMatrix B;

   if (*token<0 || *token > maxm) {
      fprintf(stderr," Invalid token number! token = %d\n", *token);
      info = -1;
   }
   else {
      n = (Ztoken_list[*token].A).ncol;
      for (i=0;i<n;i++) x[i] = rhs[i];
      zCreate_Dense_Matrix(&B, n, 1, x, n, DN, _Z, GE);
      zgstrs(trans, &(Ztoken_list[*token].L), 
             &(Ztoken_list[*token].U),
             Ztoken_list[*token].perm_r, 
             Ztoken_list[*token].perm_c,
             &B, &Ztoken_list[*token].info);
      if (Ztoken_list[*token].info != 0) {
         fprintf(stderr, " Triangular solve failed, info = %d\n",
                 Ztoken_list[*token].info);
         info = -1;
      }
      else { 
         info = 0;
      }
   }
   return info;
}

/* ----------------------- Clean up and deallocate -----------------*/

int zsparse_destroy_(int *token)
{
   int info;

   if (*token<0 || *token > maxm) {
      fprintf(stderr," Invalid token number! token = %d\n", *token);
      info = -1;
   }
   else {
         SUPERLU_FREE(Ztoken_list[*token].perm_c);
         SUPERLU_FREE(Ztoken_list[*token].perm_r);
         SUPERLU_FREE(Ztoken_list[*token].etree);
         Destroy_SuperMatrix_Store(&(Ztoken_list[*token].A));
         Destroy_CompCol_Permuted(&(Ztoken_list[*token].AC));
         Destroy_SuperNode_Matrix(&(Ztoken_list[*token].L));
         Destroy_CompCol_Matrix(&(Ztoken_list[*token].U));
         StatFree();
         info = 0;
   }
   return info;
}
