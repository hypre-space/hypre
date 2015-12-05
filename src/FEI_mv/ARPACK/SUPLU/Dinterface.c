#include "dsp_defs.h"
#include "util.h"

/* global variable */

typedef struct Dnode {
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
} Dnode_type;

#define maxm 5

Dnode_type Dtoken_list[maxm];

extern SuperLUStat_t SuperLUStat;
extern int dsparse_preprocess_(int *, int *, int *, int *, double *, int *);
extern int dsparse_factor_(int *);
extern int dsparse_solve_(int *, double *, double *);
extern int dsparse_destroy(int *);

/* ------------------ Matrix ordering ----------------------------*/

int dsparse_preprocess_(int *token,   int *n,         int *pointers,
                        int *indices, double *values, int *order_type)
{
   int  nnz,        info;
   char refac[1];

   *refac = 'N';
   if (*token < 0 || *token > maxm) {
      fprintf(stderr," Invalid token number! token = %d\n", *token);
      info = -1;
   }
   else {
      nnz = pointers[*n];

      dCreate_CompCol_Matrix(&(Dtoken_list[*token].A), *n,     *n, 
                             nnz,                      values, indices, 
                             pointers,                 NC,     _D,
                             GE);
      if (!(Dtoken_list[*token].etree = intMalloc((Dtoken_list[*token].A).ncol)))
         ABORT("Malloc fails for etree[]."); 
      if ( !(Dtoken_list[*token].perm_c = intMalloc(*n)) ) 
         ABORT("Malloc fails for perm_c.");
      if ( !(Dtoken_list[*token].perm_r = intMalloc(*n)) ) 
         ABORT("Malloc fails for perm_r.");

      Dtoken_list[*token].diag_pivot_thresh = 1.0;
      Dtoken_list[*token].drop_tol = 0.0;
      Dtoken_list[*token].panel_size = sp_ienv(1);
      Dtoken_list[*token].relax = sp_ienv(2);

      StatInit(Dtoken_list[*token].panel_size, Dtoken_list[*token].relax);

      get_perm_c(*order_type,                &(Dtoken_list[*token].A), 
                 Dtoken_list[*token].perm_c);

      sp_preorder(refac,                      &(Dtoken_list[*token].A), 
                  Dtoken_list[*token].perm_c, Dtoken_list[*token].etree, 
                  &(Dtoken_list[*token].AC));

      info = 0;
   }
   return info;
}

/*------------------- Numerical factorization ----------------------*/

int dsparse_factor_(int *token)
{
   char  refac[1];
   int   lwork = 0, info;

   *refac = 'N';
   if (*token < 0 || *token > maxm) {
      fprintf(stderr," Invalid token number! token = %d\n", *token);
      info = -1;
   }
   else {
      dgstrf(refac, &(Dtoken_list[*token].AC), 
             Dtoken_list[*token].diag_pivot_thresh,
             Dtoken_list[*token].drop_tol, 
             Dtoken_list[*token].relax, 
             Dtoken_list[*token].panel_size,
             Dtoken_list[*token].etree,
             NULL, lwork, Dtoken_list[*token].perm_r, 
             Dtoken_list[*token].perm_c,
             &(Dtoken_list[*token].L), &(Dtoken_list[*token].U), 
             &(Dtoken_list[*token].info));
      info = 0;
   }
   return info;      
}

/* --------------------- Triangular Solves ---------------------*/

int dsparse_solve_(int *token, double *x, double *rhs)
{
   char*       trans="N";
   int         info, n, i;
   SuperMatrix B;

   if (*token<0 || *token > maxm) {
      fprintf(stderr," Invalid token number! token = %d\n", *token);
      info = -1;
   }
   else {
      n = (Dtoken_list[*token].A).ncol;
      for (i=0;i<n;i++) x[i] = rhs[i];
      dCreate_Dense_Matrix(&B, n, 1, x, n, DN, _D, GE);
      dgstrs(trans, &(Dtoken_list[*token].L), 
             &(Dtoken_list[*token].U),
             Dtoken_list[*token].perm_r, 
             Dtoken_list[*token].perm_c,
             &B, &Dtoken_list[*token].info);
      info = 0;
   }
   return info;
}
/* ----------------------- Clean up and deallocate -----------------*/

int dsparse_destroy_(int *token)
{
   int info;

   if (*token<0 || *token > maxm) {
      fprintf(stderr," Invalid token number! token = %d\n", *token);
      info = -1;
   }
   else {
         SUPERLU_FREE(Dtoken_list[*token].perm_c);
         SUPERLU_FREE(Dtoken_list[*token].perm_r);
         SUPERLU_FREE(Dtoken_list[*token].etree);
         Destroy_SuperMatrix_Store(&(Dtoken_list[*token].A));
         Destroy_CompCol_Permuted(&(Dtoken_list[*token].AC));
         Destroy_SuperNode_Matrix(&(Dtoken_list[*token].L));
         Destroy_CompCol_Matrix(&(Dtoken_list[*token].U));
         StatFree();
         info = 0;
   }
   return info;
}
