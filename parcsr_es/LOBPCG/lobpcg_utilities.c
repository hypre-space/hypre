/*BHEADER**********************************************************************
 * lobpcg_utilities.c
 *
 * $Revision$
 * Date: 10/7/2002
 * Authors: M. Argentati and A. Knyazev
 *********************************************************************EHEADER*/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include <malloc.h>
#include <ctype.h>

#include "lobpcg.h"
#include "mmio.h"


/***************************************************************************/
/* reading a matrix from a file in ija format (first row : nrows, nnz)     */
/* (read by a single processor)                                            */
/* Store and assemble in a hypre ij matrix                                 */
/*-------------------------------------------------------------------------*/
void HYPRE_Load_IJAMatrix(Matx *A, int matrix_input_type, char *matfile,int *partitioning)
{
   int    mypid, nprocs, *ia, *ja, nrows, ncols, nnz, mybegin, myend;
   int    i, local_nrows, *row_sizes, ncnt;
   double *val;
   HYPRE_IJMatrix IJA;

   MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

   /*------------------------------------------------------------------
    * read the matrix and broadcast
    *----------------------------------------------------------------*/

   if ( mypid == 0 ) 
   {
      if (matrix_input_type==MATRIX_INPUT_PLANE_IJ){
         Get_IJAMatrixFromFileStandard(&val, &ia, &ja, &nrows, matfile);
         ncols=nrows; /* assume square in this case */
         }
      else if (matrix_input_type==MATRIX_INPUT_MTX){
         Get_IJAMatrixFromFileMtx(&val, &ia, &ja, &nrows, &ncols, matfile);
         if (nrows != ncols){
            fprintf(stderr, "Error: non-square matrix (filename=%s).\n",matfile);         
            exit(1);
         }
      }
      else {
         fprintf(stderr, "Bad matrix input type (filename=%s).\n",matfile);
         exit(1);
      }

      nnz = ia[nrows];
      MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&nnz,   1, MPI_INT, 0, MPI_COMM_WORLD);

      MPI_Bcast(ia,  nrows+1, MPI_INT,    0, MPI_COMM_WORLD);
      MPI_Bcast(ja,  nnz,     MPI_INT,    0, MPI_COMM_WORLD);
      MPI_Bcast(val, nnz,     MPI_DOUBLE, 0, MPI_COMM_WORLD);
   } 
   else 
   {
      MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&nnz,   1, MPI_INT, 0, MPI_COMM_WORLD);

      if ((ia=(int *) malloc((nrows+1)*sizeof(int)))==NULL){
        fprintf(stderr, "Could not allocate memory.\n");
        abort();
      }
      if ((ja=(int *) malloc(nnz*sizeof(int)))==NULL){
        fprintf(stderr, "Could not allocate memory.\n");
        abort();
      }
      if ((val=(double *) malloc(nnz*sizeof(double)))==NULL){
        fprintf(stderr, "Could not allocate memory.\n");
        abort();
      }

      MPI_Bcast(ia,  nrows+1, MPI_INT,    0, MPI_COMM_WORLD);
      MPI_Bcast(ja,  nnz,     MPI_INT,    0, MPI_COMM_WORLD);
      MPI_Bcast(val, nnz,     MPI_DOUBLE, 0, MPI_COMM_WORLD);
   }

   mybegin=partitioning[mypid];
   myend=partitioning[mypid+1]-1;

   printf("Processor %d : begin/end = %d %d\n", mypid, mybegin, myend);
   fflush(stdout);

   /*------------------------------------------------------------------
    * create matrix in the HYPRE context
    *----------------------------------------------------------------*/

   local_nrows = myend - mybegin + 1;
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, mybegin, myend, 
           mybegin, myend, &IJA);

    HYPRE_IJMatrixSetObjectType(IJA, HYPRE_PARCSR);
   

   row_sizes = (int *) malloc( local_nrows * sizeof(int) );
   for ( i = mybegin; i <= myend; i++ )
      row_sizes[i-mybegin] = ia[i+1] - ia[i];

   HYPRE_IJMatrixSetRowSizes(IJA, row_sizes);
    HYPRE_IJMatrixInitialize(IJA);
   
   free( row_sizes );

   for ( i = mybegin; i <= myend; i++ ) 
   {
      ncnt = ia[i+1] - ia[i];
      HYPRE_IJMatrixSetValues(IJA, 1, &ncnt, (const int *) &i, 
            (const int *) &(ja[ia[i]]), (const double *) &(val[ia[i]]));
      
   }
   HYPRE_IJMatrixAssemble(IJA);
   

   HYPRE_IJMatrixGetObject(IJA, (void **) &A->MPar);
   

   /* update A matrix parameters */
   A->m=nrows;
   A->n=ncols;
   A->nz=nnz;


   /*------------------------------------------------------------------
    * clean up
    *----------------------------------------------------------------*/

   free( ia );
   free( ja );
   free( val );
}

/***************************************************************************/
/* reading a matrix from a file in ija format (first row : nrows, nnz)     */
/* (read by a single processor)                                            */
/* Store and assemble in a hypre ij matrix                                 */
/*-------------------------------------------------------------------------*/
void HYPRE_Load_IJAMatrix2(HYPRE_ParCSRMatrix  *A_ptr,
     int matrix_input_type, char *matfile,int *partitioning)
{
   int    mypid, nprocs, *ia, *ja, nrows, ncols, nnz, mybegin, myend;
   int    i, local_nrows, *row_sizes, ncnt;
   double *val;
   HYPRE_IJMatrix IJA;

   MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

   /*------------------------------------------------------------------
    * read the matrix and broadcast
    *----------------------------------------------------------------*/

   if ( mypid == 0 )
   {
      if (matrix_input_type==MATRIX_INPUT_PLANE_IJ){
         Get_IJAMatrixFromFileStandard(&val, &ia, &ja, &nrows, matfile);
         ncols=nrows; /* assume square in this case */
         }
      else if (matrix_input_type==MATRIX_INPUT_MTX){
         Get_IJAMatrixFromFileMtx(&val, &ia, &ja, &nrows, &ncols, matfile);
         if (nrows != ncols){
            fprintf(stderr, "Error: non-square matrix (filename=%s).\n",matfile);
            exit(1);
         }
      }
      else if (matrix_input_type==MATRIX_INPUT_BIN){
         Get_CRSMatrixFromFileBinary(&val, &ia, &ja, &nrows, &ncols, matfile);
         if (nrows != ncols){
            fprintf(stderr, "Error: non-square matrix (filename=%s).\n",matfile);
            exit(1);
         }
      }
      else {
         fprintf(stderr, "Bad matrix input type (filename=%s).\n",matfile);
         exit(1);
      }

      nnz = ia[nrows];
      MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&nnz,   1, MPI_INT, 0, MPI_COMM_WORLD);

      MPI_Bcast(ia,  nrows+1, MPI_INT,    0, MPI_COMM_WORLD);
      MPI_Bcast(ja,  nnz,     MPI_INT,    0, MPI_COMM_WORLD);
      MPI_Bcast(val, nnz,     MPI_DOUBLE, 0, MPI_COMM_WORLD);
   }
   else
   {
      MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&nnz,   1, MPI_INT, 0, MPI_COMM_WORLD);

      if ((ia=(int *) malloc((nrows+1)*sizeof(int)))==NULL){
        fprintf(stderr, "Could not allocate memory.\n");
        abort();
      }
      if ((ja=(int *) malloc(nnz*sizeof(int)))==NULL){
        fprintf(stderr, "Could not allocate memory.\n");
        abort();
      }
      if ((val=(double *) malloc(nnz*sizeof(double)))==NULL){
        fprintf(stderr, "Could not allocate memory.\n");
        abort();
      }

      MPI_Bcast(ia,  nrows+1, MPI_INT,    0, MPI_COMM_WORLD);
      MPI_Bcast(ja,  nnz,     MPI_INT,    0, MPI_COMM_WORLD);
      MPI_Bcast(val, nnz,     MPI_DOUBLE, 0, MPI_COMM_WORLD);
   }

   mybegin=partitioning[mypid];
   myend=partitioning[mypid+1]-1;

   fprintf(stderr, "Processor %d : begin/end = %d %d\n", mypid, mybegin, myend);
   fflush(stdout);

   /*------------------------------------------------------------------
    * create matrix in the HYPRE context
    *----------------------------------------------------------------*/
   local_nrows = myend - mybegin + 1;
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, mybegin, myend,
           mybegin, myend, &IJA);

    HYPRE_IJMatrixSetObjectType(IJA, HYPRE_PARCSR);
   

   row_sizes = (int *) malloc( local_nrows * sizeof(int) );
   for ( i = mybegin; i <= myend; i++ )
      row_sizes[i-mybegin] = ia[i+1] - ia[i];

   HYPRE_IJMatrixSetRowSizes(IJA, row_sizes);
    HYPRE_IJMatrixInitialize(IJA);
   
   free( row_sizes );

   for ( i = mybegin; i <= myend; i++ )
   {
      ncnt = ia[i+1] - ia[i];
      HYPRE_IJMatrixSetValues(IJA, 1, &ncnt, (const int *) &i,
            (const int *) &(ja[ia[i]]), (const double *) &(val[ia[i]]));
      
   }
   HYPRE_IJMatrixAssemble(IJA);
   

   HYPRE_IJMatrixGetObject(IJA, (void **) A_ptr);
   

   /*------------------------------------------------------------------
    * clean up
    *----------------------------------------------------------------*/

   free( ia );
   free( ja );
   free( val );
}

/***************************************************************************/
/* Read a square matrix from a file in ija format (first row : nrows, nnz) */
/* (read by a single processor)                                            */
/*-------------------------------------------------------------------------*/
void Get_IJAMatrixFromFileStandard(double **val, int **ia, 
     int **ja, int *N, char *matfile)
{
    int    i, Nrows, nnz, icount, rowindex, colindex, curr_row;
    int    *mat_ia, *mat_ja;
    double *mat_a, value;
    FILE   *fp;
    /*------------------------------------------------------------------*/
    /* read matrix file                                                 */
    /*------------------------------------------------------------------*/
    printf("Reading standard matrix file = %s \n", matfile );
    fp = fopen( matfile, "r" );
    if ( fp == NULL ) {
       fprintf(stderr, "Error : file open error (filename=%s).\n", matfile);
       exit(1);
    }
    fscanf(fp, "%d %d", &Nrows, &nnz);
    if ( Nrows <= 0 || nnz <= 0 ) {
       fprintf(stderr, "Error : nrows,nnz = %d %d\n", Nrows, nnz);
       exit(1);
    }
    mat_ia = (int *) malloc((Nrows+1) * sizeof(int));
    mat_ja = (int *) malloc( nnz * sizeof(int));
    mat_a  = (double *) malloc( nnz * sizeof(double));
    mat_ia[0] = 0;
    curr_row = 0;
    icount   = 0;
    for ( i = 0; i < nnz; i++ ) {
       fscanf(fp, "%d %d %lg", &rowindex, &colindex, &value);
       rowindex--;
       colindex--;
       if ( rowindex != curr_row ) mat_ia[++curr_row] = icount;
       if ( rowindex < 0 || rowindex >= Nrows )
          fprintf(stderr, "Error reading row %d (curr_row = %d)\n", rowindex, curr_row);
       if ( colindex < 0 || colindex >= Nrows )
          fprintf(stderr, "Error reading col %d (rowindex = %d)\n", colindex, rowindex);
         /*if ( value != 0.0 ) {*/
          mat_ja[icount] = colindex;
          mat_a[icount++]  = value;
         /*}*/
    }
    fclose(fp);
    for ( i = curr_row+1; i <= Nrows; i++ ) mat_ia[i] = icount;
    (*val) = mat_a;
    (*ia)  = mat_ia;
    (*ja)  = mat_ja;
    (*N) = Nrows;
    printf("matrix has %6d rows and %7d nonzeros\n", Nrows, mat_ia[Nrows]);
    printf("returning from reading matrix from standard ijA file\n\n");
}

/***************************************************************************/
/* Read an m x n matrix from a matrix market file.                         */
/* (read by a single processor)                                            */
/*-------------------------------------------------------------------------*/
void Get_IJAMatrixFromFileMtx(double **val, int **ia, 
     int **ja, int *m, int *n, char *matfile)
{
    int    i,j,k,kk,Nrows, nnz, icount, rowindex, colindex, curr_row;
    int    m1,n1,off_diag_count=0;
    int    *mat_ia, *mat_ja;
    double *mat_a, value;
    MM_typecode matcode; /* Matrix Market type code */
    input_data *input1;
    FILE   *fp;

    /*------------------------------------------------------------------*/
    /* read matrix file                                                 */
    /*------------------------------------------------------------------*/
    printf("Reading matrix file = %s \n", matfile );
    fp = fopen( matfile, "r" );
    if ( fp == NULL ) {
       fprintf(stderr, "Error : file open error (filename=%s).\n", matfile);
       exit(1);
    }

    /* read Matrix Market Banner */
    if (mm_read_banner(fp,&matcode) != 0) {
       fprintf(stderr, "Could not process Matrix Market banner (filename=%s).\n",matfile);
       exit(1);
    }

    /* read in general sparse matrix */
    if (mm_is_sparse(matcode) && mm_is_general(matcode)){

       /* read array parameters for sparse matrix */
       if ((mm_read_mtx_crd_size(fp,&m1,&n1,&nnz)) !=0){
          fprintf(stderr, "Could not process Matrix Market parameters (filename=%s).\n",matfile);
          exit(1);
       }

       if ( m1 <= 0 || n1<=0  || nnz <= 0 ) {
          fprintf(stderr, "Error reading matrix market matrix: m1,n1,nnz = %d %d %d\n",
             m1,n1,nnz);
          exit(1);
       }
    
       /* allocate memory */
       if ((input1=(input_data *) malloc(nnz*sizeof(input_data)))==NULL){
          fprintf(stderr, "Could not allocate memory for input data.\n");
          exit(1);
       }

       /* read in values */
       for (i=0; i<nnz; i++) {
          fscanf(fp,"%le %le %le\n",&input1[i].row,&input1[i].col,&input1[i].val);
          input1[i].row--; input1[i].col--;  /* set index set starts at 0 */
          input1[i].index=input1[i].row*m1+input1[i].col;
       }
    }

    /* read in symmetric sparse matrix */
    else if (mm_is_sparse(matcode) && mm_is_symmetric(matcode)){

       /* read array parameters for sparse matrix */
       if ((mm_read_mtx_crd_size(fp,&m1,&n1,&nnz)) !=0){
          fprintf(stderr, "Could not process Matrix Market parameters (filename=%s).\n",matfile);
          exit(1);
       }

       if ( m1 <= 0 || n1<=0  || nnz <= 0 ) {
          fprintf(stderr, "Error reading matrix market matrix: m1,n1,nnz = %d %d %d\n",
             m1,n1,nnz);
          exit(1);
       }
       
       /* allocate memory */
       if ((input1=(input_data *) malloc(nnz*sizeof(input_data)))==NULL){
          fprintf(stderr, "Could not allocate memory for input data.\n");
          exit(1);
       }

       /* read in values */
       for (i=0; i<nnz; i++) {
          fscanf(fp,"%le %le %le\n",&input1[i].row,&input1[i].col,&input1[i].val);
          input1[i].row--; input1[i].col--;  /* set index set starts at 0 */
          input1[i].index=input1[i].row*m1+input1[i].col;
          if (input1[i].row!=input1[i].col) ++off_diag_count;
       }

       /* reallocate memory */
       if ((input1=(input_data *) realloc(input1,(nnz+off_diag_count)*sizeof(input_data)))==NULL){
          fprintf(stderr, "Could not allocate memory for input data.\n");
          exit(1);
       }

       /* add in elements to make symmetric */
       j=nnz;
       for (i=0; i<nnz; i++) {
          if (input1[i].row != input1[i].col){
             input1[j].row=input1[i].col;
             input1[j].col=input1[i].row;
             input1[j].val=input1[i].val;
             input1[j].index=input1[j].row*m1+input1[j].col;
             ++j;
          }
       }
       nnz =nnz+off_diag_count;
    }

    /* read in dense matrix */
    else if (mm_is_dense(matcode) && mm_is_general(matcode)){

       /* read array parameters for dense matrix */
       if ((mm_read_mtx_array_size(fp,&m1,&n1)) !=0){
          exit(1);
       }
       nnz=m1*n1;

       /* allocate memory */
       if ((input1=(input_data *) malloc(nnz*sizeof(input_data)))==NULL){
          if (Get_Rank()==0) fprintf(stderr, "Could not allocate memory for input data.\n");
       exit(1);
       }

       /* read in values */
       k=0;
       for (j=0; j<n1; j++){
          for (i=0; i<m1; i++){
          fscanf(fp,"%le\n",&input1[k].val);
          input1[k].row=i; input1[k].col=j;  /* set index set starts at 0 */
          input1[k].index=input1[k].row*m1+input1[k].col;
          k++;
          }
       }
    }

    /* read in dense matrix that is symmetric */
    else if (mm_is_dense(matcode) && mm_is_symmetric(matcode)){

       /* read array parameters for dense matrix */
       if ((mm_read_mtx_array_size(fp,&m1,&n1)) !=0){
          if (Get_Rank()==0) fprintf(stderr, "Could not process Matrix Market parameters.\n");
          exit(1);
       }
       nnz=m1*n1;

       /* allocate memory */
       if ((input1=(input_data *) malloc(2*nnz*sizeof(input_data)))==NULL){
          if (Get_Rank()==0) fprintf(stderr, "Could not allocate memory for input data.\n");
          exit(1);
       }

       /* read in values */
       k=0;
       for (j=0; j<n1; j++){
          for (i=j; i<m1; i++){
             fscanf(fp,"%le\n",&input1[k].val);
             input1[k].row=i; input1[k].col=j;  /* set index set starts at 0 */
             input1[k].index=input1[k].row*m1+input1[k].col;
             k++;
             }
          }

       /* fill in remaining values */
       kk=k;
       k=0;
       for (i=0; i<kk; i++) {
          if (input1[i].row > input1[i].col){
          j=kk+k;
          input1[j].row=input1[i].col;
          input1[j].col=input1[i].row;
          input1[j].val=input1[i].val;
          input1[j].index=input1[j].row*m1+input1[j].col;
          k++;
          }
       }
       assert(nnz!=k);
    }

    else {
       fprintf(stderr, "Can't processs Matrix Market format (filename=%s).\n",matfile);
       fclose(fp);
       exit(1);
    }

    fclose(fp);

    /* sort input data by rows and then columns */
    qsort(input1,nnz,sizeof(input_data),comp);

    Nrows=m1;
    mat_ia = (int *) malloc((Nrows+1) * sizeof(int));
    mat_ja = (int *) malloc( nnz * sizeof(int));
    mat_a  = (double *) malloc( nnz * sizeof(double));

    mat_ia[0] = 0;
    curr_row = 0;
    icount   = 0;
    for ( i = 0; i < nnz; i++ ) {
       rowindex= input1[i].row;
       colindex=input1[i].col;
       value=input1[i].val;
       if ( rowindex != curr_row ) mat_ia[++curr_row] = icount;
       if ( rowindex < 0 || rowindex >= Nrows )
          fprintf(stderr, "Error reading row %d (curr_row = %d)\n", rowindex, curr_row);
       if ( colindex < 0 || colindex >= Nrows )
          fprintf(stderr, "Error reading col %d (rowindex = %d)\n", colindex, rowindex);
       mat_ja[icount] = colindex;
       mat_a[icount++]  = value;
    }

    for ( i = curr_row+1; i <= Nrows; i++ ) mat_ia[i] = icount;
    (*val) = mat_a;
    (*ia)  = mat_ia;
    (*ja)  = mat_ja;
    (*m) = m1;
    (*n) = n1;

    free(input1);
   
    printf("matrix has %d rows, %d columns and %d nonzeros\n",
       Nrows,n1, mat_ia[Nrows]);
    printf("returning from reading matrix market file\n\n");
}

/***************************************************************************/
/* Load an array of hypre parallel vectors from a matrix market file       */
/* There are n vectors, each of length m. So this is like an m x n matrix. */
/* (read by a single processor)                                            */
/*-------------------------------------------------------------------------*/
void HYPRE_LoadMatrixVectorMtx(Matx *A,char *matfile,int *partitioning)
{
   int    mypid, nprocs, *ia, *ja, nrows, ncols, nnz, mybegin, myend;
   int    i,j;
   int rowindex;
   double *val,*tempvector;
   HYPRE_IJVector *vv;

   MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

   /*------------------------------------------------------------------
    * read the matrix and broadcast
    *----------------------------------------------------------------*/
   if ( mypid == 0 )
   {
      Get_IJAMatrixFromFileMtx(&val, &ia, &ja, &nrows, &ncols, matfile);
      nnz = ia[nrows];
      MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&nnz,   1, MPI_INT, 0, MPI_COMM_WORLD);

      MPI_Bcast(ia,  nrows+1, MPI_INT,    0, MPI_COMM_WORLD);
      MPI_Bcast(ja,  nnz,     MPI_INT,    0, MPI_COMM_WORLD);
      MPI_Bcast(val, nnz,     MPI_DOUBLE, 0, MPI_COMM_WORLD);
   }
   else
   {
      MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&nnz,   1, MPI_INT, 0, MPI_COMM_WORLD);

      if ((ia=(int *) malloc((nrows+1)*sizeof(int)))==NULL){
        fprintf(stderr, "Could not allocate memory.\n");
        abort();
      }
      if ((ja=(int *) malloc(nnz*sizeof(int)))==NULL){
        fprintf(stderr, "Could not allocate memory.\n");
        abort();
      }
      if ((val=(double *) malloc(nnz*sizeof(double)))==NULL){
        fprintf(stderr, "Could not allocate memory.\n");
        abort();
      }

      MPI_Bcast(ia,  nrows+1, MPI_INT,    0, MPI_COMM_WORLD);
      MPI_Bcast(ja,  nnz,     MPI_INT,    0, MPI_COMM_WORLD);
      MPI_Bcast(val, nnz,     MPI_DOUBLE, 0, MPI_COMM_WORLD);
   }

   /* Mat_Init needs this call */
   hypre_LobpcgSetGetPartition(0,&partitioning);
   Mat_Init(A,nrows,ncols,nnz,HYPRE_VECTORS,GENERAL);

   /* allocate storage */
   if ((tempvector=(double *) malloc(nrows*sizeof(double)))==NULL){
        fprintf(stderr, "Could not allocate memory.\n");
        abort();
   }
   if ((vv=(HYPRE_IJVector  *) malloc(ncols*sizeof(HYPRE_IJVector)))==NULL){
      fprintf(stderr, "Could not allocate memory.\n");
      abort();
   }

   mybegin=partitioning[mypid];
   myend=partitioning[mypid+1]-1;

   /*------------------------------------------------------------------
    * Store each vector. This is not efficent,
    * but is designed to deal with a small number of vectors.  
    *----------------------------------------------------------------*/
   for (j=0;j<ncols;++j){
      for(i=0;i<nrows;++i) tempvector[i]=0;
      for ( i = 0; i < nrows; i++ ) {
         for (rowindex=ia[i];rowindex<ia[i+1];++rowindex){
            if (ja[rowindex]==j) tempvector[i]=val[rowindex];     
         }
      }
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, mybegin, myend, &vv[j]);
      HYPRE_IJVectorSetObjectType(vv[j], HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(vv[j]);
      for ( i = mybegin; i <= myend; i++ )
         HYPRE_IJVectorSetValues(vv[j], 1, (const int *) &i,
                                 (const double *) &(tempvector[i]));
       HYPRE_IJVectorAssemble(vv[j]);
      
      HYPRE_IJVectorGetObject(vv[j], (void ** ) &A->vsPar[j]);
   }

   /*------------------------------------------------------------------
    * clean up
    *----------------------------------------------------------------*/
   free( ia );
   free( ja );
   free( val );
   free(tempvector);
}

/*--------------------------------------------------------------------------
 * Get_CRSMatrixFromFileBinary
 * This is a CRS binary format (see Jan Mandel- University of CO at Denver)
 * which only contains the upper/lower part of a symmetric matrix.
 * (read by a single processor)                                       
 *--------------------------------------------------------------------------*/
int Get_CRSMatrixFromFileBinary(double **val, int **ia,
     int **ja, int *m, int *n, char *file_name)
{
  int ncols,nrows,nzs;
  int *cols, *rows;
  double *val1;
  FILE *fid;
  int i;

  if ((fid=fopen(file_name,"r"))==NULL){
    fprintf(stderr, "Cannot open file: %s\n",file_name);
    exit(1);
  }

  expect(fid,4);
  expect(fid,1);
  expect(fid,4);
  expect(fid,4);
  fread(&nrows,sizeof(int),1,fid);
  expect(fid,4);
  expect(fid,4);
  fread(&ncols,sizeof(int),1,fid);
  expect(fid,4);
  expect(fid,4);
  fread(&nzs,sizeof(int),1,fid);
  expect(fid,4);
  expect(fid,4*(ncols+1));

  printf("\nFortran Binary Input File: %s\n",file_name);
  printf("=============================================\n");
  printf("    Number of Columns:   %d\n",ncols);
  printf("    Number of Rows:      %d\n",nrows);
  printf("    Number of NonZeroes: %d\n",nzs);
  printf("\n");

  /* allocate storage */
  if (!(rows=(int *) malloc((nrows+1)*sizeof(int)))) {
    fprintf(stderr, "Out of memory\n");
    abort();
  }
  if (!(cols=(int *) malloc(nzs*sizeof(int)))) {
    fprintf(stderr, "Out of memory\n");
    abort();
  }
  if (!(val1=(double *) malloc(nzs*sizeof(double)))) {
    fprintf(stderr, "Out of memory\n");
    abort();
  }

  fread(rows,sizeof(int),(long)nrows+1,fid);
  expect(fid,4*(ncols+1));
  expect(fid,4*nzs);
  fread(cols,sizeof(int),(long)nzs,fid);
  expect(fid,4*nzs);
  expect(fid,8*nzs);

  fread(val1,sizeof(double),(long)nzs,fid);
  expect(fid,8*nzs);
  fclose(fid);

  rows[nrows]=nzs;
  for (i=0;i<nrows;++i) --rows[i];
  for (i=0;i<nzs;++i)   --cols[i];
  (*val) = val1;
  (*ia)  =rows;
  (*ja)  =cols;
  (*m) = nrows;
  (*n) = ncols;

  /* make symmetric */
  IJAMatrixiToSymmetric(val,ia,ja,m);

  printf("matrix has %d rows, %d columns and %d nonzeros\n",
    nrows,ncols, (*ia)[nrows]);
  printf("returning from reading binary file\n\n");

  return 0;
}

/*-------------------------------------------------------------------------
 * IJAMatrixiToSymmetric
 * Update IJA data to make it symmetric                       
 *-------------------------------------------------------------------------*/
int IJAMatrixiToSymmetric(double **val, int **ia,
     int **ja, int *m)
{
  int i,j,nrows,nnz,count=0, icount, rowindex, colindex, curr_row;
  int    *mat_ia, *mat_ja;
  double *mat_a, value;

  int off_diag_count=0;
  input_data *input1;

  nrows=*m;
  nnz=(*ia)[nrows];

  /* allocate memory */
  if ((input1=(input_data *) malloc(nnz*sizeof(input_data)))==NULL){
    fprintf(stderr, "Could not allocate memory for input data.\n");
    exit(1);
  }

  /* load ija data */
  for (i=0;i<nrows;++i)
  {
    for (j=(*ia)[i];j<(*ia)[i+1];++j)
    {
      input1[count].row=i;  
      input1[count].col=(*ja)[j];  
      input1[count].val=(*val)[j];  
      input1[count].index=input1[count].row*nrows+input1[count].col;
      ++count;
    }
  }

  /* find number of off-diagonal values */
  for (i=0; i<nnz; i++) {
    if (input1[i].row!=input1[i].col) ++off_diag_count;
  }

  /* reallocate memory */
  if ((input1=(input_data *) realloc(input1,(nnz+off_diag_count)*sizeof(input_data)))==NULL){
    fprintf(stderr, "Could not allocate memory for input data.\n");
    exit(1);
  }

  /* add in elements to make symmetric */
  j=nnz;
  for (i=0; i<nnz; i++) {
    if (input1[i].row != input1[i].col){
      input1[j].row=input1[i].col;
      input1[j].col=input1[i].row;
      input1[j].val=input1[i].val;
      input1[j].index=input1[j].row*nrows+input1[j].col;
      ++j;
    }
  }
  nnz =nnz+off_diag_count;

  /* sort input data by rows and then columns */
  qsort(input1,nnz,sizeof(input_data),comp);

  free((*ia));
  free((*ja));
  free((*val));

  mat_ia = (int *) malloc((nrows+1) * sizeof(int));
  mat_ja = (int *) malloc( nnz * sizeof(int));
  mat_a  = (double *) malloc( nnz * sizeof(double));

  mat_ia[0] = 0;
  mat_ia[nrows] = nnz;
  curr_row = 0;
  icount   = 0;
  for ( i = 0; i < nnz; i++ ) {
     rowindex= input1[i].row;
     colindex=input1[i].col;
     value=input1[i].val;
     if ( rowindex != curr_row ) mat_ia[++curr_row] = icount;
     if ( rowindex < 0 || rowindex >= nrows )
        fprintf(stderr, "Error reading row %d (curr_row = %d)\n", rowindex, curr_row);
     if ( colindex < 0 || colindex >= nrows )
        fprintf(stderr, "Error reading col %d (rowindex = %d)\n", colindex, rowindex);
     mat_ja[icount] = colindex;
     mat_a[icount++]  = value;
  }

  (*val) = mat_a;
  (*ia)  = mat_ia;
  (*ja)  = mat_ja;

  free(input1);
  return 0;
}


/*****************************************************************************/
void expect(FILE *fid,int i)
{
  int ii;

  fread(&ii,sizeof(int),1,fid);
  if (ii != i){
    fprintf(stderr, "Error: Expected %d, got %d\n",i,ii);
    exit(1);
  }
}

/*****************************************************************************/
int PrintArray(Matx *A,char fname[])
{
  /* print array to a matrix market file - matrix n x m */
  int i,j;
  int    mypid, nprocs;
  MM_typecode matcode;
  FILE *fp=NULL;
  double  *data=NULL;
  hypre_Vector *local_vector;

  MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* check to see if there is any data */
  if (A->m==0 || A->n==0 || A->nz==0) return 0;

  if (A->mat_storage_type==DENSE){
    if ((fp=fopen(fname,"w"))==NULL){
      fprintf(stderr, "Cannot open file %s for writing\n",fname);
      exit(1);
    }

    /* set up matrix market parameters */
    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_dense(&matcode);
    mm_set_real(&matcode);
    mm_set_general(&matcode);
    mm_write_banner(fp, matcode);
    fprintf(fp,"%d %d\n",A->m,A->n);

    /* print to file column by column */
    for (j=0;j<A->n;j++){
      for (i=0; i<A->m; i++) {
        fprintf(fp,"%22.16e\n",A->val[i][j]);
     }
    }
    fclose(fp);
    return 0;
  }
  if (A->mat_storage_type==HYPRE_VECTORS){
    if ( mypid == 0 ){
      if ((fp=fopen(fname,"w"))==NULL){
        fprintf(stderr, "Cannot open file %s for writing\n",fname);
        exit(1);
      }

      /* set up matrix market parameters */
      mm_initialize_typecode(&matcode);
      mm_set_matrix(&matcode);
      mm_set_dense(&matcode);
      mm_set_real(&matcode);
      mm_set_general(&matcode);
      mm_write_banner(fp, matcode);
      fprintf(fp,"%d %d\n",A->m,A->n);
    }
    for (i=0;i<A->n;i++){
      local_vector = hypre_ParVectorToVector(MPI_COMM_WORLD,(hypre_ParVector *) A->vsPar[i]);
      if ( mypid==0){
        data = hypre_VectorData(local_vector);
        for (j=0; j<A->m;j++) {
          fprintf(fp,"%22.16e\n",data[j]);
        }
        hypre_SeqVectorDestroy(local_vector);
      }
    }

   /*------------------------------------------------------------------
    * clean up
    *----------------------------------------------------------------*/
    if ( mypid == 0 ) fclose(fp);
    return 0;
  }
  else fprintf(stderr, "Can't save this file to matrix market format\n");
  return 1;
}

/*--------------------------------------------------------------------------
 * PrintPartitioning:
 * Print partitioning.
 *--------------------------------------------------------------------------*/
int PrintPartitioning(int *partitioning,char *name)
{
  int i,mypid,nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if (mypid==0)
  {
     printf("Partitioning: %s\n",name);
     for (i=0;i<nprocs;++i) printf("%d %d\n",partitioning[i],partitioning[i+1]);
  }
  return 0;
}

/*--------------------------------------------------------------------------
 * Print_Par_Matrix_To_Mtx:
 * Print a parallel matrix to a dense matrix market file.
 * This is only for testing purposes for small matrices.
 *--------------------------------------------------------------------------*/
int Print_Par_Matrix_To_Mtx(HYPRE_ParCSRMatrix A,char *filename)
{
  /* allocate and initialize matrix of size m x n to random numbers */
  int i,j;
  hypre_Vector  *v_temp;
  HYPRE_ParVector     b=NULL;
  double  *vector_data;
  int   mypid,nprocs;
  int M,N;
  Matx *X;
  int *partitioning,*part2;

  /* allocate storage */
  X=Mat_Alloc1();

  /* get partitioning and size of matrix */
  HYPRE_ParCSRMatrixGetRowPartitioning(A,&partitioning);
  HYPRE_ParCSRMatrixGetDims(A,&M,&N);
  
  part2=CopyPartition(partitioning);
  if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorCreate_Data,0);
  HYPRE_ParVectorCreate(MPI_COMM_WORLD,M,part2,&b);
  HYPRE_ParVectorInitialize(b);

  /* setup X */
  X->m=M;
  X->n=N;
  X->nz=M*N;
  X->mat_storage_type=HYPRE_VECTORS;
  X->mat_type=GENERAL;

  /* allocate memory */
  if ((X->vsPar=(HYPRE_ParVector *) malloc(X->n*sizeof(HYPRE_ParVector)))==NULL){
     fprintf(stderr, "Could not allocate memory.\n");
     abort();
  }

    MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    v_temp=hypre_SeqVectorCreate(M);
    hypre_SeqVectorInitialize(v_temp);
    vector_data = hypre_VectorData(v_temp);
    for (i=0; i<N; i++) {
       for (j = 0; j < N; j++){
          if (j==i) vector_data[j] = 1.0;
          else vector_data[j] = 0.0;
       }
       part2=CopyPartition(partitioning);
       HYPRE_VectorToParVector(MPI_COMM_WORLD,(HYPRE_Vector) v_temp,part2,&X->vsPar[i]);
       HYPRE_ParVectorCopy(X->vsPar[i],b);
       HYPRE_ParCSRMatrixMatvec(1.0,A,b,0.0,X->vsPar[i]);
    }
    hypre_SeqVectorDestroy(v_temp);
    if (verbose2(1)==TRUE) collect_data(0,HYPRE_ParVectorDestroy_Data,0);
    HYPRE_ParVectorDestroy(b);
    PrintArray(X,filename);

    Mat_Free(X);free(X);
    return 0;
}

/*****************************************************************************/
void display_input(input_data *input1,int n)
/* display input for array to screen */
{
  /* display input to screen */
  int i;
  printf("n= %d\n",n);
  for (i=0;i<n;++i){
    printf("%e %e %e %e\n",input1[i].row,input1[i].col,input1[i].val,input1[i].index);
  }
}

/*****************************************************************************/
void PrintVector(double *data,int n,char fname[])
{
  /* print dense vector to a matrix market file - matrix n x 1 */
  int i;
  Matx  *A;

  A=Mat_Alloc1();
  Mat_Init_Dense(A,n,1,GENERAL);
  for (i=0; i<n; i++) A->val[i][0]=data[i];
  PrintArray(A,fname);

  Mat_Free(A);free(A);
}

/*****************************************************************************/
void PrintMatrix(double **data,int m,int n,char fname[])
{
  /* print dense matrix to a matrix market file - matrix m x n */
  int i,j;
  Matx  *A;
  A=Mat_Alloc1();
  Mat_Init_Dense(A,m,n,GENERAL);
  for (i=0; i<m; i++){
    for (j=0; j<n; j++) A->val[i][j]=data[i][j];
  }
  PrintArray(A,fname);

  Mat_Free(A);free(A);
}

/*****************************************************************************/
int MatViewUtil(Matx *A)
{
  /* view a matrix  */
  int i,j;

  if(A->mat_storage_type == DENSE){
    printf("MatViewUtil(DENSE) n=%d m=%d\n",A->m,A->n);
    for (i=0;i<A->m;i++){
      for (j=0;j<A->n;j++){
        printf("%11.4e",A->val[i][j]);
      }
      printf("\n");
    }
  }
  return 0;
}

/*****************************************************************************/
int VecViewUtil(double *b,int n)
{
  /* view a regular double vector  */
  int i;

  printf("VecViewUtil n=%d\n",n);
  for (i=0;i<n;i++){
      printf("%24.15e\n",b[i]);
    }
  printf("\n");
  return 0;
}

/*****************************************************************************/
int VecViewUtil2(int *b,int n)
{
  /* view a regular integer vector  */
  int i;

  printf("VecViewUtil n=%d\n",n);
  for (i=0;i<n;i++){
      printf("%d\n",b[i]);
    }
  printf("\n");
  return 0;
}

/*****************************************************************************/
void PrintMatrixParameters(Matx *A)
{
printf("m=%d,n=%d,nz=%d,mat_storage_type=%d,mat_type=%d,max. vec.=%d\n",
       A->m,A->n,A->nz,A->mat_storage_type,A->mat_type,A->numb_par_vectors_alloc);
}

/*--------------------------------------------------------------------------
 * readmatrix
 *
 * Note: If lobpcg is executed using multiple processors or a different
 * partition with the same number of processors, slightly different results
 * may be observed, although they will be correct. This stems from the
 * a characteristic of HYPRE solvers. 
 *
 * This occurs in the case where the same matrix is read as input from
 * a matrix market file or is generated internally with a different
 * partition of data across processors.
 *--------------------------------------------------------------------------*/
int readmatrix(char Ain[],Matx *A,mst mat_storage_type,int *partitioning)
{
  /* Read matrix in Matrix Market Format and assemble into A */ 

  int         mA,nA,nzA;
  int         i,j,k,kk,ii;
  FILE        *Afile;
  MM_typecode matcode; /* Matrix Market type code */
  mt mat_type;

  input_data *input1;

  /* handle hypre multiple processor implementation */
  if (mat_storage_type==HYPRE_MATRIX){
    i=strlen(Ain);
    if (strcmp(&Ain[i-4],".mtx")==0)
      /* matrix market format */
      HYPRE_Load_IJAMatrix(A,MATRIX_INPUT_MTX,Ain,partitioning);
    else if (strcmp(&Ain[i-4],".data")==0)
      /* standard simple IJA format (see Charles Tong) */
      HYPRE_Load_IJAMatrix(A,MATRIX_INPUT_PLANE_IJ,Ain,partitioning);
    else {
      fprintf(stderr, "%s is in invalid file name\n",Ain);
      exit(1);
    }
    A->mat_storage_type=mat_storage_type;
    return 0;
  }
  else if (mat_storage_type==HYPRE_VECTORS){
    i=strlen(Ain);
    if (strcmp(&Ain[i-4],".mtx")==0){
      HYPRE_LoadMatrixVectorMtx(A,Ain,partitioning);
      }
    else {
      fprintf(stderr, "%s is in invalid file name\n",Ain);
      exit(1);
    }
    A->mat_storage_type=mat_storage_type;
    return 0;
  }



  if ((Afile=fopen(Ain,"r"))==NULL){
    if (Get_Rank()==0) fprintf(stderr, "Cannot open file: %s\n",Ain);
    exit(1);
  }

  /* read Matrix Market Banner */
  if (mm_read_banner(Afile,&matcode) != 0) {
    if (Get_Rank()==0) fprintf(stderr, "Could not process Matrix Market banner.\n");
    exit(1);
  }

  /* read in general sparse matrix */
  if (mm_is_sparse(matcode) && mm_is_general(matcode)){

    /* read array parameters for sparse matrix */
    if ((mm_read_mtx_crd_size(Afile,&mA,&nA,&nzA)) !=0){
      if (Get_Rank()==0) fprintf(stderr, "Could not process Matrix Market parameters.\n");
      exit(1);
    }

    /* allocate memory */
    if ((input1=(input_data *) malloc(nzA*sizeof(input_data)))==NULL){
      if (Get_Rank()==0) fprintf(stderr, "Could not allocate memory for input data.\n");
      exit(1);
    }

    /* read in values */
    ii=1;
    if(nzA>10) ii=nzA/10;
    for (i=0; i<nzA; i++) {
      if (i % ii == 0 && Get_Rank()==0)  printf("nzA= %d %d\n",nzA,i); /* output every 10% */
      fscanf(Afile,"%le %le %le\n",&input1[i].row,&input1[i].col,&input1[i].val);
      input1[i].row--; input1[i].col--;  /* set index set starts at 0 */
      input1[i].index=input1[i].row*nA+input1[i].col;
    }

    fclose(Afile);
    mat_type=GENERAL;
  }

  /* read in symmetric sparse matrix */
  else if (mm_is_sparse(matcode) && mm_is_symmetric(matcode)){

    /* read array parameters for sparse matrix */
    if ((mm_read_mtx_crd_size(Afile,&mA,&nA,&nzA)) !=0){
      if (Get_Rank()==0) fprintf(stderr, "Could not process Matrix Market parameters.\n");
      exit(1);
    }

    /* allocate memory */
    if ((input1=(input_data *) malloc(nzA*sizeof(input_data)))==NULL){
      if (Get_Rank()==0) fprintf(stderr, "Could not allocate memory for input data.\n");
      exit(1);
    }

    /* read in values */
    ii=1;
    if(nzA>10) ii=nzA/10;
    for (i=0; i<nzA; i++) {
      if (i % ii == 0 && Get_Rank()==0)  printf("nzA= %d %d\n",nzA,i); /* output every 10% */
      fscanf(Afile,"%le %le %le\n",&input1[i].row,&input1[i].col,&input1[i].val);
      input1[i].row--; input1[i].col--;  /* set index set starts at 0 */
      input1[i].index=input1[i].row*nA+input1[i].col;
    }

    fclose(Afile);
    mat_type=SYMMETRIC;
  }

  /* read in dense matrix */
  else if (mm_is_dense(matcode) && mm_is_general(matcode)){

    /* read array parameters for dense matrix */
    if ((mm_read_mtx_array_size(Afile,&mA,&nA)) !=0){
      if (Get_Rank()==0) fprintf(stderr, "Could not process Matrix Market parameters.\n");
      exit(1);
    }
    nzA=mA*nA;

    /* allocate memory */
    if ((input1=(input_data *) malloc(nzA*sizeof(input_data)))==NULL){
      if (Get_Rank()==0) fprintf(stderr, "Could not allocate memory for input data.\n");
      exit(1);
    }

    /* read in values */
    k=0;
    for (j=0; j<nA; j++){
      for (i=0; i<mA; i++){
        fscanf(Afile,"%le\n",&input1[k].val);
        input1[k].row=i; input1[k].col=j;  /* set index set starts at 0 */
        input1[k].index=input1[k].row*nA+input1[k].col;
        k++;
        }
    }
    fclose(Afile);
    mat_type=GENERAL;
  }

  /* read in dense matrix that is symmetric */
  else if (mm_is_dense(matcode) && mm_is_symmetric(matcode)){

    /* read array parameters for dense matrix */
    if ((mm_read_mtx_array_size(Afile,&mA,&nA)) !=0){
      if (Get_Rank()==0) fprintf(stderr, "Could not process Matrix Market parameters.\n");
      exit(1);
    }
    nzA=mA*nA;

    /* allocate memory */
    if ((input1=(input_data *) malloc(2*nzA*sizeof(input_data)))==NULL){
      if (Get_Rank()==0) fprintf(stderr, "Could not allocate memory for input data.\n");
      exit(1);
    }

    /* read in values */
    k=0;
    for (j=0; j<nA; j++){
      for (i=j; i<mA; i++){
        fscanf(Afile,"%le\n",&input1[k].val);
        input1[k].row=i; input1[k].col=j;  /* set index set starts at 0 */
        input1[k].index=input1[k].row*nA+input1[k].col;
        k++;
        }
    }
    fclose(Afile);

    /* fill in remaining values */
    kk=k;
    k=0;
    for (i=0; i<kk; i++) {
      if (input1[i].row > input1[i].col){
        j=kk+k;
        input1[j].row=input1[i].col;
        input1[j].col=input1[i].row;
        input1[j].val=input1[i].val;
        input1[j].index=input1[j].row*nA+input1[j].col;
        k++;
      }
    }
    assert(nzA!=k);
    mat_type=SYMMETRIC;
  }

  else {
    if (Get_Rank()==0) fprintf(stderr, "Unabel to read Matrix Market format\n");
    exit(1);
  }

  /* sort input data */
  qsort(input1,nzA,sizeof(input_data),comp);

  /* assemble matrix */
  switch(mat_storage_type)
  {
    case DENSE:
      if (Get_Rank()==0) printf("Assembling DENSE matrix\n");
      Assemble_DENSE(A,input1,mA,nA,nzA,mat_type);
      break;
    default:
      if (Get_Rank()==0) fprintf(stderr, "Invalid matrix type\n");
      exit(1);
  }

  free(input1);
  return 0;
}

/*****************************************************************************/
int comp(const void *p1,const void *p2)
{
/* comparison function used by qsort */
/* sort values by increasing */

  const input_data *ps1=(input_data *) p1;
  const input_data *ps2=(input_data *) p2;
  
  if (ps1->index < ps2->index) return -1;
  else if (ps1->index == ps2->index) return 0;
  else if (ps1->index > ps2->index) return 1;

  return 0;
}

/*****************************************************************************/
int Mat_Size_Mtx(char *file_name,int rc)
{
  /* get size of matrix market matrix from a file */

  int         mA,nA,nzA;
  FILE        *Afile;
  MM_typecode matcode; /* Matrix Market type code */

  if ((Afile=fopen(file_name,"r"))==NULL){
    if (Get_Rank()==0) fprintf(stderr, "Cannot open file: %s\n",file_name);
    exit(1);
  }

  /* read Matrix Market Banner */
  if (mm_read_banner(Afile,&matcode) != 0) {
    if (Get_Rank()==0) fprintf(stderr, "Could not process Matrix Market banner.\n");
    exit(1);
  }

  if (mm_is_sparse(matcode)){
    /* read array parameters for sparse matrix */
    if ((mm_read_mtx_crd_size(Afile,&mA,&nA,&nzA)) !=0){
      if (Get_Rank()==0) fprintf(stderr, "Could not process Matrix Market parameters.\n");
      exit(1);
    }
  }
  else if (mm_is_dense(matcode)){
    /* read array parameters for dense matrix */
    if ((mm_read_mtx_array_size(Afile,&mA,&nA)) !=0){
      if (Get_Rank()==0) fprintf(stderr, "Could not process Matrix Market parameters.\n");
      exit(1);
    }
  }
  else {
	fprintf(stderr, "Mat_Size_Mtx function failed.\n");
	abort();
  }

  if (rc==1) return mA;
  else return nA;
}

/*****************************************************************************/
int Mat_Size_Bin(char *file_name,int rc)
{
  /* File is in Fortran Binary format (see Jan Mandel) */
  int ncols,nrows;
  FILE *fid;

  if ((fid=fopen(file_name,"r"))==NULL){
    fprintf(stderr, "Cannot open file: %s\n",file_name);
    exit(1);
  }

  expect(fid,4);
  expect(fid,1);
  expect(fid,4);
  expect(fid,4);
  fread(&nrows,sizeof(int),1,fid);
  expect(fid,4);
  expect(fid,4);
  fread(&ncols,sizeof(int),1,fid);
  fclose(fid);

  if (rc==1) return nrows;
  else return ncols;
}

/*****************************************************************************/
int Get_Lobpcg_Options(int argc,char **argv,lobpcg_options *opts)
{
   int ftype,count;

   /* read in all command line options */
   int arg_index=1;

   opts->verbose=1; /* default is standard amount of output */
   opts->flag_f=FALSE;
   opts->flag_A=FALSE;
   opts->flag_B=FALSE;
   opts->flag_V=FALSE;
   opts->flag_T=FALSE;
   opts->flag_precond=FALSE;
   opts->Vrand = 0;
   opts->Veye = 0;
   opts->pcg_max_flag=FALSE;  
   opts->flag_feig=FALSE;
   opts->flag_tol=FALSE;
   opts->flag_itr=FALSE;
   opts->printA=FALSE;
   opts->flag_orth_check=FALSE;

   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-v") == 0 )
      {
         arg_index++;
         opts->verbose = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-f") == 0 )
      {
         arg_index++;
         opts->flag_f=TRUE;
      }
      else if ( strcmp(argv[arg_index], "-ain") == 0 )
      {
         arg_index++;
         opts->flag_A=TRUE;
         strcpy(opts->Ain,argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-bin") == 0 )
      {
         arg_index++;
         opts->flag_B=TRUE;
         strcpy(opts->Bin,argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-vin") == 0 )
      {
         arg_index++;
         opts->flag_V=TRUE;
         strcpy(opts->Vin,argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tin") == 0 )
      {
         arg_index++;
         opts->flag_T=TRUE;
         strcpy(opts->Tin,argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-prec") == 0 )
      {
         arg_index++;
         opts->flag_precond=TRUE;
      }
      else if ( strcmp(argv[arg_index], "-chk") == 0 )
      {
         arg_index++;
         opts->flag_orth_check=TRUE;
      }

      else if ( strcmp(argv[arg_index], "-vrand") == 0 )
      {
         arg_index++;
         opts->Vrand = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-veye") == 0 )
      {
         arg_index++;
         opts->Veye = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-f") == 0 )
      {
         arg_index++;
         opts->flag_feig=TRUE;
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         opts->flag_tol=TRUE;
         opts->tol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-itr") == 0 )
      {
         arg_index++;
         opts->flag_itr=TRUE;
         opts->max_iter_count = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgitr") == 0 )
      {
         arg_index++;
         opts->pcg_max_itr = atoi(argv[arg_index++]);
         opts->pcg_max_flag=TRUE;  
      }
      else if ( strcmp(argv[arg_index], "-pcgtol") == 0 )
      {
         arg_index++;
         opts->pcg_tol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-printA") == 0 )
      {
         arg_index++;
         opts->printA=TRUE;
      }
      else if ( strcmp(argv[arg_index], "-eyeA") == 0 )
      {
         /* uses identity for A - used for debugging */
         arg_index++;
         misc_flags(0,0);
      }
      else if ( strcmp(argv[arg_index], "-eyeT") == 0 )
      {
         /* uses identity for T solver  - used for debugging */
         arg_index++;
         misc_flags(0,1);
      }
      else if ( strcmp(argv[arg_index], "-time") == 0 )
      {
         arg_index++;
         ftype=atoi(argv[arg_index++]);
         count=atoi(argv[arg_index++]);
         time_functions(0,ftype,0,count,0);
      }
      else
      {
         arg_index++;
      }
   }

  return 0;
}

/*****************************************************************************/
void Display_Execution_Statistics()
{
    /* Display execution statistics */
    int j,sum,temp,total[4];

    for (j=0;j<4;++j) total[j]=0;

    printf("\n");
    printf("=============================================\n");
    printf("Lobpcg Execution Statistics:\n");
    printf("=============================================\n");
    printf("HYPRE Function/Phase\t\t\tSetup\tItr=1\tItr>1\tFinal\tTotal");

    printf("\nHYPRE_ParVectorCreate:\t\t");
    sum=0;
    for (j=0;j<4;++j){
      temp=collect_data(2,HYPRE_ParVectorCreate_Data,j);
      printf("\t%d",temp);
      sum=sum+temp;
      total[j]=total[j]+temp;
    }
    printf("\t%d",sum);

    printf("\nHYPRE_ParVectorDestroy:\t\t");
    sum=0;
    for (j=0;j<4;++j){
      temp=collect_data(2,HYPRE_ParVectorDestroy_Data,j);
      printf("\t%d",temp);
      sum=sum+temp;
      total[j]=total[j]+temp;
    }
    printf("\t%d",sum);

    printf("\nHYPRE_ParVectorInnerProd:\t");
    sum=0;
    for (j=0;j<4;++j){
      temp=collect_data(2,HYPRE_ParVectorInnerProd_Data,j);
      printf("\t%d",temp);
      sum=sum+temp;
      total[j]=total[j]+temp;
    }
    printf("\t%d",sum);

    printf("\nHYPRE_ParVectorCopy:\t\t");
    sum=0;
    for (j=0;j<4;++j){
      temp=collect_data(2,HYPRE_ParVectorCopy_Data,j);
      printf("\t%d",temp);
      sum=sum+temp;
      total[j]=total[j]+temp;
    }
    printf("\t%d",sum);

    printf("\nhypre_ParVectorAxpy:\t\t");
    sum=0;
    for (j=0;j<4;++j){
      temp=collect_data(2,hypre_ParVectorAxpy_Data,j);
      printf("\t%d",temp);
      sum=sum+temp;
      total[j]=total[j]+temp;
    }
    printf("\t%d",sum);

    printf("\nHYPRE_ParVectorSetConstantValues:");
    sum=0;
    for (j=0;j<4;++j){
      temp=collect_data(2,HYPRE_ParVectorSetConstantValues_Data,j);
      printf("\t%d",temp);
      sum=sum+temp;
      total[j]=total[j]+temp;
    }
    printf("\t%d",sum);

    printf("\n\t\t\t\t");
    for (j=0;j<5;++j){
      printf("\t------");
    }
    printf("\nTotal Number of Vector Operations:");
    temp=0;
    for (j=0;j<4;++j){
      printf("\t%d",total[j]);
      temp=temp+total[j];
    }
    printf("\t%d",temp);

    printf("\n");
    printf("\nNumber A-Multiplies:\t\t");
    sum=0;
    for (j=0;j<4;++j){
      temp=collect_data(2,NUMBER_A_MULTIPLIES,j);
      printf("\t%d",temp);
      sum=sum+temp;
    }
    printf("\t%d",sum);

    printf("\nNumber Solves:\t\t\t");
    sum=0;
    for (j=0;j<4;++j){
      temp=collect_data(2,NUMBER_SOLVES,j);
      printf("\t%d",temp);
      sum=sum+temp;
    }
    printf("\t%d",sum);
}



/*--------------------------------------------------------------------------
 * The following routines have been downloaded from the Matrix Market
 * home page (http://math.nist.gov/MatrixMarket/)
 *--------------------------------------------------------------------------*/

int mm_read_unsymmetric_sparse(const char *fname, int *M_, int *N_, int *nz_,
                double **val_, int **I_, int **J_)
{
    FILE *f;
    MM_typecode matcode;
    int M, N, nz;
    int i;
    double *val;
    int *I, *J;
 
    if ((f = fopen(fname, "r")) == NULL)
            return -1;
 
 
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("mm_read_unsymetric: Could not process Matrix Market banner ");
        printf(" in file [%s]\n", fname);
        return -1;
    }
 
 
 
    if ( !(mm_is_real(matcode) && mm_is_matrix(matcode) &&
            mm_is_sparse(matcode)))
    {
        fprintf(stderr, "Sorry, this application does not support ");
        fprintf(stderr, "Market Market type: [%s]\n",
                mm_typecode_to_str(matcode));
        return -1;
    }
 
    /* find out size of sparse matrix: M, N, nz .... */
 
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) !=0)
    {
        fprintf(stderr, "read_unsymmetric_sparse(): could not parse matrix size.\n");
        return -1;
    }
 
    *M_ = M;
    *N_ = N;
    *nz_ = nz;
 
    /* reseve memory for matrices */
 
    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));
 
    *val_ = val;
    *I_ = I;
    *J_ = J;
 
    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
 
    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }
    fclose(f);
 
    return 0;
}

int mm_is_valid(MM_typecode matcode)
{
    if (!mm_is_matrix(matcode)) return 0;
    if (mm_is_dense(matcode) && mm_is_pattern(matcode)) return 0;
    if (mm_is_real(matcode) && mm_is_hermitian(matcode)) return 0;
    if (mm_is_pattern(matcode) && (mm_is_hermitian(matcode) || 
                mm_is_skew(matcode))) return 0;
    return 1;
}

int mm_read_banner(FILE *f, MM_typecode *matcode)
{
    char line[MM_MAX_LINE_LENGTH];
    char banner[MM_MAX_TOKEN_LENGTH];
    char mtx[MM_MAX_TOKEN_LENGTH]; 
    char crd[MM_MAX_TOKEN_LENGTH];
    char data_type[MM_MAX_TOKEN_LENGTH];
    char storage_scheme[MM_MAX_TOKEN_LENGTH];
    char *p;


    mm_clear_typecode(matcode);  

    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL) 
        return MM_PREMATURE_EOF;

    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type, 
        storage_scheme) != 5)
        return MM_PREMATURE_EOF;

    for (p=mtx; *p; *p=tolower(*p),p++);  /* convert to lower case */
    for (p=crd; *p; *p=tolower(*p),p++);  
    for (p=data_type; *p; *p=tolower(*p),p++);
    for (p=storage_scheme; *p; *p=tolower(*p),p++);

    /* check for banner */
    if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
        return MM_NO_HEADER;

    /* first field should be "mtx" */
    /* modified by M.Argentati 8/1/02 to handle "mtx" */ 
    /*if (strcmp(mtx, MM_MTX_STR) != 0) */
    if ((strcmp(mtx, MM_MTX_STR) != 0) && (strcmp(mtx, MM_MTX_STR2) != 0))
        return  MM_UNSUPPORTED_TYPE;
    mm_set_matrix(matcode);


    /* second field describes whether this is a sparse matrix (in coordinate
            storgae) or a dense array */


    if (strcmp(crd, MM_SPARSE_STR) == 0)
        mm_set_sparse(matcode);
    else
    if (strcmp(crd, MM_DENSE_STR) == 0)
            mm_set_dense(matcode);
    else
        return MM_UNSUPPORTED_TYPE;
    

    /* third field */

    if (strcmp(data_type, MM_REAL_STR) == 0)
        mm_set_real(matcode);
    else
    if (strcmp(data_type, MM_COMPLEX_STR) == 0)
        mm_set_complex(matcode);
    else
    if (strcmp(data_type, MM_PATTERN_STR) == 0)
        mm_set_pattern(matcode);
    else
    if (strcmp(data_type, MM_INT_STR) == 0)
        mm_set_integer(matcode);
    else
        return MM_UNSUPPORTED_TYPE;
    

    /* fourth field */

    if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
        mm_set_general(matcode);
    else
    if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
        mm_set_symmetric(matcode);
    else
    if (strcmp(storage_scheme, MM_HERM_STR) == 0)
        mm_set_hermitian(matcode);
    else
    if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
        mm_set_skew(matcode);
    else
        return MM_UNSUPPORTED_TYPE;
        

    return 0;
}

int mm_write_mtx_crd_size(FILE *f, int M, int N, int nz)
{
    if (fprintf(f, "%d %d %d\n", M, N, nz) != 3)
        return MM_COULD_NOT_WRITE_FILE;
    else 
        return 0;
}

int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz )
{
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;

    /* set return null parameter values, in case we exit with errors */
    *M = *N = *nz = 0;

    /* now continue scanning until you reach the end-of-comments */
    do 
    {
        if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL) 
            return MM_PREMATURE_EOF;
    }while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d %d", M, N, nz) == 3)
        return 0;
        
    else
    do
    { 
        num_items_read = fscanf(f, "%d %d %d", M, N, nz); 
        if (num_items_read == EOF) return MM_PREMATURE_EOF;
    }
    while (num_items_read != 3);

    return 0;
}


int mm_read_mtx_array_size(FILE *f, int *M, int *N)
{
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;
    /* set return null parameter values, in case we exit with errors */
    *M = *N = 0;
	
    /* now continue scanning until you reach the end-of-comments */
    do 
    {
        if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL) 
            return MM_PREMATURE_EOF;
    }while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d", M, N) == 2)
        return 0;
        
    else /* we have a blank line */
    do
    { 
        num_items_read = fscanf(f, "%d %d", M, N); 
        if (num_items_read == EOF) return MM_PREMATURE_EOF;
    }
    while (num_items_read != 2);

    return 0;
}

int mm_write_mtx_array_size(FILE *f, int M, int N)
{
    if (fprintf(f, "%d %d\n", M, N) != 2)
        return MM_COULD_NOT_WRITE_FILE;
    else 
        return 0;
}



/*-------------------------------------------------------------------------*/

/******************************************************************/
/* use when I[], J[], and val[]J, and val[] are already allocated */
/******************************************************************/

int mm_read_mtx_crd_data(FILE *f, int M, int N, int nz, int I[], int J[],
        double val[], MM_typecode matcode)
{
    int i;
    if (mm_is_complex(matcode))
    {
        for (i=0; i<nz; i++)
            if (fscanf(f, "%d %d %lg %lg", &I[i], &J[i], &val[2*i], &val[2*i+1])
                != 4) return MM_PREMATURE_EOF;
    }
    else if (mm_is_real(matcode))
    {
        for (i=0; i<nz; i++)
        {
            if (fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i])
                != 3) return MM_PREMATURE_EOF;

        }
    }

    else if (mm_is_pattern(matcode))
    {
        for (i=0; i<nz; i++)
            if (fscanf(f, "%d %d", &I[i], &J[i])
                != 2) return MM_PREMATURE_EOF;
    }
    else
        return MM_UNSUPPORTED_TYPE;

    return 0;
        
}

int mm_read_mtx_crd_entry(FILE *f, int *I, int *J,
        double *real, double *imag, MM_typecode matcode)
{
    if (mm_is_complex(matcode))
    {
            if (fscanf(f, "%d %d %lg %lg", I, J, real, imag)
                != 4) return MM_PREMATURE_EOF;
    }
    else if (mm_is_real(matcode))
    {
            if (fscanf(f, "%d %d %lg\n", I, J, real)
                != 3) return MM_PREMATURE_EOF;

    }

    else if (mm_is_pattern(matcode))
    {
            if (fscanf(f, "%d %d", I, J) != 2) return MM_PREMATURE_EOF;
    }
    else
        return MM_UNSUPPORTED_TYPE;

    return 0;
        
}


/************************************************************************
    mm_read_mtx_crd()  fills M, N, nz, array of values, and return
                        type code, e.g. 'MCRS'

                        if matrix is complex, values[] is of size 2*nz,
                            (nz pairs of real/imaginary values)
************************************************************************/

int mm_read_mtx_crd(char *fname, int *M, int *N, int *nz, int **I, int **J, 
        double **val, MM_typecode *matcode)
{
    int ret_code;
    FILE *f;

    if (strcmp(fname, "stdin") == 0) f=stdin;
    else
    if ((f = fopen(fname, "r")) == NULL)
        return MM_COULD_NOT_READ_FILE;


    if ((ret_code = mm_read_banner(f, matcode)) != 0)
        return ret_code;

    if (!(mm_is_valid(*matcode) && mm_is_sparse(*matcode) && 
            mm_is_matrix(*matcode)))
        return MM_UNSUPPORTED_TYPE;

    if ((ret_code = mm_read_mtx_crd_size(f, M, N, nz)) != 0)
        return ret_code;


    *I = (int *)  malloc(*nz * sizeof(int));
    *J = (int *)  malloc(*nz * sizeof(int));
    *val = NULL;

    if (mm_is_complex(*matcode))
    {
        *val = (double *) malloc(*nz * 2 * sizeof(double));
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, 
                *matcode);
        if (ret_code != 0) return ret_code;
    }
    else if (mm_is_real(*matcode))
    {
        *val = (double *) malloc(*nz * sizeof(double));
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, 
                *matcode);
        if (ret_code != 0) return ret_code;
    }

    else if (mm_is_pattern(*matcode))
    {
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, 
                *matcode);
        if (ret_code != 0) return ret_code;
    }

    if (f != stdin) fclose(f);
    return 0;
}

int mm_write_banner(FILE *f, MM_typecode matcode)
{
    char *str = mm_typecode_to_str(matcode);
    int ret_code;

    ret_code = fprintf(f, "%s %s\n", MatrixMarketBanner, str);
    free(str);
    if (ret_code !=2 )
        return MM_COULD_NOT_WRITE_FILE;
    else
        return 0;
}

int mm_write_mtx_crd(char fname[], int M, int N, int nz, int I[], int J[],
        double val[], MM_typecode matcode)
{
    FILE *f;
    int i;

    if (strcmp(fname, "stdout") == 0) 
        f = stdout;
    else
    if ((f = fopen(fname, "w")) == NULL)
        return MM_COULD_NOT_WRITE_FILE;
    
    /* print banner followed by typecode */
    fprintf(f, "%s ", MatrixMarketBanner);
    fprintf(f, "%s\n", mm_typecode_to_str(matcode));

    /* print matrix sizes and nonzeros */
    fprintf(f, "%d %d %d\n", M, N, nz);

    /* print values */
    if (mm_is_pattern(matcode))
        for (i=0; i<nz; i++)
            fprintf(f, "%d %d\n", I[i], J[i]);
    else
    if (mm_is_real(matcode))
        for (i=0; i<nz; i++)
            fprintf(f, "%d %d %20.16g\n", I[i], J[i], val[i]);
    else
    if (mm_is_complex(matcode))
        for (i=0; i<nz; i++)
            fprintf(f, "%d %d %20.16g %20.16g\n", I[i], J[i], val[2*i], 
                        val[2*i+1]);
    else
    {
        if (f != stdout) fclose(f);
        return MM_UNSUPPORTED_TYPE;
    }

    if (f !=stdout) fclose(f);

    return 0;
}
    

char  *mm_typecode_to_str(MM_typecode matcode)
{
    char buffer[MM_MAX_LINE_LENGTH];
    char *types[4];

    /* check for MTX type */
    if (mm_is_matrix(matcode)) 
        types[0] = MM_MTX_STR;

    /* check for CRD or ARR matrix */
    if (mm_is_sparse(matcode))
        types[1] = MM_SPARSE_STR;
    else
    if (mm_is_dense(matcode))
        types[1] = MM_DENSE_STR;
    else
        return NULL;

    /* check for element data type */
    if (mm_is_real(matcode))
        types[2] = MM_REAL_STR;
    else
    if (mm_is_complex(matcode))
        types[2] = MM_COMPLEX_STR;
    else
    if (mm_is_pattern(matcode))
        types[2] = MM_PATTERN_STR;
    else
    if (mm_is_integer(matcode))
        types[2] = MM_INT_STR;
    else
        return NULL;


    /* check for symmetry type */
    if (mm_is_general(matcode))
        types[3] = MM_GENERAL_STR;
    else
    if (mm_is_symmetric(matcode))
        types[3] = MM_SYMM_STR;
    else 
    if (mm_is_hermitian(matcode))
        types[3] = MM_HERM_STR;
    else 
    if (mm_is_skew(matcode))
        types[3] = MM_SKEW_STR;
    else
        return NULL;

    sprintf(buffer,"%s %s %s %s", types[0], types[1], types[2], types[3]);
    return (char *) strdup(buffer);
}
