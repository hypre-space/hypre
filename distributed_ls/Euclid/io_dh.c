#include "io_dh.h"
#include "Mat_dh.h"
#include "Vec_dh.h"
#include "Mem_dh.h"
#include "Timer_dh.h"
#include "Parser_dh.h"
/* #include "euclid_petsc.h" */
#include "mat_dh_private.h"


/* ---------------------------------------------------------------------------
    NOTES/CAUTIONS/WARNINGS/REMARKS:
      1.  storage is big endian
      2.  if error flag is returned, inputs for writing matrix files
          (e.g, convertTriples_noHeader(), convertTriples_shortHeader())
          may be corrupted, due to byte swapping.  
      3.  in current implementation, only 4 byte ints and 8 byte doubles
          are supported.

      4.  header values of "-1" are interpreted as "unknown" or "indeterminate."

      5.  binary binary files consist of 5 sections:
           section 1: an integer array of length HEADER_SIZE_DH (see below)
           section 2: matrix name, length is MAT_NAME_SIZE_DH (see below)
           section 3: free-form comments, length is COMMENT_SIZE_DH (see below)
           section 4: zero or more "row data records."  Each record is formatted:
                         row number
                         row length
                         column indices ("length" integer entries)
                         column values  ("length"*BLOCK_SIZE,*BLOCK_SIZE entries)
           section 5: zero or more "row summary records."  Each record is formatted:
                         row number
                         row length
                         offset     (offset in bytes where the corresponding
                                     "row data record" starts, wrt beginning 
                                     of file)
 * ---------------------------------------------------------------------------*/



/*===== (start) header formatting secret decoder ring ========================*/

#define HEADER_SIZE_DH     128   /* size in 4-byte ints */
#define MAT_NAME_SIZE_DH   128   /* size in bytes; for storing matix name (optional) */
#define COMMENT_SIZE_DH    1024  /* size in bytes; free-form comments (optional) */

enum{ FOOTER_OFFSET,
      GLOBAL_ROW_COUNT,  /* number of rows in global matrix */
      GLOBAL_NZ,         /* number of nonzeros in global matrix */
      FILE_ROW_COUNT,    /* number of rows stored in this binary file */
      FILE_NZ,           /* number of nonzeros stored in this binary file */
      BLOCK_SIZE,        /* e.g., number of degree of freedom per node */
      ROWS_ARE_SORTED,   /* rows in this file are entered in sorted ordered
                            (probably always true?)
                          */
      COLS_ARE_SORTED,   /* column indices within each row are in sorted order */
      END_ROW,           /* global row number of first row stored in this file */

     /* Only one of the next three may be true;
      * if none are true, then all are "unknown."
      */
      IS_UPPER_TRI,      /* -1 = unknown; 0 =  false; 1 = true */
      IS_LOWER_TRI,      /* -1 = unknown; 0 =  false; 1 = true */
      IS_FULL,           /* -1 = unknown; 0 =  false; 1 = true */

      VALUES_ARE_COMPLEX,  /* -1 = unknown, in which case assumed double;
                           *  0 = float, 1 = double.
                           */

      INT_LENGTH,  /* number of bytes which each stored integer occupies; 
                      default is 4; only currently supported size is also 4
                    */
      VAL_LENGTH   /* number of bytes which each value occupies;
                      default is 8; only currently supported size is 8
                    */
};

/*===== (end) header formatting secret decoder ring ========================*/

#define ROW_COUNT_header     0
#define NZ_header            1
#define BLOCK_SIZE_header    2
#define ROWS_SORTED_header   3 /* if nonzero, rows are ordered within the file */
#define COLS_SORTED_header   4 /* if nonzero, col in each row are in sorted order */
#define IS_UPPER_TRI         5 /* -1 = not determined;  1 = true;  2 = false */
#define IS_LOWER_TRI         6 /* -1 = not determined;  1 = true;  2 = false */

/*
CAUTION: 1.  currently only implemented for block size of 1;
             a few functions need to be changed for other sized.
         2.  several places it's assumed that rows are ordered
             within the binary file.
*/

/* data-block format: data consists of a series of records, each of
     which has the following entries:
       int[1] row number
       int[1] length
       int[length] column indices
       double[length*blocksize*blocksize] values
*/

static void swap_int_private(int* dataIN, int num);
static void swap_double_private(double* dataIN, int num);

static void write_bin_row_private(FILE *fpIN, int rowIN, int lenIN, 
                                           int *colIN, double *valIN);
static void write_header_private(FILE *fpIN, int *headerIN);
static void write_footer_private(FILE *fpIN);

static void read_header_private(FILE *fpIN, int *headerOUT);
void read_footer_private(FILE *fpIN, int *footerOUT);
static void read_bin_row_private(FILE *fpIN, int *rowOUT, int *lenOUT, 
                                           int *cvalOUT, double *avalOUT);

static void check_type_lengths();
static void position_fp_at_row_private(FILE *fpIN, int begRowIN);
static void compute_rowstorage_mpi_private(FILE *fpIN, int begRowIN, 
                                            int rowCountIN, int *nzOUT);

#undef __FUNC__
#define __FUNC__  "read_triples_private"
static void read_triples_private(FILE *fp, int m, int nz, int **rpOUT, 
                                         int **cvalOUT, double **avalOUT)
{ 
  START_FUNC_DH
  if (ignoreMe) SET_V_ERROR("why are you here?");
  END_FUNC_DH
}


static void computeLocalDimensions(char *filenameIN, int num_procsIN, 
                    int myidIN, int *beg_rowOUT, int *local_row_countOUT);

/*-----------------------------------------------------------------------------*/
#define IS_UPPER_TRI    5
#define IS_LOWER_TRI    6
#define IS_FULL         987
#define IS_UNKNOWN_TRI  8
/*-----------------------------------------------------------------------------*/

bool triples_are_col_sorted(FILE *fpIN, 
                       int *maxRowLengthOUT, int *nOUT, int *nzOUT);
static bool triples_are_row_sorted(FILE *fpIN, 
                       int *maxRowLengthOUT, int *nOUT, int *nzOUT);
  /* Returns true if triples are sorted by rows within the file.
     Also returns "maxRowLengthOUT," the number of column indices
     in the longest row; this yuck dual-functionality is employed
     to reduce the number of times we must read through the file.
     For similar reasons, returns the total number of rows in nOUT,
     and the number of triples in nzOUT.
     Returned values maxRowLengthOUT, nOUT, nzOUT are only valid when
     function returns true.
   */

static void write_bin_row_sorted_triples_private(FILE *fpIN, FILE *fpOUT) {}
static void read_txt_row_private(FILE *fpIN, int *rowOUT, int *lenOUT, 
                                            int *cvalOUT, double *avalOUT);

/* suppress compiler complaints about unused static functions! */
void io_dh_junk()
{
  computeLocalDimensions(NULL,0,0,NULL,NULL);
  triples_are_row_sorted(NULL, NULL, NULL, NULL);
  write_bin_row_sorted_triples_private(NULL,NULL);
  read_triples_private(NULL,0,0,NULL,NULL,NULL);
  read_txt_row_private(NULL, NULL, NULL, NULL, NULL);
}

/*=========================================================================*/
#undef __FUNC__
#define __FUNC__ "io_dh_print_ebin_vec_private"
void io_dh_print_ebin_vec_private(int n, int beg_row, double *vals,
                   int *n2o, int *o2n, Hash_i_dh hash, char *filename)
{
  START_FUNC_DH
  FILE *fp;
  int num;

  fp = openFile_dh(filename, "w"); CHECK_V_ERROR;

  if (isSmallEndian()) {
    swap_double_private(vals, n);  CHECK_V_ERROR;
    swap_int_private(&n, 1);  CHECK_V_ERROR;
  }

  /* print length */
  if ((num = fwrite(&n, sizeof(int), 1, fp)) != 1) {
    sprintf(msgBuf_dh, "fwrite failed; returned %i items writ; should be %i\n", num, 1);
  }

  /* print values */
  if ((num = fwrite(vals, sizeof(int), n, fp)) != n) {
    sprintf(msgBuf_dh, "fwrite failed; returned %i items writ; should be %i\n", num, n);
  }

  if (isSmallEndian()) {
    swap_double_private(vals, n);  CHECK_V_ERROR;
  }

  closeFile_dh(fp); CHECK_V_ERROR;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "io_dh_read_ebin_vec_private"
void io_dh_read_ebin_vec_private(int *n, double **vals, char *filename)
{
  START_FUNC_DH
  int num, m;
  FILE *fp;
  double *v;

  fp = openFile_dh(filename, "r"); CHECK_V_ERROR;

  if (fread(&m, sizeof(int), 1, fp)  != 1) {
    sprintf(msgBuf_dh, "fread failed to read vector length");
    SET_V_ERROR(msgBuf_dh);
  }
  if (isSmallEndian()) {
    swap_int_private(&m, 1);  CHECK_V_ERROR;
  }
  *n = m;

  v = *vals = (double*)MALLOC_DH(m*sizeof(double)); CHECK_V_ERROR;
  num = fread(v, sizeof(double), m, fp);
  if (num != m) {
    sprintf(msgBuf_dh, "fread failed; read %i items, should be %i", num, m);
    SET_V_ERROR(msgBuf_dh);
  }
  if (isSmallEndian()) {
    swap_double_private(v, m);  CHECK_V_ERROR;
  }

  closeFile_dh(fp); CHECK_V_ERROR;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "io_dh_read_ebin_mat_private"
extern void io_dh_read_ebin_mat_private(int *mOUT, int **rpOUT, int **cvalOUT, 
                                     double **avalOUT, char *filename)
{
  START_FUNC_DH
  int i, m, nz, head[HEADER_SIZE_DH];
  int row, len, offset, *rp, *cval;
  double *aval;
  FILE *fp;

/* puts("STARTING: io_dh_read_ebin_private\n"); */

  /* ensure int and double lengths are supported! */
  check_type_lengths(); CHECK_V_ERROR;

  fp = openFile_dh(filename, "r"); CHECK_V_ERROR;
  read_header_private(fp, head); CHECK_V_ERROR;
  m = head[ROW_COUNT_header];
  nz = head[NZ_header];

  sprintf(msgBuf_dh, "io_dh_read_ebin_private: m= %i  nz= %i", m, nz);
  SET_INFO(msgBuf_dh);

  *mOUT = m;
  rp = *rpOUT = (int*)MALLOC_DH((m+1)*sizeof(int)); CHECK_V_ERROR;
  cval = *cvalOUT = (int*)MALLOC_DH(nz*sizeof(int)); CHECK_V_ERROR;
  aval = *avalOUT = (double*)MALLOC_DH(nz*sizeof(double)); CHECK_V_ERROR;
  rp[0] = 0;
  offset = 0;

  /* could do more error checking here! */
  for (i=0; i<m; ++i) {
    read_bin_row_private(fp, &row, &len, cval+offset, aval+offset); CHECK_V_ERROR;

    offset += len;
    rp[i+1] = offset;
  }

  closeFile_dh(fp); CHECK_V_ERROR;
  END_FUNC_DH
}


/* rows must be ordered within file */
#undef __FUNC__
#define __FUNC__ "readBinary_mpi"
void readBinary_mpi(char *filenameIN, int begRowIN, int rowCountIN, Mat_dh *Aout)
{
  START_FUNC_DH
  int i, m, nz, head[HEADER_SIZE_DH];
  int ierr, pe, row, len, offset, *rp, *cval;
  double *aval;
  FILE *fp;
  Mat_dh A;

  /* ensure int and double lengths are supported! */
  check_type_lengths(); CHECK_V_ERROR;

  Mat_dhCreate(&A); CHECK_V_ERROR;
  *Aout = A;

  for (pe=0; pe<np_dh; ++pe) {
    ierr = MPI_Barrier(comm_dh); CHECK_MPI_V_ERROR(ierr);
    if (myid_dh == pe) {

      fp = openFile_dh(filenameIN, "r"); CHECK_V_ERROR;
      read_header_private(fp, head); CHECK_V_ERROR;
      m = head[ROW_COUNT_header];
      nz = head[NZ_header];

      sprintf(msgBuf_dh, "m= %i  nz= %i\n", m, nz);
      SET_INFO(msgBuf_dh);

      A->m = rowCountIN;
      A->beg_row = begRowIN;
      rp = A->rp = (int*)MALLOC_DH((m+1)*sizeof(int)); CHECK_V_ERROR;
      rp[0] = 0;

      /* determine storage requirements for this processor's rows */
      compute_rowstorage_mpi_private(fp, begRowIN, rowCountIN, &nz); CHECK_V_ERROR;

      /* allocate storage for row indices and values */
      cval = A->cval = (int*)MALLOC_DH(nz*sizeof(int)); CHECK_V_ERROR;
      aval = A->aval = (double*)MALLOC_DH(nz*sizeof(double)); CHECK_V_ERROR;
      offset = 0;

      /* position fp to start of first local row */
      position_fp_at_row_private(fp, begRowIN); CHECK_V_ERROR; 

      for (i=0, row = begRowIN; i<rowCountIN; ++i, ++begRowIN) {
        read_bin_row_private(fp, &row, &len, cval+offset, aval+offset); CHECK_V_ERROR;
        offset += len;
        rp[i+1] = offset;
      }

      closeFile_dh(fp); CHECK_V_ERROR;
    }
  }

  /* determine global matrix dimension */
  if (np_dh == 1) {
    A->n = m;
  } else {
    int mGlobal;
    MPI_Allreduce(&m, &mGlobal, 1, MPI_INT, MPI_SUM, comm_dh);
    A->n = mGlobal;
  }

  END_FUNC_DH
}




#undef __FUNC__
#define __FUNC__ "openFile_dh"
FILE * openFile_dh(const char *filenameIN, const char *modeIN)
{
  START_FUNC_DH
  FILE *fp = NULL;

  if ((fp = fopen(filenameIN, modeIN)) == NULL) {
    sprintf(msgBuf_dh, "can't open file: %s for mode %s\n", filenameIN, modeIN);
    SET_ERROR(NULL, msgBuf_dh);
  }
  END_FUNC_VAL(fp)
}

#undef __FUNC__
#define __FUNC__ "closeFile_dh"
void closeFile_dh(FILE *fpIN)
{
  if (fclose(fpIN)) {
    SET_V_ERROR("attempt to close file failed");
  }
}

/* fpIN is unaltered on successful return;
   if error is set, may be altered.
 */
#undef __FUNC__
#define __FUNC__ "triples_are_row_sorted"
bool triples_are_row_sorted(FILE *fpIN, 
                       int *maxRowLengthOUT, int *nOUT, int *nzOUT)
{
  START_FUNC_DH
  int n, nz, items, i, j, curRow = -1; /* -1 is to suppress compiler warning */
  int c, maxLen;
  fpos_t fpos;
  bool isSorted = true;
  double v;

  if (fgetpos(fpIN, &fpos)) {
    SET_ERROR(-1, "fgetpos failed!");
  }

  n = nz = c = maxLen = 0;
  while (!feof(fpIN)) {
    items = fscanf(fpIN,"%d %d %lg",&i,&j,&v);

    if (items != 3) break;
    ++c;     /* number of items in current row */
    ++nz;    /* total number of triples (nz in matrix) */

    if (nz == 1) {  /* first triple */
      curRow = i;
      n = 1;
    } else {
      if (i < curRow) {
        isSorted = false;
        sprintf(msgBuf_dh, "rows are not sorted, triple %i", nz);
        SET_INFO(msgBuf_dh);
        break;
      } 

      else if (i > curRow) {
        ++n;
        maxLen = MAX(maxLen, (c-1));
        c = 0;
        curRow = i;
      }
    }
  }

  if (fsetpos(fpIN, &fpos)) {
    SET_ERROR(-1, "fsetpos failed!");
  }

  *maxRowLengthOUT = maxLen;
  *nOUT = n;
  *nzOUT = nz;

  END_FUNC_VAL(isSorted)
}

/* fpIN is unaltered on successful return;
   if error is set, may be altered.
 */
#undef __FUNC__
#define __FUNC__ "triples_are_col_sorted"
bool triples_are_col_sorted(FILE *fpIN, 
                       int *maxRowLengthOUT, int *nOUT, int *nzOUT)
{
  START_FUNC_DH
  int n, nz, items, i, j, curCol = -1; /* -1 is to suppress compiler warning */
  int c, maxLen;
  fpos_t fpos;
  bool isSorted = true;
  double v;

  if (fgetpos(fpIN, &fpos)) {
    SET_ERROR(-1, "fgetpos failed!");
  }

  n = nz = c = maxLen = 0;
  while (!feof(fpIN)) {
    items = fscanf(fpIN,"%d %d %lg",&i,&j,&v);
    if (items != 3) break;
    ++c;     /* number of items in current col */
    ++nz;    /* total number of triples (nz in matrix) */

    if (nz == 1) {  /* first triple */
      curCol = i;
      n = 1;
    } else {
      if (i < curCol) {
        isSorted = false;
        sprintf(msgBuf_dh, "cols are not sorted, triple %i", nz);
        SET_INFO(msgBuf_dh);
        break;
      } 

      else if (i > curCol) {
        ++n;
        maxLen = MAX(maxLen, (c-1));
        c = 0;
        curCol = i;
      }
    }
  }

  if (fsetpos(fpIN, &fpos)) {
    SET_ERROR(-1, "fsetpos failed!");
  }

  *maxRowLengthOUT = maxLen;
  *nOUT = n;
  *nzOUT = nz;
  END_FUNC_VAL(isSorted)
}


#undef __FUNC__
#define __FUNC__ "write_header_private"
void write_header_private(FILE *fpIN, int *headerIN)
{
  START_FUNC_DH
  int num;

  if (isSmallEndian()) {
    swap_int_private(headerIN, HEADER_SIZE_DH);
  }

  rewind(fpIN);
  num = fwrite(headerIN, sizeof(int), HEADER_SIZE_DH, fpIN);
  if (isSmallEndian()) {
    swap_int_private(headerIN, HEADER_SIZE_DH);
  }

  if (num != HEADER_SIZE_DH) {
    sprintf(msgBuf_dh, "fwrite failed; returned %i items writ; should be %i\n",
                                                      num, HEADER_SIZE_DH);
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "isSmallEndian"
bool isSmallEndian()
{
  START_FUNC_DH
  bool retval = true;
  int x = 255;
  if ( x >> 8 ) retval = false;       /* is big endian */
  END_FUNC_VAL(retval)
}


/* this code taken from PETSc, ANL */
#undef __FUNC__
#define __FUNC__ "swap_int_private"
void swap_int_private(int* buff, int n)
{
  START_FUNC_DH
#if 0
  int  i,j,tmp =0;
  int  *tptr = &tmp;                /* Need to access tmp indirectly to get */
  char *ptr1,*ptr2 = (char*)&tmp;   /* arround the bug in DEC-ALPHA g++ */
                                   
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff + j);
    for (i=0; i<sizeof(int); i++) {
      ptr2[i] = ptr1[sizeof(int)-1-i];
    }
    buff[j] = *tptr;
  }
#endif
  END_FUNC_DH
}

/* this code adopted from PETSc, ANL */
#undef __FUNC__
#define __FUNC__ "swap_double_private"
void swap_double_private(double* buff, int n)
{
  START_FUNC_DH
#if 0
  int    i,j;
  double tmp,*buff1 = (double*)buff;
  double *tptr = &tmp;          /* take care pf bug in DEC-ALPHA g++ */
  char   *ptr1,*ptr2 = (char*)&tmp;

  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff1 + j);
    for (i=0; i<sizeof(double); i++) {
      ptr2[i] = ptr1[sizeof(double)-1-i];
    }
    buff1[j] = *tptr;
  }
#endif
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "write_bin_row_private"
void write_bin_row_private(FILE *fpIN, int row, int len, int *col, double *val)
{
  START_FUNC_DH
  int len2, smallEndian;

  smallEndian = isSmallEndian();
  if (smallEndian) {
    len2 = len;
    swap_int_private(&row, 1); 
    swap_int_private(&len, 1); 
    swap_int_private(col, len2); 
    swap_double_private(val, len2); 
  }

  /* write row number */
  if (fwrite(&row, sizeof(int), 1, fpIN)  != 1) {
    sprintf(msgBuf_dh, "fwrite failed to write row number %i", row);
    SET_V_ERROR(msgBuf_dh);
  }

  /* write row length */
  if (fwrite(&len, sizeof(int), 1, fpIN)  != 1) {
    sprintf(msgBuf_dh, "fwrite failed to write length= %i for row= %i", len, row);
    SET_V_ERROR(msgBuf_dh);
  }

  /* write indices */
  if (fwrite(col, sizeof(int), len, fpIN)  != len) {
    sprintf(msgBuf_dh, "fwrite failed to write column indices for row= %i\n", row);
    SET_V_ERROR(msgBuf_dh);
  }

  /* write values */
  if (fwrite(val, sizeof(double), len, fpIN)  != len) {
    sprintf(msgBuf_dh, "fwrite failed to write values for row= %i\n", row);
    SET_V_ERROR(msgBuf_dh);
  }

  if (smallEndian) {
    swap_int_private(col, len2); 
    swap_double_private(val, len2); 
  }
  END_FUNC_DH
}

/*===============================*/

/* seq only; reordering not implemented */
#undef __FUNC__
#define __FUNC__ "io_dh_print_ebin_mat_private"
void io_dh_print_ebin_mat_private(int m, int beg_row, 
                               int *rp, int *cval, double *aval, 
                           int *n2o, int *o2n, Hash_i_dh hash, char *filename)
{
  START_FUNC_DH
  int i, row, head[HEADER_SIZE_DH];
  FILE *fp;

#if 0
  /* allocate objects, if none were supplied on invocation */
  if (n2o == NULL) {
    private_n2o = true;
    create_nat_ordering_private(m, &n2o); CHECK_V_ERROR;
    create_nat_ordering_private(m, &o2n); CHECK_V_ERROR;
  }
  if (hash == NULL) {
    private_hash = true;
    Hash_i_dhCreate(&hash, -1); CHECK_V_ERROR;
  }
#endif


  fp = openFile_dh(filename, "w"); CHECK_V_ERROR;

  /* write junk header; this is so fp will be properly positioned;
     later, we'll over-write with a meaningful header.
   */
  for (i=0; i<HEADER_SIZE_DH; ++i) head[i] = -1;
  write_header_private(fp, head); CHECK_V_ERROR;

  /* write each row in matrix */
  for (row=0; row<m; ++row) {
    int len = rp[row+1]-rp[row];
    write_bin_row_private(fp, row, len, cval+rp[row], aval+rp[row]); CHECK_V_ERROR;
  }


  /* now write the real header; also write the footer */
  head[ROW_COUNT_header] = m;
  head[NZ_header] = rp[m];
/*
  for possible future expansion?:

  head[BLOCK_SIZE_header] = 1;
  head[ROWS_SORTED_header] = 1;
  head[COLS_SORTED_header] = cols_are_sorted;
*/
  write_header_private(fp, head); CHECK_V_ERROR;
  write_footer_private(fp); CHECK_V_ERROR;

#if 0
  /* clean up */
  if (private_n2o) {
    destroy_nat_ordering_private(n2o); CHECK_V_ERROR;
    destroy_nat_ordering_private(o2n); CHECK_V_ERROR;
  }
  if (private_hash) {
    Hash_i_dhDestroy(hash); CHECK_V_ERROR;
  }
#endif

  closeFile_dh(fp); CHECK_V_ERROR;
  END_FUNC_DH
}
/*===============================*/

#undef __FUNC__
#define __FUNC__ "read_txt_row_private"
void read_txt_row_private(FILE *fpIN, int *rowOUT, int *lenOUT, 
                                     int *cvalOUT, double *avalOUT)
{
  START_FUNC_DH
  int items, row, rrow, col, len = 0;
  double val;
  fpos_t fpos;

  while (!feof(fpIN)) {
    if (fgetpos(fpIN, &fpos)) { SET_V_ERROR("fgetpos failed!"); }
    items = fscanf(fpIN,"%d %d %lg",&rrow, &col, &val);
    if (items != 3) break;
    if (len == 0) { 
      row = rrow; 
    }
    if (rrow != row) {
      if (fsetpos(fpIN, &fpos)) {
        SET_V_ERROR("fsetpos failed!");
      }
      break;
    }
    cvalOUT[len] = col - 1;    
    avalOUT[len] = val;
    ++len;
  }

  *rowOUT = row - 1;
  *lenOUT = len;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "read_bin_row_private"
void read_bin_row_private(FILE *fpIN, int *rowOUT, int *lenOUT, 
                                           int *cvalOUT, double *avalOUT)
{
  START_FUNC_DH
  int row, len;
  int smallEndian = isSmallEndian();

  /* read row number */
  if (fread(&row, sizeof(int), 1, fpIN)  != 1) {
    sprintf(msgBuf_dh, "fread failed to read row number");
    SET_V_ERROR(msgBuf_dh);
  }

  /* read row length */
  if (fread(&len, sizeof(int), 1, fpIN)  != 1) {
    sprintf(msgBuf_dh, "fread failed to read for row= %i", row);
    SET_V_ERROR(msgBuf_dh);
  }

  /* read column indices */
  if (fread(cvalOUT, sizeof(int), len, fpIN)  != len) {
    sprintf(msgBuf_dh, "fread failed to read column indices for row= %i\n", row);
    SET_V_ERROR(msgBuf_dh);
  }

  /* read values */
  if (fread(avalOUT, sizeof(double), len, fpIN)  != len) {
    sprintf(msgBuf_dh, "fread failed to read values for row= %i\n", row);
    SET_V_ERROR(msgBuf_dh);
  }

  if (smallEndian) {
    swap_int_private(cvalOUT, len); 
    swap_double_private(avalOUT, len); 
    swap_int_private(&row, 1); 
    swap_int_private(&len, 1); 
  }

  *rowOUT = row;
  *lenOUT = len;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "read_header_private"
void read_header_private(FILE *fpIN, int *headerOUT)
{
  START_FUNC_DH
  int num;

  rewind(fpIN);
  num = fread(headerOUT, sizeof(int), HEADER_SIZE_DH, fpIN);
  if (num != HEADER_SIZE_DH) {
    sprintf(msgBuf_dh, "fread failed for header; returned %i, should be %i", 
                                                    num, HEADER_SIZE_DH);
    SET_V_ERROR(msgBuf_dh);
  }

  if (isSmallEndian()) {
    swap_int_private(headerOUT, HEADER_SIZE_DH);
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "check_type_lengths"
void check_type_lengths()
{
  START_FUNC_DH
  if (sizeof(double) != 8)  {
    sprintf(msgBuf_dh, "only supported for 8 byte doubles; yours is %i\n", 
                                                           sizeof(double));
    SET_V_ERROR(msgBuf_dh);
  }

  if (sizeof(int) != 4) {
    sprintf(msgBuf_dh, "only supported for 4 byte ints; yours is %i\n", sizeof(int));
    SET_V_ERROR(msgBuf_dh);
  }
  END_FUNC_DH
}

/* assumes rows are stored contiguously (are ordered within the file) */
#undef __FUNC__
#define __FUNC__ "compute_rowstorage_mpi_private"
void compute_rowstorage_mpi_private(FILE *fpIN, int begRowIN, 
                                    int rowCountIN, int *nzOUT)
{
  START_FUNC_DH
  int num, i, nz = 0;
  int len, buf[2];
  int smallEndian = isSmallEndian();
  fpos_t fpos;

  position_fp_at_row_private(fpIN, begRowIN); CHECK_V_ERROR;  
  if (fgetpos(fpIN, &fpos)) SET_V_ERROR("fgetpos failed!");

  for (i=0; i<rowCountIN; ++i) {

    /* get row number and length */
    num = fread(buf, sizeof(int), 2, fpIN);
    if (num != 2) {
      sprintf(msgBuf_dh, "fread failed; returned %i, should be 2", num);
      SET_V_ERROR(msgBuf_dh);
    }

    if (smallEndian) { swap_int_private(buf, 2); CHECK_V_ERROR; }
    len = buf[1];
    nz += len;

    /* position pointer to start of next row */
    fpos += 8+ len*sizeof(int) + len*sizeof(double);
    if (fsetpos(fpIN, &fpos)) {
      SET_V_ERROR("fsetpos failed!");
    }
  }

  *nzOUT = nz;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "position_fp_at_row_private"
void position_fp_at_row_private(FILE *fpIN, int begRowIN)
{
  START_FUNC_DH
  fpos_t fpos;
  /* bool success = false; */
  int num, row, len, buf[2];
  int smallEndian = isSmallEndian();

  fpos = HEADER_SIZE_DH*sizeof(int);
  if (fsetpos(fpIN, &fpos)) {
    SET_V_ERROR("fsetpos failed!");
  }

  /* fpIN is now at start of first row */
  while (!feof(fpIN)) {
    if (fgetpos(fpIN, &fpos)) SET_V_ERROR("fgetpos failed!");

    /* get row number and length */
    num = fread(buf, sizeof(int), 2, fpIN);
    if (num != 2) {
      sprintf(msgBuf_dh, "fread failed; returned %i, should be 2", num);
      SET_V_ERROR(msgBuf_dh);
    }
    if (smallEndian) swap_int_private(buf, 2); CHECK_V_ERROR;
    row = buf[0];
    len = buf[1];

    /* if it's the row we want, we're finished */
    if (row == begRowIN) {
      /* success = true; */
      if (fsetpos(fpIN, &fpos)) SET_V_ERROR("fgetpos failed!");
      break;
    }

    /* haven't found the row, so position fpIN to start of next row */
    fpos += 2*sizeof(int) + len*sizeof(int) + len*sizeof(double);
    if (fsetpos(fpIN, &fpos)) {
      SET_V_ERROR("fsetpos failed!");
    }
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "write_footer_private"
void write_footer_private(FILE *fpIN)
{
  START_FUNC_DH
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "read_footer_private"
void read_footer_private(FILE *fpIN, int *footerOUT)
{
  START_FUNC_DH
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "computeLocalDimensions"
void computeLocalDimensions(char *filenameIN, int num_procsIN, int myidIN,
                            int *beg_rowOUT, int *local_row_countOUT)
{
  START_FUNC_DH
  int mGlobal, mLocal;
  int header[HEADER_SIZE_DH];
  FILE *fp;

  /* get number of rows in matrix */
  fp = openFile_dh(filenameIN, "r"); CHECK_V_ERROR;
  read_header_private(fp, header); CHECK_V_ERROR;
  closeFile_dh(fp); CHECK_V_ERROR;
  mGlobal = header[ROW_COUNT_header];

  mLocal = mGlobal / num_procsIN;
  *beg_rowOUT = myidIN*mLocal;
  *local_row_countOUT = mLocal;

  /* last processor is special case */
  if (myidIN == num_procsIN-1) {
    mLocal = mGlobal - *beg_rowOUT;
    *local_row_countOUT = mLocal;
  }
  END_FUNC_DH
}


