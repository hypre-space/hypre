#include <stdlib.h>
#include <assert.h>
#include "Common.h"
#include "Matrix.h"
#include "ParaSails.h"
#include "ConjGrad.h"

/* functions for instrumenting code */
/* compile using
  cc -bI:/usr/lpp/pmtoolkit/lib/pmsvcs.exp -o counter counter.c -lpmapi */

#include <stdio.h>
#if 0
#include "pmapi.h"

#define ERROR_CODE      -1
#define OK_CODE          0

void hcinit();
void hcreset(); /* will accumulate if not reset */
void hcstart(); /* use start and stop to surround sections to instrument */
void hcstop();  /* each stop will print a message */
void hcfinish();

void hcinit()
{
        pm_info_t myinfo;
        pm_prog_t prog;
        int rc;
        int filter = 0;

        prog.mode.w = 0;

        prog.events[0] = 15;   /* floating point ops */
        prog.events[1] =  6;   /* data cache misses */
        prog.events[2] =  1;   /* cycles */
        prog.events[3] =  2;   /* instr completed per cycle */

     /* prog.mode.w |= PM_KERNEL; *//* count in kernel mode */
        prog.mode.w |= PM_USER;     /* count in user mode */

        prog.mode.w |= PM_PROCESS;

        if ( (rc = pm_init(filter, &myinfo)) != OK_CODE) {
                pm_error("pm_init", rc);
                exit(ERROR_CODE);
        }

        prog.mode.b.count = 1;

        if ( (rc = pm_set_program_mygroup(&prog)) != OK_CODE) {
                pm_error("pm_set_program_mygroup", rc);
                exit(ERROR_CODE);
        }
}

void hcreset()
{
        int rc;

        if ( (rc = pm_reset_data_mygroup()) != OK_CODE)
                pm_error("pm_reset_data_mygroup", rc);
}

void hcstart()
{
        int rc;

        if ( (rc = pm_start_mygroup()) != OK_CODE)
                pm_error("pm_start_mygroup", rc);
}

void hcstop()
{
        pm_prog_t getprog;
        pm_data_t mydata;
        int rc;

        if ( (rc = pm_stop_mygroup()) != OK_CODE)
                pm_error("pm_stop_mygroup", rc);

        if ( (rc = pm_get_program_mygroup(&getprog)) != OK_CODE)
                pm_error("pm_get_program_mygroup", rc);

        if ( (rc = pm_get_data_mygroup(&mydata)) != OK_CODE)
                pm_error("pm_get_data_mygroup", rc);

        printf("--------------\n");
        printf("flops:  %-8lld\n", mydata.accu[0]);
        printf("dcmiss: %-8lld\n", mydata.accu[1]);
        printf("cycles: %-8lld\n", mydata.accu[2]);
        printf("instr:  %-8lld\n", mydata.accu[3]);
}

void hcfinish()
{
        int rc;

        if ( (rc = pm_delete_program_mygroup()) != OK_CODE)
                pm_error("pm_delete_program_mygroup", rc);
}
#endif

int rownum(const int x, const int y, const int z, 
   const int nx, const int ny, const int nz, int P, int Q)
{
   int p, q, r;
   int lowerx, lowery, lowerz;
   int id, startrow;

   p = (x-1) / nx;
   q = (y-1) / ny;
   r = (z-1) / nz;
   id = r*P*Q+q*P+p;
   startrow = id*(nx*ny*nz) + 1;
   lowerx = nx*p + 1;
   lowery = ny*q + 1;
   lowerz = nz*r + 1;
   
   return startrow + nx*ny*(z-lowerz) + nx*(y-lowery) + (x-lowerx);
}

int main(int argc, char *argv[]) 
{
   int                 npes, mype;

   int                 nx, ny, nz;
   int                 P, Q, R;
   double              dx, dy, dz;
   int                 p, q, r;
   int                 lowerx, lowery, lowerz;
   int                 upperx, uppery, upperz;
   int x, y, z;
   int num_rows;
   int row;
   int inds[100], *inds_p;
   double coefs[100], *coefs_p;
    int beg_row, end_row;
   double time0, time1;
    double setup_time, solve_time;
    double max_setup_time, max_solve_time;

   double *x0, *b;
   int i;
   Matrix *A;
   ParaSails *ps;

   MPI_Init(&argc, &argv);

   MPI_Comm_rank(MPI_COMM_WORLD, &mype);
   MPI_Comm_size(MPI_COMM_WORLD, &npes);

   if (argc > 9)
   {
      nx = atoi(argv[1]);
      ny = atoi(argv[2]);
      nz = atoi(argv[3]);
      P  = atoi(argv[4]);
      Q  = atoi(argv[5]);
      R  = atoi(argv[6]);
      dx = atof(argv[7]);
      dy = atof(argv[8]);
      dz = atof(argv[9]);
   }
   else
   {
      printf("Usage: mpirun -np %d %s <nx,ny,nz,P,Q,R,dx,dy,dz> ,\n\n",
             npes, argv[0]);
      printf("     where nx X ny X nz is the problem size per processor;\n");
      printf("           P  X  Q X  R is the processor topology;\n");
      printf("           dx, dy, dz   are the diffusion coefficients.\n");

      exit(1);
   }

   assert(npes == P*Q*R);

   printf("XX side: %4d npes: %4d n: %10d\n", nx, npes, npes*nx*ny*nz);

   /* compute p,q,r from P,Q,R and mype */
   p = mype % P;
   q = (( mype - p)/P) % Q;
   r = ( mype - p - P*q)/( P*Q );

   /* compute ilower and iupper from p,q,r and nx,ny,nz */
   /* 1-based */

   lowerx = nx*p + 1;
   upperx = nx*(p+1);
   lowery = ny*q + 1;
   uppery = ny*(q+1);
   lowerz = nz*r + 1;
   upperz = nz*(r+1);

   num_rows = P*Q*R*nx*ny*nz;
   beg_row = mype*(nx*ny*nz)+1;
   end_row = (mype+1)*(nx*ny*nz);

    x0 = (double *) malloc((end_row-beg_row+1) * sizeof(double));
    b  = (double *) malloc((end_row-beg_row+1) * sizeof(double));

        for (i=0; i<end_row-beg_row+1; i++)
	{
            b[i] = (double) (2*rand()) / (double) RAND_MAX - 1.0;
	    x0[i] = 0.0;
	}


    A = MatrixCreate(MPI_COMM_WORLD, beg_row, end_row);


   for (z=lowerz; z<=upperz; z++)
   for (y=lowery; y<=uppery; y++)
   for (x=lowerx; x<=upperx; x++)
   {
       int temp;

       coefs_p = coefs;
       inds_p = inds;
       row = rownum(x,y,z,nx,ny,nz,P,Q);

       *coefs_p++ = 2.0*(dx+dy+dz);      
       *inds_p++ = row;
       if (x != 1)    
	  {*coefs_p++ = -dx; *inds_p++ = rownum(x-1,y,z,nx,ny,nz,P,Q);}
       if (x != P*nx) 
	  {*coefs_p++ = -dx; *inds_p++ = rownum(x+1,y,z,nx,ny,nz,P,Q);}
       if (y != 1)    
	  {*coefs_p++ = -dy; *inds_p++ = rownum(x,y-1,z,nx,ny,nz,P,Q);}
       if (y != Q*ny) 
	  {*coefs_p++ = -dy; *inds_p++ = rownum(x,y+1,z,nx,ny,nz,P,Q);}
       if (z != 1)    
	  {*coefs_p++ = -dz; *inds_p++ = rownum(x,y,z-1,nx,ny,nz,P,Q);}
       if (z != R*nz) 
	  {*coefs_p++ = -dz; *inds_p++ = rownum(x,y,z+1,nx,ny,nz,P,Q);}

       temp = inds_p-inds;
       MatrixSetRow(A, row, temp, inds, coefs);
   }

   MatrixComplete(A);

        /**************
         * Setup phase
         **************/

/*
hcinit();
hcreset();
hcstart();
*/
        MPI_Barrier(MPI_COMM_WORLD);
        time0 = MPI_Wtime();

        ps = ParaSailsCreate(MPI_COMM_WORLD, beg_row, end_row, 1);

        ParaSailsSetupPattern(ps, A, .1, 3);
        ParaSailsSetupValues(ps, A, 0.00);

        time1 = MPI_Wtime();
/*
hcstop();
*/
        setup_time = time1-time0;
        fflush(NULL);
        MPI_Barrier(MPI_COMM_WORLD);

        /*****************
         * Solution phase
         *****************/

/*
hcreset();
hcstart();
*/
        MPI_Barrier(MPI_COMM_WORLD);
        time0 = MPI_Wtime();

/*
        PCG_ParaSails(A, ps, b, x0, 1.e-8, 1500);
*/

        time1 = MPI_Wtime();
/*
hcstop();
*/
        solve_time = time1-time0;

        ParaSailsStatsPattern(ps, A);
        ParaSailsStatsValues(ps, A);

        MPI_Reduce(&setup_time, &max_setup_time, 1, MPI_DOUBLE, MPI_MAX, 0,
            MPI_COMM_WORLD);
        MPI_Reduce(&solve_time, &max_solve_time, 1, MPI_DOUBLE, MPI_MAX, 0,
            MPI_COMM_WORLD);

        if (mype == 0)
        {
            printf("**********************************************\n");
            printf("***    Setup    Solve    Total\n");
            printf("III %8.1f %8.1f %8.1f\n", max_setup_time, max_solve_time,
                max_setup_time+max_solve_time);
            printf("**********************************************\n");
        }

/*
hcfinish();
*/

    ParaSailsDestroy(ps);
/*
if (mype == 6)
HashPrint(A->numb->hash);
*/
    MatrixDestroy(A);

    free(x0);
    free(b);

    MPI_Finalize();
}
