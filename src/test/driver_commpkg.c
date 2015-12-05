/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.12 $
 ***********************************************************************EHEADER*/

/* driver_commpkg.c*/
/* AHB 06/04 */
/* purpose:  to test a new communication package for the ij interface */

/* 11/06 - if you want to use this, the the hypre_NewCommPkgCreate has to be
   reinstated in parcsr_mv/new_commpkg.c - currently it won't compile*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
/*   #include <mpi.h>   */

#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_parcsr_ls.h"

/* #include "_hypre_parcsr_ls.h"
 #include "HYPRE.h" 
 #include "HYPRE_parcsr_mv.h" 
 #include "HYPRE_krylov.h"  */



/*some debugging tools*/
#define   mydebug 0
#define   mpip_on 0

/*time an allgather in addition to the current commpkg - 
  since the allgather happens outside of the communication package.*/
#define   time_gather 1

/* for timing multiple commpkg setup (if you want the time to be larger in the
   hopes of getting smaller stds - often not effective) */
#define   LOOP2  1


HYPRE_Int myBuildParLaplacian (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr, HYPRE_Int parmprint );
HYPRE_Int myBuildParLaplacian27pt (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr, HYPRE_Int parmprint );


void stats_mo(double*, HYPRE_Int, double *,double *);

/*==========================================================================*/


/*------------------------------------------------------------------
 *
 * This tests an alternate comm package for ij
 *
 * options:
 *         -laplacian              3D 7pt stencil
 *         -27pt                   3D 27pt laplacian
 *         -fromonecsrfile         read matrix from a csr file
 *         -commpkg <HYPRE_Int>          1 = new comm. package
 *                                 2  =old
 *                                 3 = both (default)
 *         -loop <HYPRE_Int>             number of times to loop (default is 0)
 *         -verbose                print more error checking   
 *         -noparmprint            don't print the parameters 
 *-------------------------------------------------------------------*/


HYPRE_Int
main( HYPRE_Int   argc,
      char *argv[] )
{


   HYPRE_Int        num_procs, myid;
   HYPRE_Int        verbose = 0, build_matrix_type = 1;
   HYPRE_Int        index, matrix_arg_index, commpkg_flag=3;
   HYPRE_Int        i, k, ierr=0;
   HYPRE_Int        row_start, row_end; 
   HYPRE_Int        col_start, col_end, global_num_rows;
   HYPRE_Int       *row_part, *col_part; 
   char      *csrfilename;
   HYPRE_Int        preload = 0, loop = 0, loop2 = LOOP2;   
   HYPRE_Int        bcast_rows[2], *info;
   


   hypre_ParCSRMatrix    *parcsr_A, *small_A;
   HYPRE_ParCSRMatrix    A_temp, A_temp_small; 
   hypre_CSRMatrix       *A_CSR;
   hypre_ParCSRCommPkg	 *comm_pkg;   

  
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;
   HYPRE_Int                 p, q, r;
   double              values[4];

   hypre_ParVector     *x_new;
   hypre_ParVector     *y_new, *y;
   HYPRE_Int                 *row_starts;
   double              ans;
   double              start_time, end_time, total_time, *loop_times;
   double              T_avg, T_std;
   
   HYPRE_Int                   noparmprint = 0;
 
#if mydebug   
   HYPRE_Int  j, tmp_int;
#endif

   /*-----------------------------------------------------------
    * Initialize MPI
    *-----------------------------------------------------------*/


   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );



   /*-----------------------------------------------------------
    * default - is 27pt laplace
    *-----------------------------------------------------------*/

    
   build_matrix_type = 2;
   matrix_arg_index = argc;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
 
   index = 1;
   while ( index < argc) 
   {
      if  ( strcmp(argv[index], "-verbose") == 0 )
      {
         index++;  
         verbose = 1;
      }
      else if ( strcmp(argv[index], "-fromonecsrfile") == 0 )
      {
         index++;
         build_matrix_type      = 1;      
         matrix_arg_index = index; /*this tells where the name is*/
      }
      else if  ( strcmp(argv[index], "-commpkg") == 0 )
      {
         index++;  
         commpkg_flag = atoi(argv[index++]);
      }
      else if ( strcmp(argv[index], "-laplacian") == 0 )
      {
         index++;
         build_matrix_type      = 2;
         matrix_arg_index = index;
      }
      else if ( strcmp(argv[index], "-27pt") == 0 )
      {
         index++;
         build_matrix_type      = 4;
         matrix_arg_index = index;
      }
/*
      else if  ( strcmp(argv[index], "-nopreload") == 0 )
      {
         index++;  
         preload = 0;
      }
*/
      else if  ( strcmp(argv[index], "-loop") == 0 )
      {
         index++;  
         loop = atoi(argv[index++]);
      }
      else if  ( strcmp(argv[index], "-noparmprint") == 0 )
      {
         index++;  
         noparmprint = 1;
         
      }
      else  
      {
	 index++;
         /*hypre_printf("Warning: Unrecogized option '%s'\n",argv[index++] );*/
      }
   }
   
   
  
   /*-----------------------------------------------------------
    * Setup the Matrix problem   
    *-----------------------------------------------------------*/

  /*-----------------------------------------------------------
    *  Get actual partitioning- 
    *  read in an actual csr matrix.
    *-----------------------------------------------------------*/


   if (build_matrix_type ==1) /*read in a csr matrix from one file */
   {
      if (matrix_arg_index < argc)
      {
	 csrfilename = argv[matrix_arg_index];
      }
      else
      {
         hypre_printf("Error: No filename specified \n");
         exit(1);
      }
      if (myid == 0)
      {
	/*hypre_printf("  FromFile: %s\n", csrfilename);*/
         A_CSR = hypre_CSRMatrixRead(csrfilename);
      }
      row_part = NULL;
      col_part = NULL;

      parcsr_A = hypre_CSRMatrixToParCSRMatrix(hypre_MPI_COMM_WORLD, A_CSR, 
					       row_part, col_part);

      if (myid == 0) hypre_CSRMatrixDestroy(A_CSR);
   }
   else if (build_matrix_type ==2)
   {
      
      myBuildParLaplacian(argc, argv, matrix_arg_index,  &A_temp, !noparmprint);
     parcsr_A = (hypre_ParCSRMatrix *) A_temp;      
 
   }
   else if (build_matrix_type ==4)
   {
      myBuildParLaplacian27pt(argc, argv, matrix_arg_index, &A_temp, !noparmprint);
     parcsr_A = (hypre_ParCSRMatrix *) A_temp;
   }

 
  /*-----------------------------------------------------------
   * create a small problem so that timings are more accurate - 
   * code gets run twice (small laplace)
   *-----------------------------------------------------------*/

   /*this is no longer being used - preload = 0 is set at the beginning */

   if (preload == 1) 
   {
 
      /*hypre_printf("preload!\n");*/
      
        
       values[1] = -1;
       values[2] = -1;
       values[3] = -1;
       values[0] = - 6.0    ;

       nx = 2;
       ny = num_procs;
       nz = 2;

       P  = 1;
       Q  = num_procs;
       R  = 1;

       p = myid % P;
       q = (( myid - p)/P) % Q;
       r = ( myid - p - P*q)/( P*Q );
       
      A_temp_small = (HYPRE_ParCSRMatrix) GenerateLaplacian(hypre_MPI_COMM_WORLD, nx, ny, nz, 
				      P, Q, R, p, q, r, values);
      small_A = (hypre_ParCSRMatrix *) A_temp_small;     

      /*do comm packages*/
      hypre_NewCommPkgCreate(small_A);
      hypre_NewCommPkgDestroy(small_A); 

      hypre_MatvecCommPkgCreate(small_A);
      hypre_ParCSRMatrixDestroy(small_A); 
  
   }





   /*-----------------------------------------------------------
    *  Prepare for timing
    *-----------------------------------------------------------*/

   /* instead of preloading, let's not time the first one if more than one*/

    
   if (!loop)
   {
      loop = 1;
      /* and don't do any timings */
      
   }
   else
   {
      
      loop +=1;
      if (loop < 2) loop = 2;
   }
      
   
   loop_times = hypre_CTAlloc(double, loop);
   


/******************************************************************************************/   

   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);

   if (commpkg_flag == 1 || commpkg_flag ==3 )
   {
  
      /*-----------------------------------------------------------
       *  Create new comm package
       *-----------------------------------------------------------*/


    
      if (!myid) hypre_printf("********************************************************\n" );  
 
      /*do loop times*/
      for (i=0; i< loop; i++) 
      {
         loop_times[i] = 0.0;
         for (k=0; k< loop2; k++) 
         {
         
            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
            
            start_time = hypre_MPI_Wtime();

#if mpip_on
            if (i==(loop-1)) hypre_MPI_Pcontrol(1); 
#endif
     
            hypre_NewCommPkgCreate(parcsr_A);

#if mpip_on
            if (i==(loop-1)) hypre_MPI_Pcontrol(0); 
#endif  
  
            end_time = hypre_MPI_Wtime();
            
            end_time = end_time - start_time;
        
            hypre_MPI_Allreduce(&end_time, &total_time, 1,
                       hypre_MPI_DOUBLE, hypre_MPI_MAX, hypre_MPI_COMM_WORLD);
         
            loop_times[i] += total_time;

            if (  !((i+1)== loop  &&  (k+1) == loop2)) hypre_NewCommPkgDestroy(parcsr_A); 
            
         }/*end of loop2 */
      
        
      } /*end of loop*/
      


      /* calculate the avg and std. */
      if (loop > 1)
      {
         
         /* calculate the avg and std. */
         stats_mo(loop_times, loop, &T_avg, &T_std);
      
         if (!myid) hypre_printf(" NewCommPkgCreate:  AVG. wall clock time =  %f seconds\n", T_avg);  
         if (!myid) hypre_printf("                    STD. for %d  runs     =  %f\n", loop-1, T_std);  
         if (!myid) hypre_printf("                    (Note: avg./std. timings exclude run 0.)\n");
         if (!myid) hypre_printf("********************************************************\n" );  
         for (i=0; i< loop; i++) 
         {
            if (!myid) hypre_printf("      run %d  =  %f sec.\n", i, loop_times[i]);  
         }
         if (!myid) hypre_printf("********************************************************\n" );  
   
       }
       else 
       {
         if (!myid) hypre_printf("********************************************************\n" );  
         if (!myid) hypre_printf(" NewCommPkgCreate:\n");  
         if (!myid) hypre_printf("      run time =  %f sec.\n", loop_times[0]);  
         if (!myid) hypre_printf("********************************************************\n" );  
       }


     /*-----------------------------------------------------------
       *  Verbose printing
       *-----------------------------------------------------------*/

      /*some verification*/

       global_num_rows = hypre_ParCSRMatrixGlobalNumRows(parcsr_A); 

       if (verbose) 
       {

	  ierr = hypre_ParCSRMatrixGetLocalRange( parcsr_A,
                                      &row_start, &row_end ,
                                       &col_start, &col_end );


	  comm_pkg = hypre_ParCSRMatrixCommPkg(parcsr_A);
     
          hypre_printf("myid = %i, my ACTUAL local range: [%i, %i]\n", myid, 
		 row_start, row_end);
	  
	
	  ierr = hypre_GetAssumedPartitionRowRange( myid, global_num_rows, &row_start, 
					      &row_end);


	  hypre_printf("myid = %i, my assumed local range: [%i, %i]\n", myid, 
		 row_start, row_end);

          hypre_printf("myid = %d, num_recvs = %d\n", myid, 
		 hypre_ParCSRCommPkgNumRecvs(comm_pkg)  );  

#if mydebug   
	  for (i=0; i < hypre_ParCSRCommPkgNumRecvs(comm_pkg); i++) 
	  {
              hypre_printf("myid = %d, recv proc = %d, vec_starts = [%d : %d]\n", 
		     myid,  hypre_ParCSRCommPkgRecvProcs(comm_pkg)[i], 
		     hypre_ParCSRCommPkgRecvVecStarts(comm_pkg)[i],
		     hypre_ParCSRCommPkgRecvVecStarts(comm_pkg)[i+1]-1);
	   }
#endif 
	  hypre_printf("myid = %d, num_sends = %d\n", myid, 
		 hypre_ParCSRCommPkgNumSends(comm_pkg)  );  

#if mydebug
	  for (i=0; i <hypre_ParCSRCommPkgNumSends(comm_pkg) ; i++) 
          {
	    tmp_int =  hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i+1] -  
                     hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i];
	    index = hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i];
	    for (j=0; j< tmp_int; j++) 
	    {
	       hypre_printf("myid = %d, send proc = %d, send element = %d\n",myid,  
		      hypre_ParCSRCommPkgSendProcs(comm_pkg)[i],
		      hypre_ParCSRCommPkgSendMapElmts(comm_pkg)[index+j]); 
	     }   
	  }
#endif
       }
       /*-----------------------------------------------------------
        *  To verify correctness (if commpkg_flag = 3)
        *-----------------------------------------------------------*/

       if (commpkg_flag == 3 ) 
       {
          /*do a matvec - we are assuming a square matrix */
          row_starts = hypre_ParCSRMatrixRowStarts(parcsr_A);
   
          x_new = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, global_num_rows, row_starts);
          hypre_ParVectorSetPartitioningOwner(x_new, 0);
          hypre_ParVectorInitialize(x_new);
          hypre_ParVectorSetRandomValues(x_new, 1);    
          
          y_new = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, global_num_rows, row_starts);
          hypre_ParVectorSetPartitioningOwner(y_new, 0);
          hypre_ParVectorInitialize(y_new);
          hypre_ParVectorSetConstantValues(y_new, 0.0);
          
          /*y = 1.0*A*x+1.0*y */
          hypre_ParCSRMatrixMatvec (1.0, parcsr_A, x_new, 1.0, y_new);
       }
   
   /*-----------------------------------------------------------
    *  Clean up after MyComm
    *-----------------------------------------------------------*/


       hypre_NewCommPkgDestroy(parcsr_A); 

   }

  




/******************************************************************************************/
/******************************************************************************************/

   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);


   if (commpkg_flag > 1 )
   {

      /*-----------------------------------------------------------
       *  Set up standard comm package
       *-----------------------------------------------------------*/

      bcast_rows[0] = 23;
      bcast_rows[1] = 1789;
      
      if (!myid) hypre_printf("********************************************************\n" );  
      /*do loop times*/
      for (i=0; i< loop; i++) 
      {

         loop_times[i] = 0.0;
         for (k=0; k< loop2; k++) 
         {
            

            hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);

         
            start_time = hypre_MPI_Wtime();

#if time_gather
                  
            info = hypre_CTAlloc(HYPRE_Int, num_procs);
            
            hypre_MPI_Allgather(bcast_rows, 1, HYPRE_MPI_INT, info, 1, HYPRE_MPI_INT, hypre_MPI_COMM_WORLD); 

#endif

            hypre_MatvecCommPkgCreate(parcsr_A);

            end_time = hypre_MPI_Wtime();


            end_time = end_time - start_time;
        
            hypre_MPI_Allreduce(&end_time, &total_time, 1,
                          hypre_MPI_DOUBLE, hypre_MPI_MAX, hypre_MPI_COMM_WORLD);

            loop_times[i] += total_time;
         
       
         if (  !((i+1)== loop  &&  (k+1) == loop2))   hypre_MatvecCommPkgDestroy(hypre_ParCSRMatrixCommPkg(parcsr_A));
               
         }/* end of loop 2*/
         
        
      } /*end of loop*/
      
      /* calculate the avg and std. */
      if (loop > 1)
      {
         
         stats_mo(loop_times, loop, &T_avg, &T_std);      
         if (!myid) hypre_printf("Current CommPkgCreate:  AVG. wall clock time =  %f seconds\n", T_avg);  
         if (!myid) hypre_printf("                        STD. for %d  runs     =  %f\n", loop-1, T_std);  
         if (!myid) hypre_printf("                        (Note: avg./std. timings exclude run 0.)\n");
         if (!myid) hypre_printf("********************************************************\n" );  
         for (i=0; i< loop; i++) 
         {
            if (!myid) hypre_printf("      run %d  =  %f sec.\n", i, loop_times[i]);  
         }
         if (!myid) hypre_printf("********************************************************\n" );  
         
      }
      else 
      {
         if (!myid) hypre_printf("********************************************************\n" );  
         if (!myid) hypre_printf(" Current CommPkgCreate:\n");  
         if (!myid) hypre_printf("      run time =  %f sec.\n", loop_times[0]);  
         if (!myid) hypre_printf("********************************************************\n" );  
      }





      /*-----------------------------------------------------------
       * Verbose printing
       *-----------------------------------------------------------*/

      /*some verification*/

    
       if (verbose) 
       {

          ierr = hypre_ParCSRMatrixGetLocalRange( parcsr_A,
						  &row_start, &row_end ,
						  &col_start, &col_end );


          comm_pkg = hypre_ParCSRMatrixCommPkg(parcsr_A);
     
          hypre_printf("myid = %i, std - my local range: [%i, %i]\n", myid, 
		 row_start, row_end);

          ierr = hypre_ParCSRMatrixGetLocalRange( parcsr_A,
						  &row_start, &row_end ,
						  &col_start, &col_end );

          hypre_printf("myid = %d, std - num_recvs = %d\n", myid, 
		 hypre_ParCSRCommPkgNumRecvs(comm_pkg)  );  

#if mydebug   
	  for (i=0; i < hypre_ParCSRCommPkgNumRecvs(comm_pkg); i++) 
          {
              hypre_printf("myid = %d, std - recv proc = %d, vec_starts = [%d : %d]\n", 
		     myid,  hypre_ParCSRCommPkgRecvProcs(comm_pkg)[i], 
		     hypre_ParCSRCommPkgRecvVecStarts(comm_pkg)[i],
		     hypre_ParCSRCommPkgRecvVecStarts(comm_pkg)[i+1]-1);
	  }
#endif
          hypre_printf("myid = %d, std - num_sends = %d\n", myid, 
		 hypre_ParCSRCommPkgNumSends(comm_pkg));  


#if mydebug
          for (i=0; i <hypre_ParCSRCommPkgNumSends(comm_pkg) ; i++) 
          {
	     tmp_int =  hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i+1] -  
	                hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i];
	     index = hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i];
	     for (j=0; j< tmp_int; j++) 
	     {
	        hypre_printf("myid = %d, std - send proc = %d, send element = %d\n",myid,  
		       hypre_ParCSRCommPkgSendProcs(comm_pkg)[i],
		       hypre_ParCSRCommPkgSendMapElmts(comm_pkg)[index+j]); 
	     }   
	  } 
#endif
       }

       /*-----------------------------------------------------------
        * Verify correctness
        *-----------------------------------------------------------*/

 

       if (commpkg_flag == 3 ) 
       { 
          global_num_rows = hypre_ParCSRMatrixGlobalNumRows(parcsr_A); 
          row_starts = hypre_ParCSRMatrixRowStarts(parcsr_A);
 
       
          y = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, global_num_rows,row_starts);
          hypre_ParVectorSetPartitioningOwner(y, 0);
          hypre_ParVectorInitialize(y);
          hypre_ParVectorSetConstantValues(y, 0.0);

          hypre_ParCSRMatrixMatvec (1.0, parcsr_A, x_new, 1.0, y);
      
       }

   }






   /*-----------------------------------------------------------
    *  Compare matvecs for both comm packages (3)
    *-----------------------------------------------------------*/

   if (commpkg_flag == 3 ) 
   { 
     /*make sure that y and y_new are the same  - now y_new should=0*/   
     hypre_ParVectorAxpy( -1.0, y, y_new );


     hypre_ParVectorSetRandomValues(y, 1);

     ans = hypre_ParVectorInnerProd( y, y_new );
     if (!myid)
     {
        
        if ( fabs(ans) > 1e-8 ) 
        {  
           hypre_printf("!!!!! WARNING !!!!! should be zero if correct = %6.10f\n", 
                  ans); 
        } 
        else
        {
           hypre_printf("Matvecs match ( should be zero = %6.10f )\n", 
                  ans); 
        }
     }
     

   }
 

   /*-----------------------------------------------------------
    *  Clean up
    *-----------------------------------------------------------*/

    
   hypre_ParCSRMatrixDestroy(parcsr_A); /*this calls the standard comm 
                                          package destroy - but we'll destroy 
                                          ours separately until it is
                                          incorporated */

  if (commpkg_flag == 3 ) 
  { 

      hypre_ParVectorDestroy(x_new);
      hypre_ParVectorDestroy(y);
      hypre_ParVectorDestroy(y_new);
  }




   hypre_MPI_Finalize();

   return(ierr);


}





/*------------------------------------
 *    Calculate the average and STD   
 *     throw away 1st timing       
 *------------------------------------*/

void stats_mo(double array[], HYPRE_Int n, double *Tavg,double *Tstd)
{

    HYPRE_Int i;
    double atmp, tmp=0.0;
    double avg = 0.0, std;

  
    for(i=1; i<n; i++) {
       atmp = array[i];
       avg += atmp;
       tmp += atmp*atmp;
    }

    n = n-1;    
    avg = avg/(double) n;
    tmp = tmp/(double) n;

    tmp = fabs(tmp - avg*avg);
    std = sqrt(tmp);

    *Tavg = avg;
    *Tstd = std;
}



/*These next two functions are from ij.c in linear_solvers/tests */


/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
myBuildParLaplacian27pt( HYPRE_Int                  argc,
                       char                *argv[],
                       HYPRE_Int                  arg_index,
                         HYPRE_ParCSRMatrix  *A_ptr  , HYPRE_Int parmprint  )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0 && parmprint)
   {
      hypre_printf("  Laplacian_27pt:\n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 2);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
	values[0] = 8.0;
   if (nx*ny == 1 || nx*nz == 1 || ny*nz == 1)
	values[0] = 2.0;
   values[1] = -1.;

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(hypre_MPI_COMM_WORLD,
                               nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}


/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/


HYPRE_Int
myBuildParLaplacian( HYPRE_Int                  argc,
                   char                *argv[],
                   HYPRE_Int                  arg_index,
                     HYPRE_ParCSRMatrix  *A_ptr , HYPRE_Int parmprint    )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;
   double              cx, cy, cz;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0 && parmprint)
   {
      hypre_printf("  Laplacian:\n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 4);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0*cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz;
   }

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian(hypre_MPI_COMM_WORLD, nx, ny, nz, 
					      P, Q, R, p, q, r, values);

   hypre_TFree(values);


   *A_ptr = A;

   return (0);
}
