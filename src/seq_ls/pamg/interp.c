/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/





#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_AMGBuildInterp
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGBuildInterp( hypre_CSRMatrix  *A,
                   HYPRE_Int                 *CF_marker,
                   hypre_CSRMatrix     *S,
                   HYPRE_Int                 *dof_func,
                   HYPRE_Int                 **coarse_dof_func_ptr,
                   hypre_CSRMatrix     **P_ptr )
{
   
   double          *A_data;
   HYPRE_Int             *A_i;
   HYPRE_Int             *A_j;

   HYPRE_Int             *S_i;
   HYPRE_Int             *S_j;

   hypre_CSRMatrix    *P; 

   double          *P_data;
   HYPRE_Int             *P_i;
   HYPRE_Int             *P_j;

   HYPRE_Int              P_size;
   
   HYPRE_Int             *P_marker;

   HYPRE_Int             *coarse_dof_func;

   HYPRE_Int              jj_counter;
   HYPRE_Int              jj_begin_row;
   HYPRE_Int              jj_end_row;
   
   HYPRE_Int              start_indexing = 0; /* start indexing for P_data at 0 */

   HYPRE_Int              n_fine;
   HYPRE_Int              n_coarse;

   HYPRE_Int              strong_f_marker;

   HYPRE_Int             *fine_to_coarse;
   HYPRE_Int              coarse_counter;
   
   HYPRE_Int              i,i1,i2;
   HYPRE_Int              jj,jj1;
   HYPRE_Int              sgn;
   
   double           diagonal;
   double           sum;
   double           distribute;          
   
   double           zero = 0.0;
   double           one  = 1.0;
   
   /*-----------------------------------------------------------------------
    *  Access the CSR vectors for A and S. Also get size of fine grid.
    *-----------------------------------------------------------------------*/

   A_data = hypre_CSRMatrixData(A);
   A_i    = hypre_CSRMatrixI(A);
   A_j    = hypre_CSRMatrixJ(A);

   S_i    = hypre_CSRMatrixI(S);
   S_j    = hypre_CSRMatrixJ(S);

   n_fine = hypre_CSRMatrixNumRows(A);

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = 0;

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int, n_fine);

   jj_counter = start_indexing;
      
   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/
    
   for (i = 0; i < n_fine; i++)
   {
      
      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity. Also set up
       *  mapping vector.
       *--------------------------------------------------------------------*/

      if (CF_marker[i] >= 0)
      {
         jj_counter++;
         fine_to_coarse[i] = coarse_counter;
         coarse_counter++;
      }
      
      /*--------------------------------------------------------------------
       *  If i is a f-point, interpolation is from the C-points that
       *  strongly influence i.
       *--------------------------------------------------------------------*/

      else
      {
         for (jj = S_i[i]; jj < S_i[i+1]; jj++)
         {
            i1 = S_j[jj];           
            if (CF_marker[i1] >= 0)
            {
               jj_counter++;
            }
         }
      }
   }
   
   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   n_coarse = coarse_counter;

   P_size = jj_counter;

   P_i    = hypre_CTAlloc(HYPRE_Int, n_fine+1);
   P_j    = hypre_CTAlloc(HYPRE_Int, P_size);
   P_data = hypre_CTAlloc(double, P_size);

   P_marker = hypre_CTAlloc(HYPRE_Int, n_fine);

   /*-----------------------------------------------------------------------
    *  Second Pass: Define interpolation and fill in P_data, P_i, and P_j.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   for (i = 0; i < n_fine; i++)
   {      
      P_marker[i] = -1;
   }
   
   strong_f_marker = -2;

   jj_counter = start_indexing;
   
   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
    
   for (i = 0; i  < n_fine  ; i ++)
   {
             
      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity.
       *--------------------------------------------------------------------*/
      
      if (CF_marker[i] > 0)
      {
         P_i[i] = jj_counter;
         P_j[jj_counter]    = fine_to_coarse[i];
         P_data[jj_counter] = one;
         jj_counter++;
      }
      
      
      /*--------------------------------------------------------------------
       *  If i is a f-point, build interpolation.
       *--------------------------------------------------------------------*/

      else
      {
         P_i[i] = jj_counter;
         jj_begin_row = jj_counter;

         for (jj = S_i[i]; jj < S_i[i+1]; jj++)
         {
            i1 = S_j[jj];   

            /*--------------------------------------------------------------
             * If nieghbor i1 is a c-point, set column number in P_j and
             * initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] >= 0)
            {
               P_marker[i1] = jj_counter;
               P_j[jj_counter]    = fine_to_coarse[i1];
               P_data[jj_counter] = zero;
               jj_counter++;
            }

            /*--------------------------------------------------------------
             * If nieghbor i1 is a f-point, mark it as a strong f-point
             * whose connection needs to be distributed.
             *--------------------------------------------------------------*/

            else
            {
               P_marker[i1] = strong_f_marker;
            }            
         }

         jj_end_row = jj_counter;
         
         diagonal = A_data[A_i[i]];
         
         for (jj = A_i[i]+1; jj < A_i[i+1]; jj++)
         {
            i1 = A_j[jj];

            /*--------------------------------------------------------------
             * Case 1: nieghbor i1 is a c-point and strongly influences i,
             * accumulate a_{i,i1} into the interpolation weight.
             *--------------------------------------------------------------*/

            if (P_marker[i1] >= jj_begin_row)
            {
               P_data[P_marker[i1]] += A_data[jj];
            }
 
            /*--------------------------------------------------------------
             * Case 2: nieghbor i1 is a f-point and strongly influences i,
             * distribute a_{i,i1} to c-points that strongly infuence i.
             * Note: currently no distribution to the diagonal in this case.
             *--------------------------------------------------------------*/
            
            else if (P_marker[i1] == strong_f_marker)
            {
               sum = zero;
               
               /*-----------------------------------------------------------
                * Loop over row of A for point i1 and calculate the sum
                * of the connections to c-points that strongly influence i.
                *-----------------------------------------------------------*/

               sgn = 1;
               if (A_data[A_i[i1]] < 0) sgn = -1;
               for (jj1 = A_i[i1]; jj1 < A_i[i1+1]; jj1++)
               {
                  i2 = A_j[jj1];
                  if (P_marker[i2] >= jj_begin_row && 
					(sgn*A_data[jj1]) < 0)
                  {
                     sum += A_data[jj1];
                  }
               }
               
               if (sum != 0)
               {
               distribute = A_data[jj] / sum;
               
               /*-----------------------------------------------------------
                * Loop over row of A for point i1 and do the distribution.
                *-----------------------------------------------------------*/

               for (jj1 = A_i[i1]; jj1 < A_i[i1+1]; jj1++)
               {
                  i2 = A_j[jj1];
                  if (P_marker[i2] >= jj_begin_row &&
					(sgn*A_data[jj1]) < 0)
                  {
                     P_data[P_marker[i2]] += distribute * A_data[jj1];
                  }
               }
               }
               else
               {
               if( dof_func[i] == dof_func[i1])
                  diagonal += A_data[jj];
               }
            }
   
            /*--------------------------------------------------------------
             * Case 3: nieghbor i1 weakly influences i, accumulate a_{i,i1}
             * into the diagonal. This is done only if i and i1 are of the
             * same function type.
             *--------------------------------------------------------------*/

            else
            {
               if( dof_func[i] == dof_func[i1])
                  diagonal += A_data[jj];
            }            
         }

         /*-----------------------------------------------------------------
          * Set interpolation weight by dividing by the diagonal.
          *-----------------------------------------------------------------*/

         for (jj = jj_begin_row; jj < jj_end_row; jj++)
         {
            P_data[jj] /= -diagonal;
         }
      }
   
      /*--------------------------------------------------------------------
       * Interpolation formula for i is done, update marker for strong
       * f connections for next i.
       *--------------------------------------------------------------------*/
   
      strong_f_marker--;
   }
  
   P_i[n_fine] = jj_counter;

   P = hypre_CSRMatrixCreate(n_fine, n_coarse, P_size);
   hypre_CSRMatrixData(P) = P_data; 
   hypre_CSRMatrixI(P) = P_i; 
   hypre_CSRMatrixJ(P) = P_j; 

   *P_ptr = P; 

   /*-----------------------------------------------------------------------
    *  Build and return dof_func array for coarse grid.
    *-----------------------------------------------------------------------*/
    coarse_dof_func = hypre_CTAlloc(HYPRE_Int, n_coarse);

    coarse_counter=0;

    for (i=0; i < n_fine; i++)
      if (CF_marker[i] >=0)
        {
          coarse_dof_func[coarse_counter] = dof_func[i];
          coarse_counter++;
        }

    /* return coarse_dof_func array: ---------------------------------------*/

    *coarse_dof_func_ptr = coarse_dof_func;


   /*-----------------------------------------------------------------------
    *  Free mapping vector and marker array.
    *-----------------------------------------------------------------------*/

   hypre_TFree(P_marker);   
   hypre_TFree(fine_to_coarse);   
 
   return(0);  
}            
          
      /* begin HANS added */

/*--------------------------------------------------------------------------
 * hypre_AMGBuildMultipass
 * This routine implements Stube's direct interpolation with multiple passes. 
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGBuildMultipass( hypre_CSRMatrix  *A,
                   HYPRE_Int                 *CF_marker,
                   hypre_CSRMatrix     *S,
                   HYPRE_Int                 *dof_func,
                   HYPRE_Int                 **coarse_dof_func_ptr,
                   hypre_CSRMatrix     **P_ptr )
{
   
   double          *A_data;
   HYPRE_Int             *A_i;
   HYPRE_Int             *A_j;

   HYPRE_Int             *S_i;
   HYPRE_Int             *S_j;

   hypre_CSRMatrix    *P; 

   double          *P_data;
   HYPRE_Int             *P_i;
   HYPRE_Int             *P_j;

   HYPRE_Int              P_size;
   
   HYPRE_Int             *P_marker;

   HYPRE_Int             *coarse_dof_func;

   HYPRE_Int              jj_counter;
   
   HYPRE_Int              start_indexing = 0; /* start indexing for P_data at 0 */

   HYPRE_Int              n_fine;
   HYPRE_Int              n_fine_remaining;
   HYPRE_Int              n_coarse;


   HYPRE_Int             *fine_to_coarse;

   HYPRE_Int             *assigned;
   HYPRE_Int             *assigned_new;
   HYPRE_Int             *elementsPerRow;
   HYPRE_Int             *elementsPerRowNew;
   HYPRE_Int              loopCount;
   HYPRE_Int              elem;
   double           wsum;
   double           rsum;
   double           factor;
   HYPRE_Int              jP;
   HYPRE_Int              found;
   HYPRE_Int              elemIndex;
   HYPRE_Int              jPStart;

   HYPRE_Int              coarse_counter;
   
   HYPRE_Int              i,i1,i2;
   HYPRE_Int              jj,jj1;

   
   double           diagonal;

   double           zero = 0.0;
   double           one  = 1.0;
   
   hypre_printf("\n");
   hypre_printf("Multi-pass interpolation...\n");

   /*-----------------------------------------------------------------------
    *  Access the CSR vectors for A and S. Also get size of fine grid.
    *-----------------------------------------------------------------------*/

   A_data = hypre_CSRMatrixData(A);
   A_i    = hypre_CSRMatrixI(A);
   A_j    = hypre_CSRMatrixJ(A);

   S_i    = hypre_CSRMatrixI(S);
   S_j    = hypre_CSRMatrixJ(S);

   n_fine = hypre_CSRMatrixNumRows(A);

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = 0;

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int, n_fine);

   assigned = hypre_CTAlloc(HYPRE_Int, n_fine);
   assigned_new = hypre_CTAlloc(HYPRE_Int, n_fine);
   elementsPerRow = hypre_CTAlloc(HYPRE_Int, n_fine);
   elementsPerRowNew = hypre_CTAlloc(HYPRE_Int, n_fine);

   jj_counter = start_indexing;
      
   /*-----------------------------------------------------------------------
    *  First Pass: determine the maximal size of P, and elementsPerRow[i].
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Assigned points are points for which we know an interpolation
    *  formula already, and which are thus available to interpolate from.
    *  assigned[i]=1 for C points, and 2, 3, 4, ... for F points, depending
    *  in which pass their interpolation formula is determined.
    *-----------------------------------------------------------------------*/
   
   /*-----------------------------------------------------------------------
    *  First mark all C points as 'assigned'.
    *-----------------------------------------------------------------------*/

   n_fine_remaining = n_fine;

   for (i = 0; i < n_fine; i++) {
     if ( CF_marker[i] == 1 ) {
       assigned[i] = 1; /* assigned point, C point */
       elementsPerRow[i] = 1; /* one element in row i of P */
       jj_counter++; /* one more element in P */
       fine_to_coarse[i] = coarse_counter; /* this C point is assigned index
					      coarse_counter on coarse grid,
					      and in column of P */
       coarse_counter++; /* one more coarse grid point */
       n_fine_remaining--; /* one less unassigned point remaining*/
       // hypre_printf("coarse point %d elementsPerRow %d\n",i,elementsPerRow[i]);
     }
   }

   /*-----------------------------------------------------------------------
    *  While not all fine grid points are assigned, loop over fine grid.
    *-----------------------------------------------------------------------*/
    
   loopCount=1;

   while( n_fine_remaining > 0 ) {

     loopCount++;
     if(loopCount > 10){
       hypre_printf("too many multipass loops, first pass...\n");
       /* end();*/
       return(-1);
     }

     hypre_printf("*** PASS %d: %d fine points to be treated out of total %d\n",loopCount-1,n_fine_remaining,n_fine);

     /* see which points can be assigned, and determine how many elements in row */
     for (i = 0; i < n_fine; i++)  {

       /* only if i is not yet assigned, do something */

       if ( assigned[i] == 0 ) {

	 /* loop over points i1 that strongly influence i;
	    if i1 is assigned, then assign point i, and
            add number of C points that
            i1 interpolates from to elementsPerRow */
         for (jj = S_i[i]; jj < S_i[i+1]; jj++) {
	   i1 = S_j[jj];           
	   if (assigned[i1] != 0) {
	     assigned_new[i] = loopCount ; /* assign F point */
	     elementsPerRow[i]=elementsPerRow[i]+elementsPerRow[i1];
	   }
	 }

	 /* in the special case where we have an F point that is not strongly influenced
            by any other point, we assign zero interpolation to this F point */
         /* this case has to be treated separately because otherwise this point would never
            get interpolation defined, resulting in an infinite loop */
         if(S_i[i]==S_i[i+1]){
	   assigned_new[i] = loopCount ;
	   elementsPerRow[i]= zero;	   
	 }

         //hypre_printf("point %d assigned_new %d elementsPerRow %d marker %d %d %d
         //    \n",i,assigned_new[i],elementsPerRow[i],CF_marker[i],S_i[i],S_i[i+1]);
       }
     }

     /* for (i = 0; i < n_fine; i++)  {
       hypre_printf("%d %d ",i,CF_marker[i]);
       for(jj=S_i[i]; jj<S_i[i+1];jj++){
	 hypre_printf("%d ",S_j[jj]);	 
       }
       hypre_printf("\n");
       } */
   
     /* assign the points that have been determined for interpolation in this loop */
     for (i = 0; i < n_fine; i++)  {
       if ( assigned_new[i] == loopCount ){
	 assigned[i] = loopCount ; /* assigned F point */
	 n_fine_remaining--; /* one less unassigned point remaining*/
	 //	 hypre_printf("F point %d elementsPerRow %d\n",i,elementsPerRow[i]);
	 jj_counter=jj_counter+elementsPerRow[i]; /* more elements in P */
       }
     }
   }

   // hypre_printf("jj_counter %d \n",jj_counter);

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   n_coarse = coarse_counter;

   P_size = jj_counter;

   P_i    = hypre_CTAlloc(HYPRE_Int, n_fine+1);
   P_j    = hypre_CTAlloc(HYPRE_Int, P_size);
   P_data = hypre_CTAlloc(double, P_size);

   /*-----------------------------------------------------------------------
    *  Second Pass: Define interpolation and fill in P_data, P_i, and P_j.
    *-----------------------------------------------------------------------*/

   /* allocate the marker array */
   /* for a given fine point i, P_marker[j]=i if j strongly influences i */
   P_marker = hypre_CTAlloc(HYPRE_Int, n_fine);
   for (i = 0; i < n_fine; i++)  {
     P_marker[i]=-1;
   }

   /*-----------------------------------------------------------------------
    *  Build P_i and initialize P_j.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   for (i = 0; i < n_fine; i++)  {
     P_i[i] = jj_counter;
     jj_counter = jj_counter + elementsPerRow[i];
   }
   P_i[n_fine] = jj_counter;

   for (i = 0; i < P_size; i++)  {
     P_j[i] = -1 ; 
     /* -1 means not used (yet) */
     /* some of the P_j will never be used when the C points
        interpolated from overlap in a given stencil; in this
        case we have overestimated the size of P, and P will
        be compressed later (the -1 will indicate where to
        compress) */
   }

   /*-----------------------------------------------------------------------
    *  First treat all the C points.
    *-----------------------------------------------------------------------*/

   n_fine_remaining = n_fine;

   for (i = 0; i < n_fine; i++) {
     if ( CF_marker[i] == 1 ) {
       jj_counter = P_i[i];   
       P_j[jj_counter]    = fine_to_coarse[i];
       P_data[jj_counter] = one;
       elementsPerRowNew[i] = one;
       n_fine_remaining--; /* one less untreated point remaining*/
     }
   }

   /*-----------------------------------------------------------------------
    *  While not all fine grid points are treated, loop over fine grid.
    *-----------------------------------------------------------------------*/
    
   loopCount=1;

   while( n_fine_remaining > 0 ) {

     loopCount++;
     if(loopCount > 10){
       hypre_printf("too many multipass loops, second pass...\n");
       /* end(); */
       return(-1);
       
     }

     hypre_printf("+++ PASS %d: %d fine points to be treated out of total %d\n",loopCount-1,n_fine_remaining,n_fine);

     for (i = 0; i < n_fine; i++)  {

       /* only if i was assigned in this loop, build interpolation for i */
       
       if ( assigned[i] == loopCount ) {

         /* set P_marker[j] for i */
         /* P_marker[i1]=i for all the points i1 that strongly influence i */
         for (jj = S_i[i]; jj < S_i[i+1]; jj++) {
	   i1 = S_j[jj];           
           P_marker[i1]=i;
	 }         

	 jj_counter = P_i[i];   

	 /* loop over points i1 that influence i;
	    if i1 was assigned in any of the previous loops and i1 strongly 
            influences i,
            then use i1 in interpolation; loop over the coarse points i2 that
            i1 interpolates from;
            find out if a column exists yet for i2, and which index this column
            has in P; if not, make a new column in P; 
            add the weight to that column */

	 /* loop over the points i1 that influence i (columns of row i in A) */
         for (jj = A_i[i]; jj < A_i[i+1]; jj++) {
	   i1 = A_j[jj];           

	   /* check if i1 stronlgy influences i and an interpolation 
              formula exists already for i1 */
	   if ( P_marker[i1]==i && assigned[i1] < loopCount ) {

	     
	     /* loop over the coarse grid points i2 that i1 interpolates from */
	     for (jj1 = P_i[i1]; jj1 < P_i[i1]+elementsPerRowNew[i1]; jj1++) {
	       i2 = P_j[jj1];
               /* hypre_printf("point %d coarse grid column i2 %d\n",i,i2); */

               /* first check if a column exists yet for this C point */
               found = 0 ;
	       for (elem = 0; elem < elementsPerRowNew[i]; elem++){
		 if(i2 == P_j[jj_counter+elem]){
		   found = 1; /* OK, we have found the column; its index is elem */
                   elemIndex=elem;
		 }
	       }

               /* if not found, make a new column */
               if(found == 0){
                 elemIndex=elementsPerRowNew[i];
                 P_j[jj_counter+elemIndex]=i2; /* i2 is already in coarse grid numbering! */
                 P_data[jj_counter+elemIndex]=zero;
		 elementsPerRowNew[i]++;
	       }

	       /* then add the weight to the column */
               /* the weight is -a(i,i1)*P(i1,i2)   */
               /*               -A_data[jj]*P_data[jj1]          */
	       P_data[jj_counter+elemIndex]=P_data[jj_counter+elemIndex] - A_data[jj] * P_data[jj1];
	     }
	   }
	 }


	 /* now calculate the normalization factors and renormalize */

         /* first calculate the sum of all the weights in the ith row of P */

         wsum=0.;
	 for(jj1=P_i[i];jj1<P_i[i]+elementsPerRowNew[i];jj1++){
	   wsum=wsum+P_data[jj1];
	 }

	 /* then calculate the negative sum of all the elements in the ith row 
            of A (not A(i,i)) */

	 rsum=0.;
         for (jj = A_i[i]+1; jj < A_i[i+1]; jj++) {
	   rsum=rsum-A_data[jj];
	 }

         diagonal = A_data[A_i[i]];
	 
         /* calculate the normalization factor */
	 factor = rsum / diagonal / wsum;
	 // hypre_printf("point %d factor %e\n",i,factor);
	 
         /* normalize the ith row of P */

	 for(jj1=P_i[i];jj1<P_i[i]+elementsPerRowNew[i];jj1++){
           /* hypre_printf("---point %d P_j %d\n",i,P_j[jj1]);*/
	   P_data[jj1] = factor * P_data[jj1] ;
	 } 

	 n_fine_remaining--; /* one less untreated point remaining*/

	 //         hypre_printf("point %d CF %d elemPerRow before %d after %d\n",i,CF_marker[i],
         //            elementsPerRow[i],elementsPerRowNew[i]);
       } /* end building interpolation for i */

     } /* end loop over fine grid points */
   
   } /* end while untreated points remaining */

   // hypre_printf("jj_counter %d \n",jj_counter);

   /*-----------------------------------------------------------------------
    *  Third Pass: Compress P.
    *-----------------------------------------------------------------------*/

   jP = 0;
   for (i = 0; i < n_fine; i++)   {
     jPStart = jP;
     for (jj = P_i[i]; jj < P_i[i+1]; jj++)      {
       if (P_j[jj] != -1)         {
	 P_j[jP]    = P_j[jj];
	 P_data[jP] = P_data[jj];
	 jP++;
       }
       /* compression going on! */
       /* else {
	        hypre_printf("compression: point %d type %d old elementsPerRow %d new %d\n",i,
         	CF_marker[i],elementsPerRow[i],elementsPerRowNew[i]);
		}*/
     }
     P_i[i] = jPStart;
   }
   P_i[n_fine] = jP;

   P = hypre_CSRMatrixCreate(n_fine, n_coarse, P_size);
   hypre_CSRMatrixData(P) = P_data; 
   hypre_CSRMatrixI(P) = P_i; 
   hypre_CSRMatrixJ(P) = P_j; 

   *P_ptr = P; 

   hypre_CSRMatrixNumNonzeros(P) = jP;

   hypre_printf("!!!!!!! compression factor %e \n",(100.0 *((double)(P_size-jP))/(double)P_size));

   /*-----------------------------------------------------------------------
    *  Build and return dof_func array for coarse grid.
    *-----------------------------------------------------------------------*/
    coarse_dof_func = hypre_CTAlloc(HYPRE_Int, n_coarse);

    coarse_counter=0;

    for (i=0; i < n_fine; i++)
      if (CF_marker[i] >=0)
        {
          coarse_dof_func[coarse_counter] = dof_func[i];
          coarse_counter++;
        }

    /* return coarse_dof_func array: ---------------------------------------*/

    *coarse_dof_func_ptr = coarse_dof_func;


   /*-----------------------------------------------------------------------
    *  Free mapping vector and marker array.
    *-----------------------------------------------------------------------*/

   hypre_TFree(P_marker);   
   hypre_TFree(fine_to_coarse);   

   hypre_TFree(assigned);
   hypre_TFree(assigned_new);
   hypre_TFree(elementsPerRow);
   hypre_TFree(elementsPerRowNew);
 
   return(0);  
}            

/*--------------------------------------------------------------------------
 * hypre_AMGJacobiIterate
 *
 * This routine performs Stube's Jacobi iteration on a given interpolation
 * matrix P, and returns the new matrix P. 
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGJacobiIterate( hypre_CSRMatrix  *A,
                   HYPRE_Int                 *CF_marker,
                   hypre_CSRMatrix     *S,
                   HYPRE_Int                 *dof_func,
                   HYPRE_Int                 **coarse_dof_func_ptr,
                   hypre_CSRMatrix     **P_ptr )
{
   
   double          *A_data;
   HYPRE_Int             *A_i;
   HYPRE_Int             *A_j;

   HYPRE_Int             *S_i;
   HYPRE_Int             *S_j;

   hypre_CSRMatrix *PJac; 

   double          *P_data;
   HYPRE_Int             *P_i;
   HYPRE_Int             *P_j;

   double          *PJac_data;
   HYPRE_Int             *PJac_i;
   HYPRE_Int             *PJac_j;

   HYPRE_Int              PJac_size;
   
   HYPRE_Int             *P_marker;

   HYPRE_Int             *coarse_dof_func;

   HYPRE_Int              jj_counter;
   
   HYPRE_Int              start_indexing = 0; /* start indexing for P_data at 0 */

   HYPRE_Int              n_fine;
   HYPRE_Int              n_fine_remaining;
   HYPRE_Int              n_coarse;



   HYPRE_Int             *fine_to_coarse;

   HYPRE_Int             *elementsPerRow;
   HYPRE_Int             *elementsPerRowNew;

   HYPRE_Int              elem;
   double           wsum;
   double           rsum;
   double           factor;
   HYPRE_Int              jP;
   HYPRE_Int              found;
   HYPRE_Int              elemIndex;
   HYPRE_Int              jPStart;
   double           eps = 0.00000001;

   HYPRE_Int              coarse_counter;
   
   HYPRE_Int              i,i1,i2;
   HYPRE_Int              jj,jj1;

   
   double           diagonal;
   
   double           zero = 0.0;
   double           one  = 1.0;
   
   hypre_printf("\n");
   hypre_printf("Jacobi iteration...\n");

   /*-----------------------------------------------------------------------
    *  Access the CSR vectors for A, S and P. Also get size of fine grid.
    *-----------------------------------------------------------------------*/

   A_data = hypre_CSRMatrixData(A);
   A_i    = hypre_CSRMatrixI(A);
   A_j    = hypre_CSRMatrixJ(A);

   S_i    = hypre_CSRMatrixI(S);
   S_j    = hypre_CSRMatrixJ(S);

   n_fine = hypre_CSRMatrixNumRows(A);

   P_data = hypre_CSRMatrixData(*P_ptr); 
   P_i = hypre_CSRMatrixI(*P_ptr); 
   P_j = hypre_CSRMatrixJ(*P_ptr); 
   
   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = 0;

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int, n_fine);

   elementsPerRow = hypre_CTAlloc(HYPRE_Int, n_fine);
   elementsPerRowNew = hypre_CTAlloc(HYPRE_Int, n_fine);

   jj_counter = start_indexing;
      
   /*-----------------------------------------------------------------------
    *  First Pass: determine the maximal size of P, and elementsPerRow[i].
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    * First treat the C points.
    *-----------------------------------------------------------------------*/

   n_fine_remaining = n_fine;

   for (i = 0; i < n_fine; i++) {
     if ( CF_marker[i] == 1 ) {
       elementsPerRow[i] = 1; /* one element in row i of P */
       jj_counter++; /* one more element in P */
       fine_to_coarse[i] = coarse_counter; /* this C point is assigned index
					      coarse_counter on coarse grid,
					      and in column of P */
       coarse_counter++; /* one more coarse grid point */
       n_fine_remaining--; /* one less unassigned point remaining*/
       // hypre_printf("coarse point %d elementsPerRow %d\n",i,elementsPerRow[i]);
     }
   }

   /*-----------------------------------------------------------------------
    *  Now treat the F points.
    *-----------------------------------------------------------------------*/
    
   hypre_printf("*** Jacobi: %d fine points to be treated out of total %d\n",n_fine_remaining,n_fine);

   /* see which points can be assigned, and determine how many elements in row */
   for (i = 0; i < n_fine; i++)  {

     /* only if i is an F point, do something */

     if ( CF_marker[i] != 1 ) {

       /* loop over points i1 that strongly influence i;
	  add number of C points that
	  i1 interpolates from to elementsPerRow */
       for (jj = S_i[i]; jj < S_i[i+1]; jj++) {
	 i1 = S_j[jj];           
	 elementsPerRow[i]=elementsPerRow[i]+P_i[i1+1]-P_i[i1];
	 /*	 hypre_printf("node %d: interpolates from %d C points\n",i1,P_i[i1+1]-P_i[i1]);*/
       }
       //	 hypre_printf("node %d: interpolates from %d C points\n",i1,elementsPerRow[i]);

       /* in the special case where we have an F point that is not strongly influenced
	  by any other point, we assign zero interpolation to this F point */
       /* this case has to be treated separately because otherwise this point would never
	  get interpolation defined, resulting in an infinite loop */
       if(S_i[i]==S_i[i+1]){
	 elementsPerRow[i]= zero;	   
       }

       n_fine_remaining--; /* one less point remaining*/
       //	 hypre_printf("F point %d elementsPerRow %d\n",i,elementsPerRow[i]);
       jj_counter=jj_counter+elementsPerRow[i]; /* more elements in P */

     }
   }

   //   hypre_printf("jj_counter %d \n",jj_counter);

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   n_coarse = coarse_counter;

   PJac_size = jj_counter;

   PJac_i    = hypre_CTAlloc(HYPRE_Int, n_fine+1);
   PJac_j    = hypre_CTAlloc(HYPRE_Int, PJac_size);
   PJac_data = hypre_CTAlloc(double, PJac_size);

   /*-----------------------------------------------------------------------
    *  Second Pass: Define interpolation and fill in PJac_data, PJac_i, and PJac_j.
    *-----------------------------------------------------------------------*/

   /* allocate the marker array */
   /* for a given fine point i, P_marker[j]=i if j strongly influences i */
   P_marker = hypre_CTAlloc(HYPRE_Int, n_fine);
   for (i = 0; i < n_fine; i++)  {
     P_marker[i]=-1;
   }

   /*-----------------------------------------------------------------------
    *  Build PJac_i and initialize PJac_j.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   for (i = 0; i < n_fine; i++)  {
     PJac_i[i] = jj_counter;
     jj_counter = jj_counter + elementsPerRow[i];
   }
   PJac_i[n_fine] = jj_counter;

   for (i = 0; i < PJac_size; i++)  {
     PJac_j[i] = -1 ; 
     /* -1 means not used (yet) */
     /* some of the PJac_j will never be used when the C points
        interpolated from overlap in a given stencil; in this
        case we have overestimated the size of PJac, and PJac will
        be compressed later (the -1 will indicate where to
        compress) */
   }

   /*-----------------------------------------------------------------------
    *  First treat all the C points.
    *-----------------------------------------------------------------------*/

   n_fine_remaining = n_fine;

   for (i = 0; i < n_fine; i++) {
     if ( CF_marker[i] == 1 ) {
       jj_counter = PJac_i[i];   
       PJac_j[jj_counter]    = fine_to_coarse[i];
       PJac_data[jj_counter] = one;
       elementsPerRowNew[i] = one;
       n_fine_remaining--; /* one less untreated point remaining*/
     }
   }

   /*-----------------------------------------------------------------------
    *  Now treat the F points.
    *-----------------------------------------------------------------------*/
    
   hypre_printf("+++ Jacobi iteration: %d fine points to be treated out of total %d\n",n_fine_remaining,n_fine);
   
   for (i = 0; i < n_fine; i++)  {
       
     if ( CF_marker[i] != 1 ) {

       /* set P_marker[j] for i */
       /* P_marker[i1]=i for all the points i1 that strongly influence i */
       for (jj = S_i[i]; jj < S_i[i+1]; jj++) {
	 i1 = S_j[jj];           
	 P_marker[i1]=i;
	 //         hypre_printf("S: i %d i1 %d\n",i,i1);
       }         
       
       jj_counter = PJac_i[i];   
       
       /* loop over points i1 that influence i;
	  if i1 strongly influences i,
	  then use i1 in interpolation; loop over the coarse points i2 that
	  i1 interpolates from;
	  find out if a column exists yet for i2, and which index this column
	  has in PJac; if not, make a new column in PJac; 
	  add the weight to that column */
       
       /* loop over the points i1 that influence i (columns of row i in A) */
       for (jj = A_i[i]; jj < A_i[i+1]; jj++) {
	 i1 = A_j[jj];           
	 
	 /* check if i1 stronlgy influences i */
	 if ( P_marker[i1]==i ) {
           // hypre_printf("influence: i %d i1 %d marker %d\n",i,i1,P_marker[i1]);
	   
	   /* loop over the coarse grid points i2 that i1 interpolates from in P */
	   for (jj1 = P_i[i1]; jj1 < P_i[i1+1]; jj1++) {
	     i2 = P_j[jj1];
	     // hypre_printf("point %d coarse grid column i2 %d\n",i,i2);
	     
	     /* first check if a column exists yet for this C point */
	     found = 0 ;
	     for (elem = 0; elem < elementsPerRowNew[i]; elem++){
	       if(i2 == PJac_j[jj_counter+elem]){
		 found = 1; /* OK, we have found the column; its index is elem */
		 elemIndex=elem;
	       }
	     }
	     
	     /* if not found, make a new column */
	     if(found == 0){
	       elemIndex=elementsPerRowNew[i];
	       PJac_j[jj_counter+elemIndex]=i2; /* i2 is already in coarse grid numbering! */
	       PJac_data[jj_counter+elemIndex]=zero;
	       elementsPerRowNew[i]++;
	     }
	     
	     /* then add the weight to the column */
	     /* the weight is -a(i,i1)*P(i1,i2)   */
	     /*               -A_data[jj]*P_data[jj1]          */
	     PJac_data[jj_counter+elemIndex]=PJac_data[jj_counter+elemIndex] - A_data[jj] * P_data[jj1];
	     //hypre_printf("i %d i1 %d i2 %d A(i,i1) %e P(i1,i2) %e\n",i,i1,i2,A_data[jj],P_data[jj1]);
	   }
	 }
       }

       /* now calculate the normalization factors and renormalize */

       /* first calculate the sum of all the weights in the ith row of PJac */
       
       wsum=0.;
       for(jj1=PJac_i[i];jj1<PJac_i[i]+elementsPerRowNew[i];jj1++){
	 wsum=wsum+PJac_data[jj1];
         //hypre_printf("i %d data %e\n",i,PJac_data[jj1]);
       }
       
       /* then calculate the negative sum of all the elements in the ith row 
	  of A (not A(i,i)) */
       
       rsum=0.;
       for (jj = A_i[i]+1; jj < A_i[i+1]; jj++) {
	 rsum=rsum-A_data[jj];
       }
       
       diagonal = A_data[A_i[i]];
       
       if ( diagonal < 0. ) {
	 hypre_printf("@@@@@@ WARNING: negative diagonal\n");
       }
       else {
	 if ( diagonal < eps ) hypre_printf("@@@@@@ WARNING: small diagonal\n");
       }

       if ( wsum < 0. ) {
	 hypre_printf("@@@@@@ WARNING: negative wsum\n");
       }
       else {
	 if ( wsum < eps ) hypre_printf("@@@@@@ WARNING: small wsum\n");
       }

       /* calculate the normalization factor */
       factor = rsum / diagonal / wsum;
       //hypre_printf("point %d factor %e\n",i,factor);
       
       /* normalize the ith row of PJac */
       
       for(jj1=PJac_i[i];jj1<PJac_i[i]+elementsPerRowNew[i];jj1++){
	 /* hypre_printf("---point %d P_j %d\n",i,P_j[jj1]);*/
	 PJac_data[jj1] = factor * PJac_data[jj1] ;
       } 
       
       n_fine_remaining--; /* one less untreated point remaining*/
       
       //                hypre_printf("point %d CF %d elemPerRow before %d after %d\n",i,CF_marker[i],
       //          elementsPerRow[i],elementsPerRowNew[i]);
     } /* end building interpolation for i */
     
   } /* end loop over fine grid points */
   
   // hypre_printf("jj_counter %d \n",jj_counter);

   /*-----------------------------------------------------------------------
    *  Third Pass: Compress PJac.
    *-----------------------------------------------------------------------*/

   jP = 0;
   for (i = 0; i < n_fine; i++)   {
     jPStart = jP;
     for (jj = PJac_i[i]; jj < PJac_i[i+1]; jj++)      {
       if (PJac_j[jj] != -1)         {
	 PJac_j[jP]    = PJac_j[jj];
	 PJac_data[jP] = PJac_data[jj];
	 jP++;
       }
       /* compression going on! */
       /* else {
	 hypre_printf("compression: point %d type %d old elementsPerRow %d new %d\n",i,
		CF_marker[i],elementsPerRow[i],elementsPerRowNew[i]);
		} */
     }
     PJac_i[i] = jPStart;
   }
   PJac_i[n_fine] = jP;

   //hypre_printf("PJac_size %d\n",PJac_size);

   PJac = hypre_CSRMatrixCreate(n_fine, n_coarse, PJac_size);
   hypre_CSRMatrixData(PJac) = PJac_data; 
   hypre_CSRMatrixI(PJac) = PJac_i; 
   hypre_CSRMatrixJ(PJac) = PJac_j; 

   *P_ptr = PJac; 

   hypre_CSRMatrixNumNonzeros(PJac) = jP;

   hypre_printf("!!!!!!! compression factor %e \n",100.*((double)(PJac_size-jP))/(double)PJac_size);

   /*-----------------------------------------------------------------------
    *  Build and return dof_func array for coarse grid.
    *-----------------------------------------------------------------------*/
    coarse_dof_func = hypre_CTAlloc(HYPRE_Int, n_coarse);

    coarse_counter=0;

    for (i=0; i < n_fine; i++)
      if (CF_marker[i] >=0)
        {
          coarse_dof_func[coarse_counter] = dof_func[i];
          coarse_counter++;
        }

    /* return coarse_dof_func array: ---------------------------------------*/

    *coarse_dof_func_ptr = coarse_dof_func;


   /*-----------------------------------------------------------------------
    *  Free mapping vector and marker array.
    *-----------------------------------------------------------------------*/

   hypre_TFree(P_marker);   
   hypre_TFree(fine_to_coarse);   

   hypre_TFree(elementsPerRow);
   hypre_TFree(elementsPerRowNew);

   hypre_TFree(P_data);
   hypre_TFree(P_i);
   hypre_TFree(P_j);

   return(0);  
}            
          
      /* end HANS added */






