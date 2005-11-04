/*---------------------------------------------------- 
 * Communication package that uses an assumed partition
 *  AHB 6/04                                            
 *-----------------------------------------------------*/

#include "headers.h"

/* some debugging tools*/
#define mydebug 0

int hypre_LocateAssummedPartition(int, int, int, hypre_IJAssumedPart*, int);
int hypre_GetAssumedPartitionProcFromRow( int, int, int* );
int hypre_GetAssumedPartitionRowRange( int, int, int*, int* );
int hypre_FillResponseIJDetermineSendProcs(void*, int, int, void*, MPI_Comm, void**, int*);
int hypre_RangeFillResponseIJDetermineRecvProcs(void*, int, int, void*, MPI_Comm,void**, int*);

#define CONTACT(a,b)  (contact_list[(a)*3+(b)])

/*==========================================================================*/




/*------------------------------------------------------------------
 * hypre_NewCommPkgCreate_core
 *
 * This does the work for  hypre_NewCommPkgCreate - we have to split it 
 * off so that it can also be used for block matrices.
 *--------------------------------------------------------------------------*/

int 
hypre_NewCommPkgCreate_core(
/* input args: */
   MPI_Comm comm, int *col_map_off_d, int first_col_diag,
   int col_start, int col_end, 
   int num_cols_off_d, int global_num_cols,
/* pointers to output args: */
   int *p_num_recvs, int **p_recv_procs, int **p_recv_vec_starts,
   int *p_num_sends, int **p_send_procs, int ** p_send_map_starts,
   int **p_send_map_elements)

{
   int        num_procs, myid;
   int        j, i, ierr=0;
   int        range_start, range_end; 

   int        size;
   int        count;  

   int        num_recvs, *recv_procs = NULL, *recv_vec_starts=NULL;
   int        tmp_id, prev_id;

   int        num_sends;

   int        ex_num_contacts, *ex_contact_procs=NULL, *ex_contact_vec_starts=NULL;
   int        *ex_contact_buf=NULL;
    
   int        num_ranges, upper_bound;
   

   int        *response_buf = NULL, *response_buf_starts=NULL;

   int        max_response_size;
   
   hypre_IJAssumedPart               apart;
   hypre_DataExchangeResponse        response_obj1, response_obj2;
   hypre_ProcListElements            send_proc_obj; 

#if mydebug
   int tmp_int, index;
#endif

   MPI_Comm_size(comm, &num_procs );
   MPI_Comm_rank(comm, &myid );


   /*-----------------------------------------------------------
    * Assumed Partitioning - each proc gets it own range. Then 
    * each needs to reconcile its actual range with its assumed
    * range.   
    *-----------------------------------------------------------*/
  
  
   /* get my assumed partitioning  - we want partitioning of the vector that the
      matrix multiplies - so we use the col start and end */

   ierr = hypre_GetAssumedPartitionRowRange( myid, global_num_cols, &apart.row_start, 
                                        &apart.row_end);

#if mydebug
    printf("myid = %i, my assumed local range: [%i, %i]\n", myid, 
                         apart.row_start, apart.row_end);
#endif


    /*allocate some space for the partition of the assumed partition */
    apart.length = 0;
    /*room for 10 owners of the assumed partition*/ 
    apart.storage_length = 10; /*need to be >=1 */ 
    apart.proc_list = hypre_TAlloc(int, apart.storage_length);
    apart.row_start_list =   hypre_TAlloc(int, apart.storage_length);
    apart.row_end_list =   hypre_TAlloc(int, apart.storage_length);

    hypre_LocateAssummedPartition(col_start, col_end, global_num_cols, &apart, myid);

#if mydebug
      for (i=0; i<apart.length; i++)
      {
        printf("myid = %d, proc %d owns assumed partition range = [%d, %d]\n", 
                myid, apart.proc_list[i], apart.row_start_list[i], 
	        apart.row_end_list[i]);
      }

      printf("myid = %d, length of apart = %d\n", myid, apart.length);

#endif



   /*-----------------------------------------------------------
    *  Now everyone knows where their assumed range is located.
    *  For the comm. package, each proc must know it's receive
    *  procs (who it will receive data from and how much data) 
    *  and its send procs 
    *  (who it will send data to) and the indices of the elements
    *  to be sent.  This is based on the non-zero
    *  entries in its rows. Each proc should know this from the user. 
    *-----------------------------------------------------------*/
   

   /*------------------------------------------------------------
    *  get the receive processors
    *  each par_csr matrix will have a certain number of columns
    *  (num_cols_off_d) given in col_map_offd[] for which it needs
    *  data from another processor. 
    *
    *------------------------------------------------------------*/

    /*calculate the assumed receive processors*/

   /* need to populate num_recvs, *recv_procs, and *recv_vec_starts 
      (correlates to starts in col_map_off_d for recv_procs) for 
      the comm. package*/


   /*create contact information*/

   ex_num_contacts = 0;

   /*estimate the storage needed*/
   if (num_cols_off_d > 0 && (apart.row_end - apart.row_start) > 0  )
   {
      size = col_map_off_d[num_cols_off_d-1] - col_map_off_d[0];
   
      size = (size/(apart.row_end - apart.row_start)) + 2;
   }
   else
   {
      size = 0;
   }
   

   /*we will contact each with a range of cols that we need*/
   /* it is ok to contact yourself - because then there doesn't
      need to be separate code */

   ex_contact_procs = hypre_CTAlloc(int, size);
   ex_contact_vec_starts =  hypre_CTAlloc(int, size+1);
   ex_contact_buf =  hypre_CTAlloc(int, size*2);

   range_end = -1;
   for (i=0; i< num_cols_off_d; i++) 
   { 
      if (col_map_off_d[i] > range_end)
      {


         hypre_GetAssumedPartitionProcFromRow(col_map_off_d[i], 
                                              global_num_cols, &tmp_id);

         if (ex_num_contacts == size) /*need more space? */ 
         {
           size += 20;
           ex_contact_procs = hypre_TReAlloc(ex_contact_procs, int, size);
           ex_contact_vec_starts = hypre_TReAlloc(ex_contact_vec_starts, int, size+1);
           ex_contact_buf = hypre_TReAlloc(ex_contact_buf, int, size*2);
         }

         /* end of prev. range */
         if (ex_num_contacts > 0)  ex_contact_buf[ex_num_contacts*2 - 1] = col_map_off_d[i-1];
         
        /*start new range*/
    	 ex_contact_procs[ex_num_contacts] = tmp_id;
         ex_contact_vec_starts[ex_num_contacts] = ex_num_contacts*2;
         ex_contact_buf[ex_num_contacts*2] =  col_map_off_d[i];
         
         
         ex_num_contacts++;

         hypre_GetAssumedPartitionRowRange(tmp_id, global_num_cols, 
                                           &range_start, &range_end); 

      }
   }

   /*finish the starts*/
   ex_contact_vec_starts[ex_num_contacts] =  ex_num_contacts*2;
   /*finish the last range*/
   if (ex_num_contacts > 0)  ex_contact_buf[ex_num_contacts*2 - 1] = col_map_off_d[num_cols_off_d-1];


   /*don't allocate space for responses */
    

   /*create response object*/
   response_obj1.fill_response = hypre_RangeFillResponseIJDetermineRecvProcs;
   response_obj1.data1 =  &apart; /* this is necessary so we can fill responses*/ 
   response_obj1.data2 = NULL;
   
   max_response_size = 6;  /* 6 means we can fit 3 ranges*/
   
   
   hypre_DataExchangeList(ex_num_contacts, ex_contact_procs, 
                    ex_contact_buf, ex_contact_vec_starts, sizeof(int), 
                     sizeof(int), &response_obj1, max_response_size, 1, 
                     comm, (void**) &response_buf, &response_buf_starts);



   /*now create recv_procs[] and recv_vec_starts[] and num_recvs 
     from the complete data in response_buf - this array contains
     a proc_id followed by an upper bound for the range.  */


   /*initialize */ 
   num_recvs = 0;
   size  = ex_num_contacts+20; /* num of recv procs should be roughly similar size 
                                 to number of contacts  - add a buffer of 20*/
 
   
   recv_procs = hypre_CTAlloc(int, size);
   recv_vec_starts =  hypre_CTAlloc(int, size+1);
   recv_vec_starts[0] = 0;
   
   /*how many ranges were returned?*/
   num_ranges = response_buf_starts[ex_num_contacts];   
   num_ranges = num_ranges/2;
   
   prev_id = -1;
   j = 0;
   count = 0;
   
   /* loop through ranges */
   for (i=0; i<num_ranges; i++)
   {
      upper_bound = response_buf[i*2+1];
      count = 0;
      /* loop through off_d entries - counting how many are in the range */
      while (col_map_off_d[j] <= upper_bound && j < num_cols_off_d)     
      {
         j++;
         count++;       
      }
      if (count > 0)        
      {
         /*add the range if the proc id != myid*/    
         tmp_id = response_buf[i*2];
         if (tmp_id != myid)
         {
            if (tmp_id != prev_id) /*increment the number of recvs */
            {
               /*check size of recv buffers*/
               if (num_recvs == size) 
               {
                  size+=20;
                  recv_procs = hypre_TReAlloc(recv_procs,int, size);
                  recv_vec_starts =  hypre_TReAlloc(recv_vec_starts,int, size+1);
               }
            
               recv_vec_starts[num_recvs+1] = j; /*the new start is at this element*/
               recv_procs[num_recvs] =  tmp_id; /*add the new processor*/
               num_recvs++;

            }
            else
            {
               /*same processor - just change the vec starts*/
               recv_vec_starts[num_recvs] = j; /*the new start is at this element*/
            }
         }
         prev_id = tmp_id;
         
      }
      
   }
 



#if mydebug
      for (i=0; i < num_recvs; i++) 
      {
          printf("myid = %d, recv proc = %d, vec_starts = [%d : %d]\n", 
                  myid, recv_procs[i], recv_vec_starts[i],recv_vec_starts[i+1]-1);
      }
#endif
 

   /*------------------------------------------------------------
    *  determine the send processors
    *  each processor contacts its recv procs to let them
    *  know they are a send processor
    *
    *-------------------------------------------------------------*/

   /* the contact information is the recv_processor infomation - so
      nothing more to do to generate contact info*/

   /* the response we expect is just a confirmation*/
   hypre_TFree(response_buf);
   hypre_TFree(response_buf_starts);
   response_buf = NULL;
   response_buf_starts = NULL;

   /*build the response object*/
   /*estimate for inital storage allocation that we send to as many procs 
     as we recv from + pad by 5*/
   send_proc_obj.length = 0;
   send_proc_obj.storage_length = num_recvs + 5;
   send_proc_obj.id = hypre_CTAlloc(int, send_proc_obj.storage_length);
   send_proc_obj.vec_starts = hypre_CTAlloc(int, send_proc_obj.storage_length + 1); 
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = num_cols_off_d;
   send_proc_obj.elements = hypre_CTAlloc(int, send_proc_obj.element_storage_length);

   response_obj2.fill_response = hypre_FillResponseIJDetermineSendProcs;
   response_obj2.data1 = NULL;
   response_obj2.data2 = &send_proc_obj; /*this is where we keep info from contacts*/
  
   max_response_size = 0;
      


   hypre_DataExchangeList(num_recvs, recv_procs, 
                     col_map_off_d, recv_vec_starts, sizeof(int),
                    sizeof(int), &response_obj2, max_response_size, 2, 
                    comm,  (void **) &response_buf, &response_buf_starts);



   num_sends = send_proc_obj.length; 

   /*send procs are in send_proc_object.id */
   /*send proc starts are in send_proc_obj.vec_starts */

#if mydebug
      printf("myid = %d, num_sends = %d\n", myid, num_sends);   
      for (i=0; i < num_sends; i++) 
      {
        tmp_int = send_proc_obj.vec_starts[i+1] - send_proc_obj.vec_starts[i];
        index = send_proc_obj.vec_starts[i];
        for (j=0; j< tmp_int; j++) 
	{
	  printf("myid = %d, send proc = %d, send element = %d\n",myid,  
                  send_proc_obj.id[i],send_proc_obj.elements[index+j]); 
	 }   
      }
#endif

   /*-----------------------------------------------------------
    *  Return output info for setting up the comm package
    *-----------------------------------------------------------*/


   *p_num_recvs = num_recvs;
   *p_recv_procs = recv_procs;
   *p_recv_vec_starts = recv_vec_starts;
   *p_num_sends = num_sends;
   *p_send_procs = send_proc_obj.id;
   *p_send_map_starts = send_proc_obj.vec_starts;


   /*send map elements have global index - need local instead*/

   if (num_sends)
   {
      for (i=0; i<send_proc_obj.vec_starts[num_sends]; i++)
      {   
         send_proc_obj.elements[i] -= first_col_diag;
      }
   }
   *p_send_map_elements =  send_proc_obj.elements;



   /*-----------------------------------------------------------
    *  Clean up
    *-----------------------------------------------------------*/
 
   if(apart.storage_length > 0) 
   {      
      hypre_TFree(apart.proc_list);
      hypre_TFree(apart.row_start_list);
      hypre_TFree(apart.row_end_list);
      hypre_TFree(apart.sort_index);
   }

  
   if(ex_contact_procs)      hypre_TFree(ex_contact_procs);
   if(ex_contact_vec_starts) hypre_TFree(ex_contact_vec_starts);
   hypre_TFree(ex_contact_buf);
   

   if(response_buf)        hypre_TFree(response_buf);
   if(response_buf_starts) hypre_TFree(response_buf_starts);

   
   /* don't free send_proc_obj.id,send_proc_obj.vec_starts,send_proc_obj.elements;
      recv_procs, recv_vec_starts.  These are aliased to the comm package and
      will be destroyed there */
  

   return(ierr);

}

/*------------------------------------------------------------------
 * hypre_NewCommPkgCreate
 * this is an alternate way of constructing the comm package                                 
 * (compare to hypre_MatvecCommPkgCreate() in par_csr_communication.c
 * that should be more scalable 
 *-------------------------------------------------------------------*/

int 
hypre_NewCommPkgCreate( hypre_ParCSRMatrix *parcsr_A)
{

   int        row_start=0, row_end=0, col_start = 0, col_end = 0;
   int        num_recvs, *recv_procs, *recv_vec_starts;

   int        num_sends, *send_procs, *send_map_starts;
   int        *send_map_elements;

   int        num_cols_off_d; 
   int       *col_map_off_d; 

   int        first_col_diag;
   int        global_num_cols;

   int        ierr = 0;

   MPI_Comm   comm;

   hypre_ParCSRCommPkg	 *comm_pkg;

   
   /*-----------------------------------------------------------
    * get parcsr_A information 
    *----------------------------------------------------------*/

   ierr = hypre_ParCSRMatrixGetLocalRange( parcsr_A,
                                           &row_start, &row_end ,
                                           &col_start, &col_end );
   
   col_map_off_d =  hypre_ParCSRMatrixColMapOffd(parcsr_A);
   num_cols_off_d = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(parcsr_A));
   
   global_num_cols = hypre_ParCSRMatrixGlobalNumCols(parcsr_A); 

   comm = hypre_ParCSRMatrixComm(parcsr_A);

   first_col_diag = hypre_ParCSRMatrixFirstColDiag(parcsr_A);

   /*-----------------------------------------------------------
    * get commpkg info information 
    *----------------------------------------------------------*/

   hypre_NewCommPkgCreate_core( comm, col_map_off_d, first_col_diag, 
                                col_start, col_end, 
                                num_cols_off_d, global_num_cols,
                                &num_recvs, &recv_procs, &recv_vec_starts,
                                &num_sends, &send_procs, &send_map_starts, 
                                &send_map_elements);
   

   if (!num_recvs)
   {
      hypre_TFree(recv_procs);
      recv_procs = NULL;
   }
   if (!num_sends)
   {
      hypre_TFree(send_procs);
      hypre_TFree(send_map_elements);
      send_procs = NULL;
      send_map_elements = NULL;
   }
   

  /*-----------------------------------------------------------
   * setup commpkg
   *----------------------------------------------------------*/

   comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg, 1);

   hypre_ParCSRCommPkgComm(comm_pkg) = comm;
   hypre_ParCSRCommPkgNumRecvs(comm_pkg) = num_recvs;
   hypre_ParCSRCommPkgRecvProcs(comm_pkg) = recv_procs;
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg) = recv_vec_starts;
   hypre_ParCSRCommPkgNumSends(comm_pkg) = num_sends;
   hypre_ParCSRCommPkgSendProcs(comm_pkg) = send_procs;
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts;
   hypre_ParCSRCommPkgSendMapElmts(comm_pkg) = send_map_elements;
   
   hypre_ParCSRMatrixCommPkg(parcsr_A) = comm_pkg;


   return ierr;
   
   
}

/*------------------------------------------------------------------
 *  hypre_NewCommPkgDestroy
 *  Destroy the comm package
 *------------------------------------------------------------------*/


int
hypre_NewCommPkgDestroy(hypre_ParCSRMatrix *parcsr_A)
{


   hypre_ParCSRCommPkg	 *comm_pkg = hypre_ParCSRMatrixCommPkg(parcsr_A);
   int ierr = 0;

   /*even if num_sends and num_recvs  = 0, storage may have been allocated */

   if (hypre_ParCSRCommPkgSendProcs(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgSendProcs(comm_pkg));
   } 
   if (hypre_ParCSRCommPkgSendMapElmts(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgSendMapElmts(comm_pkg));
   }
   if (hypre_ParCSRCommPkgSendMapStarts(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgSendMapStarts(comm_pkg));
   }
   if (hypre_ParCSRCommPkgRecvProcs(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgRecvProcs(comm_pkg));
   }
   if (hypre_ParCSRCommPkgRecvVecStarts(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgRecvVecStarts(comm_pkg));
   }

   hypre_TFree(comm_pkg);
   hypre_ParCSRMatrixCommPkg(parcsr_A) = NULL;  /*this gets freed again in destroy 
                                                  parscr since there are two comm 
                                                  packages now*/  

   return ierr;
}


/*--------------------------------------------------------------------
 * hypre_LocateAssummedPartition
 * Reconcile assumed partition with actual partition.  Essentially
 * each processor ends of with a partition of its assumed partition.
 *--------------------------------------------------------------------*/


int 
hypre_LocateAssummedPartition(int row_start, int row_end, int global_num_rows, 
                        hypre_IJAssumedPart *part, int myid)
{  

   int       i, ierr;

   int       *contact_list;
   int        contact_list_length, contact_list_storage;
   
   int        contact_row_start[2], contact_row_end[2], contact_ranges;
   int        owner_start, owner_end;
   int        tmp_row_start, tmp_row_end, complete;

   int        locate_row_start[2], locate_ranges;

   int        locate_row_count, rows_found;  
 
   int        tmp_range[2];
   int       *si, *sortme;   

   const int  flag1 = 17;  

   MPI_Request  *requests;
   MPI_Status   status0, *statuses;




   /*-----------------------------------------------------------
    *  Contact ranges - 
    *  which rows do I have that others are assumed responsible for?
    *  (at most two ranges - maybe none)
    *-----------------------------------------------------------*/
  

   contact_row_start[0]=0;
   contact_row_end[0]=0;
   contact_row_start[1]=0;
   contact_row_end[1]=0;
   contact_ranges = 0;
 
   if (row_start <= row_end ) { /*must own at least one row*/

      if ( part->row_end < row_start  || row_end < part->row_start  )   
      {  /*no overlap - so all of my rows and only one range*/
         contact_row_start[0] = row_start;
         contact_row_end[0] = row_end;
         contact_ranges++;
      }
      else /* the two regions overlap - so one or two ranges */
      {  
         /* check for contact rows on the low end of the local range */
	 if (row_start < part->row_start) 
	 {
            contact_row_start[0] = row_start;
	    contact_row_end[0] = part->row_start - 1;
	    contact_ranges++;
	 } 
	 if (part->row_end < row_end) /* check the high end */
         {
            if (contact_ranges) /* already found one range */ 
            {
	       contact_row_start[1] = part->row_end +1;
	       contact_row_end[1] = row_end;
	    }
	    else
	    {
	       contact_row_start[0] =  part->row_end +1;
	       contact_row_end[0] = row_end;
	    } 
	    contact_ranges++;
	 }
      }
   }

   /*-----------------------------------------------------------
    *  Contact: find out who is assumed responsible for these 
    *       ranges of contact rows and contact them 
    *
    *-----------------------------------------------------------*/

     
   contact_list_length = 0;
   contact_list_storage = 5; 
   contact_list = hypre_TAlloc(int, contact_list_storage*3); /*each contact needs 3 ints */

   for (i=0; i<contact_ranges; i++)
   {
  
      /*get start and end row owners */
      ierr = hypre_GetAssumedPartitionProcFromRow(contact_row_start[i], global_num_rows, 
                                            &owner_start);
      ierr = hypre_GetAssumedPartitionProcFromRow(contact_row_end[i], global_num_rows, 
                                             &owner_end);

      if (owner_start == owner_end) /* same processor owns the whole range */
      {

         if (contact_list_length == contact_list_storage)
         {
            /*allocate more space*/
            contact_list_storage += 5;
            contact_list = hypre_TReAlloc(contact_list, int, (contact_list_storage*3));
         }
         CONTACT(contact_list_length, 0) = owner_start;   /*proc #*/
         CONTACT(contact_list_length, 1) = contact_row_start[i];  /* start row */
         CONTACT(contact_list_length, 2) = contact_row_end[i];  /*end row */
         contact_list_length++;
      }
      else
      { 
        complete = 0;
        while (!complete) 
        {
            hypre_GetAssumedPartitionRowRange(owner_start, global_num_rows, 
                                         &tmp_row_start, &tmp_row_end); 
           
            if (tmp_row_end >= contact_row_end[i])
	    {
	       tmp_row_end =  contact_row_end[i];
               complete = 1; 
	    }     
            if (tmp_row_start <  contact_row_start[i])
	    {
	      tmp_row_start =  contact_row_start[i];
            }


            if (contact_list_length == contact_list_storage)
            {
               /*allocate more space*/
               contact_list_storage += 5;
               contact_list = hypre_TReAlloc(contact_list, int, (contact_list_storage*3));
            }


            CONTACT(contact_list_length, 0) = owner_start;   /*proc #*/
            CONTACT(contact_list_length, 1) = tmp_row_start;  /* start row */
            CONTACT(contact_list_length, 2) = tmp_row_end;  /*end row */
            contact_list_length++;
	    owner_start++; /*processors are seqential */
        }
      }
   }

   requests = hypre_CTAlloc(MPI_Request, contact_list_length);
   statuses = hypre_CTAlloc(MPI_Status, contact_list_length);

   /*send out messages */
   for (i=0; i< contact_list_length; i++) 
   {
      MPI_Isend(&CONTACT(i,1) ,2, MPI_INT, CONTACT(i,0), flag1 , 
                 MPI_COMM_WORLD, &requests[i]);
   }

   /*-----------------------------------------------------------
    *  Locate ranges - 
    *  which rows in my assumed range do I not own
    *  (at most two ranges - maybe none)
    *  locate_row_count = total number of rows I must locate
    *-----------------------------------------------------------*/


   locate_row_count = 0;
 
   locate_row_start[0]=0;
   locate_row_start[1]=0;

   locate_ranges = 0;

   if (part->row_end < row_start  || row_end < part->row_start  ) 
   /*no overlap - so all of my assumed rows */ 
   {
      locate_row_start[0] = part->row_start;
      locate_ranges++;
      locate_row_count += part->row_end - part->row_start + 1; 
   }
   else /* the two regions overlap */
   {
      if (part->row_start < row_start) 
      {/* check for locate rows on the low end of the local range */
         locate_row_start[0] = part->row_start;
         locate_ranges++;
         locate_row_count += (row_start-1) - part->row_start + 1;
      } 
      if (row_end < part->row_end) /* check the high end */
      {
         if (locate_ranges) /* already have one range */ 
         {
	    locate_row_start[1] = row_end +1;
	 }
         else
         {
	    locate_row_start[0] = row_end +1;
         } 
         locate_ranges++;
         locate_row_count += part->row_end - (row_end + 1) + 1;
      }
   }


    /*-----------------------------------------------------------
     * Receive messages from other procs telling us where
     * all our  locate rows actually reside 
     *-----------------------------------------------------------*/


    /* we will keep a partition of our assumed partition - list ourselves 
       first.  We will sort later with an additional index.
       In practice, this should only contain a few processors */
 
   /*which part do I own?*/
   tmp_row_start = hypre_max(part->row_start, row_start);
   tmp_row_end = hypre_min(row_end, part->row_end);

   if (tmp_row_start <= tmp_row_end)
   {
      part->proc_list[0] =   myid;
      part->row_start_list[0] = tmp_row_start;
      part->row_end_list[0] = tmp_row_end;
      part->length++;
   }
  
   /* now look for messages that tell us which processor has our locate rows */
   /* these will be blocking receives as we know how many to expect and they should
       be waiting (and we don't want to continue on without them) */

   rows_found = 0;

   while (rows_found != locate_row_count) {
  
      MPI_Recv( tmp_range, 2 , MPI_INT, MPI_ANY_SOURCE, 
                flag1 , MPI_COMM_WORLD, &status0);
     
      if (part->length==part->storage_length)
      {
	part->storage_length+=10;
        part->proc_list = hypre_TReAlloc(part->proc_list, int, part->storage_length);
        part->row_start_list =   hypre_TReAlloc(part->row_start_list, int, part->storage_length);
        part->row_end_list =   hypre_TReAlloc(part->row_end_list, int, part->storage_length);

      }
      part->row_start_list[part->length] = tmp_range[0];
      part->row_end_list[part->length] = tmp_range[1]; 

      part->proc_list[part->length] = status0.MPI_SOURCE;
      rows_found += tmp_range[1]- tmp_range[0] + 1;
      
      part->length++;
   } 

   /*In case the partition of the assumed partition is longish, 
     we would like to know the sorted order */
   si= hypre_CTAlloc(int, part->length); 
   sortme = hypre_CTAlloc(int, part->length); 

   for (i=0; i<part->length; i++) 
   {
       si[i] = i;
       sortme[i] = part->row_start_list[i];
   }
   hypre_qsort2i( sortme, si, 0, (part->length)-1);
   part->sort_index = si;

  /*free the requests */
   ierr = MPI_Waitall(contact_list_length, requests, 
                    statuses);

   
   hypre_TFree(statuses);
   hypre_TFree(requests);
   

   hypre_TFree(sortme);
   hypre_TFree(contact_list);


   return(ierr);

}

/*--------------------------------------------------------------------
 * hypre_RangeFillResponseIJDetermineRecvProcs
 * Fill response function for determining the recv. processors
 * data exchange
 *--------------------------------------------------------------------*/

int
hypre_RangeFillResponseIJDetermineRecvProcs(void *p_recv_contact_buf,
                                      int contact_size, int contact_proc, void *ro, 
                                      MPI_Comm comm, void **p_send_response_buf, 
                                      int *response_message_size)
{
   int    myid, tmp_id, row_end;
   int    j;
   int    row_val, index, size;
   
   int   *send_response_buf = (int *) *p_send_response_buf;
   int   *recv_contact_buf = (int * ) p_recv_contact_buf;


   hypre_DataExchangeResponse  *response_obj = ro; 
   hypre_IJAssumedPart               *part = response_obj->data1;
   
   int overhead = response_obj->send_response_overhead;

   /*-------------------------------------------------------------------
    * we are getting a range of off_d entries - need to see if we own them
    * or how many ranges to send back  - send back
    * with format [proc_id end_row  proc_id #end_row  proc_id #end_row etc...].
    *----------------------------------------------------------------------*/ 

   MPI_Comm_rank(comm, &myid );


   /* populate send_response_buf */
      
   index = 0; /*count entries in send_response_buf*/
   
   j = 0; /*marks which partition of the assumed partition we are in */
   row_val = recv_contact_buf[0]; /*beginning of range*/
   row_end = part->row_end_list[part->sort_index[j]];
   tmp_id = part->proc_list[part->sort_index[j]];

   /*check storage in send_buf for adding the ranges */
   size = 2*(part->length);
         
   if ( response_obj->send_response_storage  < size  )
   {

      response_obj->send_response_storage =  hypre_max(size, 20); 
      send_response_buf = hypre_TReAlloc( send_response_buf, int, 
                                         response_obj->send_response_storage + overhead );
      *p_send_response_buf = send_response_buf;    /* needed when using ReAlloc */
   }


   while (row_val > row_end) /*which partition to start in */
   {
      j++;
      row_end = part->row_end_list[part->sort_index[j]];   
      tmp_id = part->proc_list[part->sort_index[j]];
   }

   /*add this range*/
   send_response_buf[index++] = tmp_id;
   send_response_buf[index++] = row_end; 

   j++; /*increase j to look in next partition */
      
    
   /*any more?  - now compare with end of range value*/
   row_val = recv_contact_buf[1]; /*end of range*/
   while ( j < part->length && row_val > row_end  )
   {
      row_end = part->row_end_list[part->sort_index[j]];  
      tmp_id = part->proc_list[part->sort_index[j]];

      send_response_buf[index++] = tmp_id;
      send_response_buf[index++] = row_end; 

      j++;
      
   }


   *response_message_size = index;
   *p_send_response_buf = send_response_buf;


   return(0);
}



/*--------------------------------------------------------------------
 * hypre_FillResponseIJDetermineSendProcs
 * Fill response function for determining the send processors
 * data exchange
 *--------------------------------------------------------------------*/

int
hypre_FillResponseIJDetermineSendProcs(void *p_recv_contact_buf, 
                                 int contact_size, int contact_proc, void *ro, 
                                 MPI_Comm comm, void **p_send_response_buf, 
                                 int *response_message_size )
{
   int    myid;
   int    i, index, count, elength;

   int    *recv_contact_buf = (int * ) p_recv_contact_buf;

   hypre_DataExchangeResponse  *response_obj = ro;  

   hypre_ProcListElements      *send_proc_obj = response_obj->data2;   


   MPI_Comm_rank(comm, &myid );


   /*check to see if we need to allocate more space in send_proc_obj for ids*/
   if (send_proc_obj->length == send_proc_obj->storage_length)
   {
      send_proc_obj->storage_length +=20; /*add space for 20 more processors*/
      send_proc_obj->id = hypre_TReAlloc(send_proc_obj->id,int, 
					 send_proc_obj->storage_length);
      send_proc_obj->vec_starts = hypre_TReAlloc(send_proc_obj->vec_starts,int, 
                                  send_proc_obj->storage_length + 1);
   }
  
   /*initialize*/ 
   count = send_proc_obj->length;
   index = send_proc_obj->vec_starts[count]; /*this is the number of elements*/

   /*send proc*/ 
   send_proc_obj->id[count] = contact_proc; 

   /*do we need more storage for the elements?*/
     if (send_proc_obj->element_storage_length < index + contact_size)
   {
      elength = hypre_max(contact_size, 50);   
      elength += index;
      send_proc_obj->elements = hypre_TReAlloc(send_proc_obj->elements, 
					       int, elength);
      send_proc_obj->element_storage_length = elength; 
   }
   /*populate send_proc_obj*/
   for (i=0; i< contact_size; i++) 
   { 
      send_proc_obj->elements[index++] = recv_contact_buf[i];
   }
   send_proc_obj->vec_starts[count+1] = index;
   send_proc_obj->length++;
   

  /*output - no message to return (confirmation) */
   *response_message_size = 0; 
  
   
   return(0);

}

/*--------------------------------------------------------------------
 * hypre_GetAssumedPartitionProcFromRow
 * Assumed partition for IJ case. Given a particular row j, return
 * the processor that is assumed to own that row.
 *--------------------------------------------------------------------*/


int
hypre_GetAssumedPartitionProcFromRow( int row, int global_num_rows, int *proc_id)
{

   int     num_procs;
   int     size, switch_row, extra;
   
  
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
 
   /* j = floor[(row*p/N]  - this overflows*/
   /* *proc_id = (row*num_procs)/global_num_rows;*/
       
   /* this looks a bit odd, but we have to be very careful that
      this function and the next are inverses - and rounding 
      errors make this difficult!!!!! */

   size = global_num_rows /num_procs;
   extra = global_num_rows - size*num_procs;
   switch_row = (size + 1)*extra;
   
   if (row >= switch_row)
   {
      *proc_id = extra + (row - switch_row)/size;
   }
   else
   {
      *proc_id = row/(size+1);      
   }



   return(0);

}
/*--------------------------------------------------------------------
 * hypre_GetAssumedPartitionRowRange
 * Assumed partition for IJ case. Given a particular processor id, return
 * the assumed range of rows ([row_start, row_end]) for that processor.
 *--------------------------------------------------------------------*/


int
hypre_GetAssumedPartitionRowRange( int proc_id, int global_num_rows, 
                             int *row_start, int* row_end) 
{

   int    num_procs;
   int    size, extra;
   

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );


  /* this may look non-intuitive, but we have to be very careful that
      this function and the next are inverses - and avoiding overflow and
      rounding errors makes this difficult! */

   size = global_num_rows /num_procs;
   extra = global_num_rows - size*num_procs;

   *row_start = size*proc_id;
   *row_start += hypre_min(proc_id, extra);
   

   *row_end =  size*(proc_id+1);
   *row_end += hypre_min(proc_id+1, extra);
   *row_end = *row_end - 1;




   return(0);

}
