/* cliches.h

 m4 definitions.
*/

ifelse(THREADED,1,<<

/* ================================================================
 Self-scheduled parallelism macros.
*/

/*
Usage:  BARRIER(number_of_threads)
Argument:  $1 -- integer containing the number of threads being used
*/

define(BARRIER,<<
   if ($1 > 1) 
      abarrier($1);
>>)

/*
Usage:  FBARRIER(number_of_threads)
Argument:  $1 -- integer containing the number of threads being used
*/

define(FBARRIER,<<
   if ($1 > 1)
      fbarrier($1);
>>)

/* PLOOPEND waits until the last proc. resets the loop index to zero.
Usage:  PLOOPEND(index_array_name, array_index)
Arguments:  $1 -- the name of the array of indices
            $2 -- an index of the given array
To wait until element n of array "indx" is set to zero, enter 
   PLOOPEND(indx,n)
*/

define(PLOOPEND,<<
   loopend($1[$2]);
>>)

/* IWAIT is used to examine values of the global flag ipencil in calchyd
 If there is only one thread, this is a noop.
Usage:  IWAIT(flag_array_name, array_index)
Arguments:  $1 -- the name of an array of flags
            $2 -- an index of the given array
*/
define(IWAIT,<<
   iwaiter(ipencil[$1]);
>>)


/* PLOOP parallel loop macro.
 Example:
     PLOOP(z,lz,mz,indx,3,<<body>>)
 The indx used ($5) must not be reused for a loop
 until a synch. point. guarantees all threads have exited.

Usage: PLOOP(increment_variable, loop_initial_value, loop_stopping value,
             number_of_threads, index_array_name, array_index,
             <<loop_body>>)
Arguments:  $1 -- the name of the increment variable for the loop
            $2 -- the initial value for the increment variable
            $3 -- the stopping value for the increment variable
                  (loop will not be entered when increment reaches this value)
            $4 -- the name of an array of indices
            $5 -- an index of the given array
            $6 -- The body of the loop (enclose between << and >> )
*/

define(PLOOP,<<
/*
         $1 = ifetchadd( indx($4), $4 ) + $2
         do while ($1 .le. $3 )
$5
         $1 = ifetchadd( indx($4), $4 ) + $2
         enddo
         if ( $1 .eq. $3+msg_nthreads ) indx($4)=0
*/


   for ($1 = ifetchadd(&$4[$5]) + $2; 
        $1 <  $3;
        $1 = ifetchadd(&$4[$5]) + $2) {
      $6
   }

   if ($1 == $3)
      $4[$5] =0;
>>)


>>,<<


/* Dummy macros for non-threaded version. */
define(BARRIER,<<>>)
define(FBARRIER,<<>>)
define(PLOOPEND,<<>>)
define(IWAIT,<<>>)

/* PLOOP parallel loop macro.
 Example:
     PLOOP(z,lz,mz,indx,3,<<body>>)
*/
define(PLOOP,<<
/*
        $1 = $2
        do while ($1 .le. $3) !ploop started here
$5
        $1 = $1 + 1
	enddo   ! ploop expanded above
*/

   for ($1 = $2; $1 < $3; $1++) {
      $6
   }
>>)


>>)




