c cliches.h
c
c m4 definitions.
c

ifelse(THREADED,1,<<

c ================================================================
c Self-scheduled parallelism macros.
c

define(BARRIER,<<if (msg_nthreads.gt.1) call abarrier(msg_nthreads)>>)
define(FBARRIER,<<if (msg_nthreads.gt.1) call
fbarrier(msg_nthreads,istatus)>>)

c PLOOPEND waits until the last proc. resets the loop index to zero.
define(PLOOPEND,<<call loopend(indx($1))>>)

c IWAIT is used to examine values of the global flag ipencil in calchyd
c If there is only one thread, this is a noop.
define(IWAIT,<<call iwaiter(ipencil($1))>>)

c PLOOP parallel loop macro.
c Example:
c     PLOOP(z,lz,mz,3,<<body>>)
c The indx used ($4) must not be reused for a loop
c until a synch. point. guarantees all threads have exited.

define(PLOOP,<<$1 = ifetchadd( indx($4), $4 ) + $2
         do while ($1 .le. $3 )
$5
         $1 = ifetchadd( indx($4), $4 ) + $2
         enddo
         if ( $1 .eq. $3+msg_nthreads ) indx($4)=0>>)

c================================================================
c >>,<<
c================================================================

c Dummy macros for non-threaded version.
define(BARRIER,<<>>)
define(FBARRIER,<<>>)
define(PLOOPEND,<<>>)
define(IWAIT,<<>>)

c PLOOP parallel loop macro.
c Example:
c     PLOOP(z,lz,mz,3,<<body>>)

define(PLOOP,<<$1 = $2
        do while ($1 .le. $3) !ploop started here
$5
        $1 = $1 + 1
	enddo   ! ploop expanded above

	>>)
c================================================================
c >>)





