      SUBROUTINE DMVM( N, A, IA, JA, X, Y, OPT )
*
*     .. Scalar Arguments .. 
      INTEGER            N, OPT 
*     ..
*     .. Array Arguments .. 
      INTEGER            IA( * ), JA( * )       
      DOUBLE PRECISION   A( * ) 
      DOUBLE PRECISION   X( * ), Y( * ) 
*
*==============================================================
*
*  does a sparse matrix * vector multiplication to compute 
*
*                   y = A * x   if opt = 1
*                   y = A' * x  otherwise 
*
*  where A is stored in the sparse column format
*
*==============================================================
*
*     .. Parameter .. 
*
*     .. Local Scalars .. 
      INTEGER            I, J, K, K1, K2
*
*     Initialization 
*
      DO 10 J = 1, N
	 Y( J ) = 0.0d0
  10  CONTINUE
*
      IF( OPT.EQ.1 )THEN
*
*        Compute y = A*x
*
   	 DO 100 J = 1,N
	    K1 = JA( J )
	    K2 = JA( J+1 ) -1
	    DO 110 K = K1, K2
               I = IA( K )
*
*              Compute y(i) = y(i) + a(i,j) * x(j)
*
	       Y( I ) = Y( I ) + A( K )*X( J )
 110	    CONTINUE
 100   	 CONTINUE
*
      ELSE
*
*        Compute y = A'*x 
*
   	 DO 200 J = 1,N
	    K1 = JA( J )
	    K2 = JA( J+1 ) - 1
	    DO 210 K = K1, K2
               I = IA( K )
*
*              Compute y(j) = y(j) + a(i,j) * x(i)
*
	       Y( J ) = Y( J ) + A( K )* X( I )
 210	    CONTINUE
 200   	 CONTINUE
      ENDIF
*
      RETURN
*
*     End of MVM
*
      END
