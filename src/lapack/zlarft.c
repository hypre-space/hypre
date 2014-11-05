#include "../blas/hypre_blas.h"
#include "hypre_lapack.h"
#include "f2c.h"

/* Subroutine */ HYPRE_Int zlarft_(char *direct, char *storev, integer *n, integer *
	k, doublecomplex *v, integer *ldv, doublecomplex *tau, doublecomplex *
	t, integer *ldt)
{
/*  -- LAPACK auxiliary routine (version 3.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       September 30, 1994   


    Purpose   
    =======   

    ZLARFT forms the triangular factor T of a complex block reflector H   
    of order n, which is defined as a product of k elementary reflectors.   

    If DIRECT = 'F', H = H(1) H(2) . . . H(k) and T is upper triangular;   

    If DIRECT = 'B', H = H(k) . . . H(2) H(1) and T is lower triangular.   

    If STOREV = 'C', the vector which defines the elementary reflector   
    H(i) is stored in the i-th column of the array V, and   

       H  =  I - V * T * V'   

    If STOREV = 'R', the vector which defines the elementary reflector   
    H(i) is stored in the i-th row of the array V, and   

       H  =  I - V' * T * V   

    Arguments   
    =========   

    DIRECT  (input) CHARACTER*1   
            Specifies the order in which the elementary reflectors are   
            multiplied to form the block reflector:   
            = 'F': H = H(1) H(2) . . . H(k) (Forward)   
            = 'B': H = H(k) . . . H(2) H(1) (Backward)   

    STOREV  (input) CHARACTER*1   
            Specifies how the vectors which define the elementary   
            reflectors are stored (see also Further Details):   
            = 'C': columnwise   
            = 'R': rowwise   

    N       (input) INTEGER   
            The order of the block reflector H. N >= 0.   

    K       (input) INTEGER   
            The order of the triangular factor T (= the number of   
            elementary reflectors). K >= 1.   

    V       (input/output) COMPLEX*16 array, dimension   
                                 (LDV,K) if STOREV = 'C'   
                                 (LDV,N) if STOREV = 'R'   
            The matrix V. See further details.   

    LDV     (input) INTEGER   
            The leading dimension of the array V.   
            If STOREV = 'C', LDV >= max(1,N); if STOREV = 'R', LDV >= K.   

    TAU     (input) COMPLEX*16 array, dimension (K)   
            TAU(i) must contain the scalar factor of the elementary   
            reflector H(i).   

    T       (output) COMPLEX*16 array, dimension (LDT,K)   
            The k by k triangular factor T of the block reflector.   
            If DIRECT = 'F', T is upper triangular; if DIRECT = 'B', T is   
            lower triangular. The rest of the array is not used.   

    LDT     (input) INTEGER   
            The leading dimension of the array T. LDT >= K.   

    Further Details   
    ===============   

    The shape of the matrix V and the storage of the vectors which define   
    the H(i) is best illustrated by the following example with n = 5 and   
    k = 3. The elements equal to 1 are not stored; the corresponding   
    array elements are modified but restored on exit. The rest of the   
    array is not used.   

    DIRECT = 'F' and STOREV = 'C':         DIRECT = 'F' and STOREV = 'R':   

                 V = (  1       )                 V = (  1 v1 v1 v1 v1 )   
                     ( v1  1    )                     (     1 v2 v2 v2 )   
                     ( v1 v2  1 )                     (        1 v3 v3 )   
                     ( v1 v2 v3 )   
                     ( v1 v2 v3 )   

    DIRECT = 'B' and STOREV = 'C':         DIRECT = 'B' and STOREV = 'R':   

                 V = ( v1 v2 v3 )                 V = ( v1 v1  1       )   
                     ( v1 v2 v3 )                     ( v2 v2 v2  1    )   
                     (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )   
                     (     1 v3 )   
                     (        1 )   

    =====================================================================   


       Quick return if possible   

       Parameter adjustments */
    /* Table of constant values */
    static doublecomplex c_b2 = {0.,0.};
    static integer c__1 = 1;
    
    /* System generated locals */
    integer t_dim1, t_offset, v_dim1, v_offset, i__1, i__2, i__3, i__4;
    doublecomplex z__1;
    /* Local variables */
    static integer i__, j;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ HYPRE_Int zgemv_(char *, integer *, integer *, 
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *, 
	    integer *, doublecomplex *, doublecomplex *, integer *), 
	    ztrmv_(char *, char *, char *, integer *, doublecomplex *, 
	    integer *, doublecomplex *, integer *), 
	    zlacgv_(integer *, doublecomplex *, integer *);
    static doublecomplex vii;
#define t_subscr(a_1,a_2) (a_2)*t_dim1 + a_1
#define t_ref(a_1,a_2) t[t_subscr(a_1,a_2)]
#define v_subscr(a_1,a_2) (a_2)*v_dim1 + a_1
#define v_ref(a_1,a_2) v[v_subscr(a_1,a_2)]


    v_dim1 = *ldv;
    v_offset = 1 + v_dim1 * 1;
    v -= v_offset;
    --tau;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1 * 1;
    t -= t_offset;

    /* Function Body */
    if (*n == 0) {
	return 0;
    }

    if (lsame_(direct, "F")) {
	i__1 = *k;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__;
	    if (tau[i__2].r == 0. && tau[i__2].i == 0.) {

/*              H(i)  =  I */

		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    i__3 = t_subscr(j, i__);
		    t[i__3].r = 0., t[i__3].i = 0.;
/* L10: */
		}
	    } else {

/*              general case */

		i__2 = v_subscr(i__, i__);
		vii.r = v[i__2].r, vii.i = v[i__2].i;
		i__2 = v_subscr(i__, i__);
		v[i__2].r = 1., v[i__2].i = 0.;
		if (lsame_(storev, "C")) {

/*                 T(1:i-1,i) := - tau(i) * V(i:n,1:i-1)' * V(i:n,i) */

		    i__2 = *n - i__ + 1;
		    i__3 = i__ - 1;
		    i__4 = i__;
		    z__1.r = -tau[i__4].r, z__1.i = -tau[i__4].i;
		    zgemv_("Conjugate transpose", &i__2, &i__3, &z__1, &v_ref(
			    i__, 1), ldv, &v_ref(i__, i__), &c__1, &c_b2, &
			    t_ref(1, i__), &c__1);
		} else {

/*                 T(1:i-1,i) := - tau(i) * V(1:i-1,i:n) * V(i,i:n)' */

		    if (i__ < *n) {
			i__2 = *n - i__;
			zlacgv_(&i__2, &v_ref(i__, i__ + 1), ldv);
		    }
		    i__2 = i__ - 1;
		    i__3 = *n - i__ + 1;
		    i__4 = i__;
		    z__1.r = -tau[i__4].r, z__1.i = -tau[i__4].i;
		    zgemv_("No transpose", &i__2, &i__3, &z__1, &v_ref(1, i__)
			    , ldv, &v_ref(i__, i__), ldv, &c_b2, &t_ref(1, 
			    i__), &c__1);
		    if (i__ < *n) {
			i__2 = *n - i__;
			zlacgv_(&i__2, &v_ref(i__, i__ + 1), ldv);
		    }
		}
		i__2 = v_subscr(i__, i__);
		v[i__2].r = vii.r, v[i__2].i = vii.i;

/*              T(1:i-1,i) := T(1:i-1,1:i-1) * T(1:i-1,i) */

		i__2 = i__ - 1;
		ztrmv_("Upper", "No transpose", "Non-unit", &i__2, &t[
			t_offset], ldt, &t_ref(1, i__), &c__1);
		i__2 = t_subscr(i__, i__);
		i__3 = i__;
		t[i__2].r = tau[i__3].r, t[i__2].i = tau[i__3].i;
	    }
/* L20: */
	}
    } else {
	for (i__ = *k; i__ >= 1; --i__) {
	    i__1 = i__;
	    if (tau[i__1].r == 0. && tau[i__1].i == 0.) {

/*              H(i)  =  I */

		i__1 = *k;
		for (j = i__; j <= i__1; ++j) {
		    i__2 = t_subscr(j, i__);
		    t[i__2].r = 0., t[i__2].i = 0.;
/* L30: */
		}
	    } else {

/*              general case */

		if (i__ < *k) {
		    if (lsame_(storev, "C")) {
			i__1 = v_subscr(*n - *k + i__, i__);
			vii.r = v[i__1].r, vii.i = v[i__1].i;
			i__1 = v_subscr(*n - *k + i__, i__);
			v[i__1].r = 1., v[i__1].i = 0.;

/*                    T(i+1:k,i) :=   
                              - tau(i) * V(1:n-k+i,i+1:k)' * V(1:n-k+i,i) */

			i__1 = *n - *k + i__;
			i__2 = *k - i__;
			i__3 = i__;
			z__1.r = -tau[i__3].r, z__1.i = -tau[i__3].i;
			zgemv_("Conjugate transpose", &i__1, &i__2, &z__1, &
				v_ref(1, i__ + 1), ldv, &v_ref(1, i__), &c__1,
				 &c_b2, &t_ref(i__ + 1, i__), &c__1);
			i__1 = v_subscr(*n - *k + i__, i__);
			v[i__1].r = vii.r, v[i__1].i = vii.i;
		    } else {
			i__1 = v_subscr(i__, *n - *k + i__);
			vii.r = v[i__1].r, vii.i = v[i__1].i;
			i__1 = v_subscr(i__, *n - *k + i__);
			v[i__1].r = 1., v[i__1].i = 0.;

/*                    T(i+1:k,i) :=   
                              - tau(i) * V(i+1:k,1:n-k+i) * V(i,1:n-k+i)' */

			i__1 = *n - *k + i__ - 1;
			zlacgv_(&i__1, &v_ref(i__, 1), ldv);
			i__1 = *k - i__;
			i__2 = *n - *k + i__;
			i__3 = i__;
			z__1.r = -tau[i__3].r, z__1.i = -tau[i__3].i;
			zgemv_("No transpose", &i__1, &i__2, &z__1, &v_ref(
				i__ + 1, 1), ldv, &v_ref(i__, 1), ldv, &c_b2, 
				&t_ref(i__ + 1, i__), &c__1);
			i__1 = *n - *k + i__ - 1;
			zlacgv_(&i__1, &v_ref(i__, 1), ldv);
			i__1 = v_subscr(i__, *n - *k + i__);
			v[i__1].r = vii.r, v[i__1].i = vii.i;
		    }

/*                 T(i+1:k,i) := T(i+1:k,i+1:k) * T(i+1:k,i) */

		    i__1 = *k - i__;
		    ztrmv_("Lower", "No transpose", "Non-unit", &i__1, &t_ref(
			    i__ + 1, i__ + 1), ldt, &t_ref(i__ + 1, i__), &
			    c__1);
		}
		i__1 = t_subscr(i__, i__);
		i__2 = i__;
		t[i__1].r = tau[i__2].r, t[i__1].i = tau[i__2].i;
	    }
/* L40: */
	}
    }
    return 0;

/*     End of ZLARFT */

} /* zlarft_ */

#undef v_ref
#undef v_subscr
#undef t_ref
#undef t_subscr


