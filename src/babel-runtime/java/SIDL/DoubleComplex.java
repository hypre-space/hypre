/*
 * -------------------------------------------------------------------------
 * $Id: DoubleComplex.java,v 1.4 2003/04/07 21:44:19 painter Exp $
 * -------------------------------------------------------------------------
 * Copyright (c) 1997 - 1998 by Visual Numerics, Inc. All rights reserved.
 *
 * Permission to use, copy, modify, and distribute this software is freely
 * granted by Visual Numerics, Inc., provided that the copyright notice
 * above and the following warranty disclaimer are preserved in human
 * readable form.
 *
 * Because this software is licenses free of charge, it is provided
 * "AS IS", with NO WARRANTY.  TO THE EXTENT PERMITTED BY LAW, VNI
 * DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
 * TO ITS PERFORMANCE, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 * VNI WILL NOT BE LIABLE FOR ANY DAMAGES WHATSOEVER ARISING OUT OF THE USE
 * OF OR INABILITY TO USE THIS SOFTWARE, INCLUDING BUT NOT LIMITED TO DIRECT,
 * INDIRECT, SPECIAL, CONSEQUENTIAL, PUNITIVE, AND EXEMPLARY DAMAGES, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. 
 *
 * -------------------------------------------------------------------------
 */

/*
 * This file has been modified from the original VNI file.  In particular,
 * the namespace has been changed to SIDL and the name has been changed to
 * DoubleComplex.  Holder and array inner classes have been added, as well.
 */

package SIDL;
 
import SIDL.Sfun;
import java.lang.String;

/**
 * This class implements complex numbers. It provides the basic operations
 * (addition, subtraction, multiplication, division) as well as a set of
 * complex functions.
 *
 * The binary operations have the form, where op is <code>plus</code>,
 * <code>minus</code>, <code>times</code> or <code>over</code>.
 * <pre>
 * public static DoubleComplex op(DoubleComplex x, DoubleComplex y)   // x op y
 * public static DoubleComplex op(DoubleComplex x, double y)    // x op y
 * public static DoubleComplex op(double x, DoubleComplex y)    // x op y
 * public DoubleComplex op(DoubleComplex y)                     // this op y
 * public DoubleComplex op(double y)                      // this op y
 * public DoubleComplex opReverse(double x)               // x op this
 * </pre>
 *
 * The functions in this class follow the rules for complex  arithmetic
 * as defined C9x Annex G:"IEC 559-compatible complex arithmetic."
 * The API is not the same, but handling of infinities, NaNs, and positive
 * and negative zeros is intended to follow the same rules.
 *
 * This class depends on the standard java.lang.Math class following
 * certain rules, as defined in the C9x Annex F, for the handling of
 * infinities, NaNs, and positive and negative zeros. Sun's specification
 * is that java.lang.Math should reproduce the results in the Sun's fdlibm
 * C library. This library appears to follow the Annex F specification.
 * At least on Windows, Sun's JDK 1.0 and 1.1 do NOT follow this specification.
 * Sun's JDK 1.2(RC2) does follow the Annex F specification. Thesefore,
 * this class will not give the expected results for edge cases with
 * JDK 1.0 and 1.1.
 */
public class DoubleComplex implements java.io.Serializable, Cloneable {
  /**  
   *  @serial Real part of the DoubleComplex.
   */
  private double re;

  /**
   *  @serial Imaginary part of the DoubleComplex.
   */
  private double im;

  /**
   *  Serialization ID
   */
  static final long serialVersionUID = -633126172485117692L;

  /**
   *  String used in converting DoubleComplex to String.
   *  Default is "i", but sometimes "j" is desired.
   *  Note that this is set for the class, not for
   *  a particular instance of a DoubleComplex.
   */
  public static String suffix = "i";

  private final static long negZeroBits =
    java.lang.Double.doubleToLongBits(1.0/java.lang.Double.NEGATIVE_INFINITY);

  /** 
   *  Constructs a DoubleComplex equal to the argument.
   *  @param  z  A DoubleComplex object
   *      If z is null then a NullPointerException is thrown.
   */
  public DoubleComplex(DoubleComplex z) {
    re = z.re;
    im = z.im;
  }

  /** 
   *  Constructs a DoubleComplex with real and imaginary parts given
   *  by the input arguments.
   *  @param re A double value equal to the real part of the
   *            DoubleComplex object.
   *  @param im A double value equal to the imaginary part of
   *            the DoubleComplex object.
   */
  public DoubleComplex(double re, double im) {
    this.re = re;
    this.im = im;
  }

  /** 
   *  Constructs a DoubleComplex with a zero imaginary part. 
   *  @param re A double value equal to the real part of DoubleComplex object.
   */
  public DoubleComplex(double re) {
    this.re = re;
    this.im = 0.0;
  }

  /**
   *  Constructs a DoubleComplex equal to zero.
   */
  public DoubleComplex() {
    re = 0.0;
    im = 0.0;
  }

  /** 
   *  Tests if this is a complex Not-a-Number (NaN) value. 
   *  @return  True if either component of the DoubleComplex object is NaN;
   *  false, otherwise. 
   */
  private boolean isNaN() {
    return (java.lang.Double.isNaN(re) || java.lang.Double.isNaN(im));
  }
  
  /** 
   *  Compares with another DoubleComplex. 
   *  <p><em>Note: To be useful in hashtables this method
   *  considers two NaN double values to be equal. This
   *  is not according to IEEE specification.</em>
   *  @param  z  A DoubleComplex object.
   *  @return True if the real and imaginary parts of this object
   *      are equal to their counterparts in the argument; false, otherwise.
   */
  public boolean equals(DoubleComplex z) {
    if (isNaN() && z.isNaN()) {
      return true;
    } else {
      return (re == z.re  &&  im == z.im);
    }
  }

  /**
   *  Compares this object against the specified object.
   *  <p><em>Note: To be useful in hashtables this method
   *  considers two NaN double values to be equal. This
   *  is not according to IEEE specification</em>
   *  @param  obj  The object to compare with.
   *  @return True if the objects are the same; false otherwise.
   */
  public boolean equals(Object obj) {
    if (obj == null) {
      return false;
    } else if (obj instanceof DoubleComplex) {
      return equals((DoubleComplex)obj);
    } else {
      return false;
    }
  }

  /**
   *  Returns a hashcode for this DoubleComplex.
   *  @return  A hash code value for this object. 
   */
  public int hashCode() {
    long re_bits = java.lang.Double.doubleToLongBits(re);
    long im_bits = java.lang.Double.doubleToLongBits(im);
    return (int)((re_bits^im_bits)^((re_bits^im_bits)>>32));
  }

  /** 
   *  Returns the real part of a DoubleComplex object. 
   *  @return  The real part of z.
   */
  public double real() {
    return re;
  }

  /** 
   *  Returns the imaginary part of a DoubleComplex object. 
   *  @param  z  A DoubleComplex object.
   *  @return  The imaginary part of z.
   */
  public double imag() {
    return im;
  }

  /**
   * Set the real and imaginary parts of the DoubleComplex object.
   */
  public void set(double real, double imag) {
    re = real;
    im = imag;
  }

  /** 
   *  Returns the real part of a DoubleComplex object. 
   *  @param  z  A DoubleComplex object.
   *  @return  The real part of z.
   */
  public static double real(DoubleComplex z) {
    return z.re;
  }

  /** 
   *  Returns the imaginary part of a DoubleComplex object. 
   *  @param  z  A DoubleComplex object.
   *  @return  The imaginary part of z.
   */
  public static double imag(DoubleComplex z) {
    return z.im;
  }

  /** 
   *  Returns the negative of a DoubleComplex object, -z. 
   *  @param  z  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to
   *      the negative of the argument.
   */
  public static DoubleComplex negative(DoubleComplex z) {
    return new DoubleComplex(-z.re, -z.im);
  }
  
  /** 
   *  Returns the complex conjugate of a DoubleComplex object.
   *  @param  z  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to
   *          complex conjugate of z.
   */
  public static DoubleComplex conjugate(DoubleComplex z) {
    return new DoubleComplex(z.re, -z.im);
  }
  
  /** 
   *  Returns the sum of two DoubleComplex objects, x+y.
   *  @param  x  A DoubleComplex object.
   *  @param  y  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to x+y.
   */
  public static DoubleComplex plus(DoubleComplex x, DoubleComplex y) {
    return new DoubleComplex(x.re+y.re, x.im+y.im);
  }

  /** 
   *  Returns the sum of a DoubleComplex and a double, x+y. 
   *  @param  x  A DoubleComplex object.
   *  @param  y  A double value.
   *  @return A newly constructed DoubleComplex initialized to x+y.
   */
  public static DoubleComplex plus(DoubleComplex x, double y) {
    return new DoubleComplex(x.re+y, x.im);
  }

  /** 
   *  Returns the sum of a double and a DoubleComplex, x+y. 
   *  @param  x  A double value.
   *  @param  y  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to x+y.
   */
  public static DoubleComplex plus(double x, DoubleComplex y) {
    return new DoubleComplex(x+y.re, y.im);
  }

  /** 
   *  Returns the sum of this DoubleComplex and another DoubleComplex, this+y. 
   *  @param  y  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to this+y.
   */
  public DoubleComplex plus(DoubleComplex y) {
    return new DoubleComplex(re+y.re, im+y.im);
  }

  /** 
   *  Returns the sum of this DoubleComplex a double, this+y. 
   *  @param  y  A double value.
   *  @return A newly constructed DoubleComplex initialized to this+y.
   */
  public DoubleComplex plus(double y) {
    return new DoubleComplex(re+y, im);
  }
  
  /** 
   *  Returns the sum of this DoubleComplex and a double, x+this. 
   *  @param  x  A double value.
   *  @return A newly constructed DoubleComplex initialized to x+this.
   */
  public DoubleComplex plusReverse(double x) {
    return new DoubleComplex(re+x, im);
  }

  /** 
   *  Returns the difference of two DoubleComplex objects, x-y.
   *  @param  x  A DoubleComplex object.
   *  @param  y  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to x-y.
   */
  public static DoubleComplex minus(DoubleComplex x, DoubleComplex y) {
    return new DoubleComplex(x.re-y.re, x.im-y.im);
  }

  /** 
   *  Returns the difference of a DoubleComplex object and a double, x-y. 
   *  @param  x  A DoubleComplex object.
   *  @param  y  A double value.
   *  @return A newly constructed DoubleComplex initialized to x-y.
   */
  public static DoubleComplex minus(DoubleComplex x, double y) {
    return new DoubleComplex(x.re-y, x.im);
  }
  
  /** 
   *  Returns the difference of a double and a DoubleComplex object, x-y. 
   *  @param  x  A double value.
   *  @param  y  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to x-y..
   */
  public static DoubleComplex minus(double x, DoubleComplex y) {
    return new DoubleComplex(x-y.re, -y.im);
  }

  /** 
   *  Returns the difference of this DoubleComplex object and
   *  another DoubleComplex object, this-y. 
   *  @param  y  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to this-y.
   */
  public DoubleComplex minus(DoubleComplex y) {
    return new DoubleComplex(re-y.re, im-y.im);
  }

  /** 
   *  Subtracts a double from this DoubleComplex and returns the difference,
   *  this-y.
   *  @param  y  A double value.
   *  @return A newly constructed DoubleComplex initialized to this-y.
   */
  public DoubleComplex minus(double y) {
    return new DoubleComplex(re-y, im);
  }

  /** 
   *  Returns the difference of this DoubleComplex object and a double, this-y.
   *  @param  y  A double value.
   *  @return A newly constructed DoubleComplex initialized to x-this.
   */
  public DoubleComplex minusReverse(double x) {
    return new DoubleComplex(x-re, -im);
  }

  /** 
   *  Returns the product of two DoubleComplex objects, x*y. 
   *  @param  x  A DoubleComplex object.
   *  @param  y  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to x*y.
   */
  public static DoubleComplex times(DoubleComplex x, DoubleComplex y) {
    DoubleComplex t = new DoubleComplex( x.re*y.re-x.im*y.im,
                                         x.re*y.im+x.im*y.re);
    if (java.lang.Double.isNaN(t.re) &&
        java.lang.Double.isNaN(t.im)) timesNaN(x, y, t);
    return t;
  }

  /*
   *  Returns sign(b)*|a|.
   */
  private static double copysign(double a, double b) {
    double abs = Math.abs(a);
    return ((b < 0) ? -abs : abs);
  }

  /**
   *  Recovers infinities when computed x*y = NaN+i*NaN.
   *  This code is not part of times(), so that times
   *  could be inlined by an optimizing compiler.
   *  <p>
   *  This algorithm is adapted from the C9x Annex G:
   *  "IEC 559-compatible complex arithmetic."
   *  @param  x  First DoubleComplex operand.
   *  @param  y  Second DoubleComplex operand.
   *  @param  t  The product x*y, computed without regard to NaN.
   *        The real and/or the imaginary part of t is
   *        expected to be NaN.
   *  @return  The corrected product of x*y.
   */
  private static void timesNaN(DoubleComplex x,
                               DoubleComplex y,
                               DoubleComplex t) {
    boolean  recalc = false;
    double  a = x.re;
    double  b = x.im;
    double  c = y.re;
    double  d = y.im;

    if (java.lang.Double.isInfinite(a) || java.lang.Double.isInfinite(b)) {
      // x is infinite
      a = copysign(java.lang.Double.isInfinite(a)?1.0:0.0, a);
      b = copysign(java.lang.Double.isInfinite(b)?1.0:0.0, b);
      if (java.lang.Double.isNaN(c))  c = copysign(0.0, c);
      if (java.lang.Double.isNaN(d))  d = copysign(0.0, d);
      recalc = true;
    }

    if (java.lang.Double.isInfinite(c) || java.lang.Double.isInfinite(d)) {
      // x is infinite
      a = copysign(java.lang.Double.isInfinite(c)?1.0:0.0, c);
      b = copysign(java.lang.Double.isInfinite(d)?1.0:0.0, d);
      if (java.lang.Double.isNaN(a))  a = copysign(0.0, a);
      if (java.lang.Double.isNaN(b))  b = copysign(0.0, b);
      recalc = true;
    }

    if (!recalc) {
      if (java.lang.Double.isInfinite(a*c) ||
          java.lang.Double.isInfinite(b*d) ||
          java.lang.Double.isInfinite(a*d) ||
          java.lang. Double.isInfinite(b*c)) {
        // Change all NaNs to 0
        if (java.lang.Double.isNaN(a))  a = copysign(0.0, a);
        if (java.lang.Double.isNaN(b))  b = copysign(0.0, b);
        if (java.lang.Double.isNaN(c))  c = copysign(0.0, c);
        if (java.lang.Double.isNaN(d))  d = copysign(0.0, d);
        recalc = true;
      }
    }

    if (recalc) {
      t.re = java.lang.Double.POSITIVE_INFINITY * (a*c - b*d);
      t.im = java.lang.Double.POSITIVE_INFINITY * (a*d + b*c);
    }
  }

  /** 
   *  Returns the product of a DoubleComplex object and a double, x*y. 
   *  @param  x  A DoubleComplex object.
   *  @param  y  A double value.
   *  @return  A newly constructed DoubleComplex initialized to x*y.
   */
  public static DoubleComplex times(DoubleComplex x, double y) {
    return new DoubleComplex(x.re*y, x.im*y);
  }

  /** 
   *  Returns the product of a double and a DoubleComplex object, x*y. 
   *  @param  x  A double value.
   *  @param  y  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to x*y.
   */
  public static DoubleComplex times(double x, DoubleComplex y) {
    return new DoubleComplex(x*y.re, x*y.im);
  }

  /** 
   * Returns the product of this DoubleComplex object and another
   * DoubleComplex object, this*y. 
   * @param  y  A DoubleComplex object.
   * @return  A newly constructed DoubleComplex initialized to this*y.
   */
  public DoubleComplex times(DoubleComplex y) {
    return times(this,y);
  }

  /** 
   *  Returns the product of this DoubleComplex object and a double, this*y.
   *  @param  y  A double value.
   *  @return A newly constructed DoubleComplex initialized to this*y.
   */
  public DoubleComplex times(double y) {
    return new DoubleComplex(re*y, im*y);
  }

  /** 
   *  Returns the product of a double and this DoubleComplex, x*this. 
   *  @param  y  A double value.
   *  @return A newly constructed DoubleComplex initialized to x*this.
   */
  public DoubleComplex timesReverse(double x) {
    return new DoubleComplex(x*re, x*im);
  }

  private static boolean isFinite(double x) {
    return !(java.lang.Double.isInfinite(x) || java.lang.Double.isNaN(x));
  }

  /** 
   *  Returns DoubleComplex object divided by a DoubleComplex object, x/y. 
   *  @param  x  The numerator, a DoubleComplex object.
   *  @param  y  The denominator, a DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to x/y.
   */
  public static DoubleComplex over(DoubleComplex x, DoubleComplex y) {
    double  a = x.re;
    double  b = x.im;
    double  c = y.re;
    double  d = y.im;

    double scale = Math.max(Math.abs(c), Math.abs(d));
    boolean isScaleFinite = isFinite(scale);
    if (isScaleFinite) {
      c /= scale;
      d /= scale;
    }

    double den = c*c + d*d;
    DoubleComplex z = new DoubleComplex((a*c+b*d)/den, (b*c-a*d)/den);
    
    if (isScaleFinite) {
      z.re /= scale;
      z.im /= scale;
    }

    // Recover infinities and zeros computed as NaN+iNaN.
    if (java.lang.Double.isNaN(z.re) && java.lang.Double.isNaN(z.im)) {
      if (den == 0.0  && (!java.lang.Double.isNaN(a) ||
                          !java.lang.Double.isNaN(b))) {
        double s = copysign(java.lang.Double.POSITIVE_INFINITY, c);
        z.re = s * a;
        z.im = s * b;
      
      } else if ((java.lang.Double.isInfinite(a) ||
                  java.lang.Double.isInfinite(b)) &&
        isFinite(c) && isFinite(d)) {
        a = copysign(java.lang.Double.isInfinite(a)?1.0:0.0, a);
        b = copysign(java.lang.Double.isInfinite(b)?1.0:0.0, b);
        z.re = java.lang.Double.POSITIVE_INFINITY * (a*c + b*d);
        z.im = java.lang.Double.POSITIVE_INFINITY * (b*c - a*d);
      
      } else if (java.lang.Double.isInfinite(scale)  &&
        isFinite(a) && isFinite(b)) {
        c = copysign(java.lang.Double.isInfinite(c)?1.0:0.0, c);
        d = copysign(java.lang.Double.isInfinite(d)?1.0:0.0, d);
        z.re = 0.0 * (a*c + b*d);
        z.im = 0.0 * (b*c - a*d);
      }
    }
    return z;
  }

  /** 
   *  Returns DoubleComplex object divided by a double, x/y.
   *  @param  x  The numerator, a DoubleComplex object.
   *  @param  y  The denominator, a double.
   *  @return A newly constructed DoubleComplex initialized to x/y.
   */
  public static DoubleComplex over(DoubleComplex x, double y) {
    return new DoubleComplex(x.re/y, x.im/y);
  }

  /** 
   *  Returns a double divided by a DoubleComplex object, x/y. 
   *  @param  x  A double value.
   *  @param  y  The denominator, a DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to x/y.
   */
  public static DoubleComplex over(double x, DoubleComplex y) {
    return y.overReverse(x);
  }

  /** 
   *  Returns this DoubleComplex object divided by another
   *  DoubleComplex object, this/y. 
   *  @param  y  The denominator, a DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to x/y.
   */
  public DoubleComplex over(DoubleComplex y) {
    return over(this, y);
  }

  /** 
   *  Returns this DoubleComplex object divided by double, this/y. 
   *  @param  y  The denominator, a double.
   *  @return  A newly constructed DoubleComplex initialized to x/y.
   */
  public DoubleComplex over(double y) {
    return over(this, y);
  }

  /** 
   *  Returns a double dividied by this DoubleComplex object, x/this. 
   *  @param  x  The numerator, a double.
   *  @return A newly constructed DoubleComplex initialized to x/this.
   */
  public DoubleComplex overReverse(double x) {
    double        den;
    double        t;
    DoubleComplex z;

    if (Math.abs(re) > Math.abs(im)) {
      t = im / re;
      den = re + im*t;
      z = new DoubleComplex(x/den, -x*t/den);
    } else {
      t = re / im;
      den = im + re*t;
      z = new DoubleComplex(x*t/den, -x/den);
    }
    return z;
  }

  /** 
   *  Returns the absolute value (modulus) of a DoubleComplex, |z|. 
   *  @param  z  A DoubleComplex object.
   *  @return A double value equal to the absolute value of the argument.
   */
  public static double abs(DoubleComplex z) {
    double x = Math.abs(z.re);
    double y = Math.abs(z.im);
    
    if (java.lang.Double.isInfinite(x) || java.lang.Double.isInfinite(y))
      return java.lang.Double.POSITIVE_INFINITY;
    
    if (x + y == 0.0) {
      return 0.0;
    } else if (x > y) {
      y /= x;
      return x*Math.sqrt(1.0+y*y);
    } else {
      x /= y;
      return y*Math.sqrt(x*x+1.0);
    }
  }

  /** 
   *  Returns the argument (phase) of a DoubleComplex, in radians,
   *  with a branch cut along the negative real axis.
   *  @param  z  A DoubleComplex object.
   *  @return A double value equal to the argument (or phase) of
   *          a DoubleComplex.  It is in the interval [-pi,pi].
   */
  public static double argument(DoubleComplex z) {
    return Math.atan2(z.im, z.re);
  }
  
  /** 
   *  Returns the square root of a DoubleComplex,
   *  with a branch cut along the negative real axis.
   *  @param  z  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized
   *          to square root of z. Its real part is non-negative.
   */
  public static DoubleComplex sqrt(DoubleComplex z) {
    DoubleComplex  result = new DoubleComplex();

    if (java.lang.Double.isInfinite(z.im)) {
      result.re = java.lang.Double.POSITIVE_INFINITY;
      result.im = z.im;
    } else if (java.lang.Double.isNaN(z.re)) {
      result.re = result.im = java.lang.Double.NaN;
    } else if (java.lang.Double.isNaN(z.im)) {
      if (java.lang.Double.isInfinite(z.re)) {
        if (z.re > 0) {
          result.re = z.re;
          result.im = z.im;
        } else {
          result.re = z.im;
          result.im = java.lang.Double.POSITIVE_INFINITY;
        }
      } else {
        result.re = result.im = java.lang.Double.NaN;
      }
    } else {
      // Numerically correct version of formula 3.7.27
      // in the NBS Hanbook, as suggested by Pete Stewart.
      double t = abs(z);
    
      if (Math.abs(z.re) <= Math.abs(z.im)) {
        // No cancellation in these formulas
        result.re = Math.sqrt(0.5*(t+z.re));
        result.im = Math.sqrt(0.5*(t-z.re));
      } else {
        // Stable computation of the above formulas
        if (z.re > 0) {
          result.re = t + z.re;
          result.im = Math.abs(z.im)*Math.sqrt(0.5/result.re);
          result.re = Math.sqrt(0.5*result.re);
        } else {
          result.im = t - z.re;
          result.re = Math.abs(z.im)*Math.sqrt(0.5/result.im);
          result.im = Math.sqrt(0.5*result.im);
        }
      }
      if (z.im < 0)
        result.im = -result.im;
    }
    return result;
  }

  /** 
   *  Returns the exponential of a DoubleComplex z, exp(z).
   *  @param  z  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to exponential
   *      of the argument. 
   */
  public static DoubleComplex exp(DoubleComplex z) {
    DoubleComplex result = new DoubleComplex();
    
    double r = Math.exp(z.re);

    double cosa = Math.cos(z.im);
    double sina = Math.sin(z.im);
    if (java.lang.Double.isInfinite(z.im) ||
        java.lang.Double.isNaN(z.im)      || Math.abs(cosa)>1) {
      cosa = sina = java.lang.Double.NaN;
    }

    if (java.lang.Double.isInfinite(z.re) ||
        java.lang.Double.isInfinite(r)) {
      if (z.re < 0) {
        r = 0;
        if (java.lang.Double.isInfinite(z.im) ||
            java.lang.Double.isNaN(z.im)) {
          cosa = sina = 0;
        } else {
          cosa /= java.lang.Double.POSITIVE_INFINITY;
          sina /= java.lang.Double.POSITIVE_INFINITY;
        }
      } else {
        r = z.re;
        if (java.lang.Double.isNaN(z.im)) cosa = 1;
      }
    }
        
    if (z.im == 0.0) {
      result.re = r;
      result.im = z.im;
    } else {
      result.re = r*cosa;
      result.im = r*sina;
    }
    return result;
  }

  /** 
   *  Returns the logarithm of a DoubleComplex z,
   *  with a branch cut along the negative real axis.
   *  @param  z  A DoubleComplex object.
   *  @return Newly constructed DoubleComplex initialized to logarithm of
   *          the argument. Its imaginary part is in the interval [-i*pi,i*pi].
   */
  public static DoubleComplex log(DoubleComplex z) {
    DoubleComplex  result = new DoubleComplex();

    if (java.lang.Double.isNaN(z.re)) {
      result.re = result.im = z.re;
      if (java.lang.Double.isInfinite(z.im))
        result.re = java.lang.Double.POSITIVE_INFINITY;
    } else if (java.lang.Double.isNaN(z.im)) {
      result.re = result.im = z.im;
      if (java.lang.Double.isInfinite(z.re))
        result.re = java.lang.Double.POSITIVE_INFINITY;
    } else {
      result.re = Math.log(abs(z));
      result.im = argument(z);
    }
    return result;
  }

  /** 
   *  Returns the sine of a DoubleComplex. 
   *  @param  z  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to
   *          sine of the argument.
   */
  public static DoubleComplex sin(DoubleComplex z) {
    // sin(z) = -i*sinh(i*z)
    DoubleComplex iz = new DoubleComplex(-z.im,z.re);
    DoubleComplex s = sinh(iz);
    double re = s.im;
    s.im = -s.re;
    s.re = re;
    return s;
  }

  /** 
   *  Returns the cosine of a DoubleComplex. 
   *  @param  z  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to
   *          cosine of the argument.
   */
  public static DoubleComplex cos(DoubleComplex z) {
    // cos(z) = cosh(i*z)
    return cosh(new DoubleComplex(-z.im,z.re));
  }

  /** 
   *  Returns the tangent of a DoubleComplex. 
   *  @param  z  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized
   *      to tangent of the argument.
   */
  public static DoubleComplex tan(DoubleComplex z) {
    // tan = -i*tanh(i*z)
    DoubleComplex iz = new DoubleComplex(-z.im,z.re);
    DoubleComplex s = tanh(iz);
    double re = s.im;
    s.im = -s.re;
    s.re = re;
    return s;
  }

  /** 
   *  Returns the inverse sine (arc sine) of a DoubleComplex,
   *  with branch cuts outside the interval [-1,1] along the
   *  real axis.
   *  @param  z  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to inverse
   *      (arc) sine of the argument. The real part of the
   *      result is in the interval [-pi/2,+pi/2].
   */
  public static DoubleComplex asin(DoubleComplex z) {
    DoubleComplex  result = new DoubleComplex();

    double r = abs(z);

    if (java.lang.Double.isInfinite(r)) {
      boolean infiniteX = java.lang.Double.isInfinite(z.re);
      boolean infiniteY = java.lang.Double.isInfinite(z.im);
      if (infiniteX) {
        double  pi2 = 0.5*Math.PI;
        result.re = (z.re>0 ? pi2 : -pi2);
        if (infiniteY) result.re /= 2;
      } else if (infiniteY) {
        result.re = z.re/java.lang.Double.POSITIVE_INFINITY;
      }
      if (java.lang.Double.isNaN(z.im)) {
        result.im = -z.re;
        result.re = z.im;
      } else {
        result.im = z.im*java.lang.Double.POSITIVE_INFINITY;
      }
      return result;
    } else if (java.lang.Double.isNaN(r)) {
      result.re = result.im = java.lang.Double.NaN;
      if (z.re == 0)  result.re = z.re;
    } else if (r < 2.58095e-08) {
      // sqrt(6.0*dmach(3)) = 2.58095e-08
      result.re = z.re;
      result.im = z.im;
    } else if (z.re == 0) {
      result.re = 0;
      result.im = Sfun.asinh(z.im);
    } else if (r <= 0.1) {
      DoubleComplex z2 = times(z,z);
      //log(eps)/log(rmax) = 8 where rmax = 0.1
      for (int i = 1;  i <= 8;  i++) {
        double twoi = 2*(8-i) + 1;
        result = times(times(result,z2),twoi/(twoi+1.0));
        result.re += 1.0/twoi;
      }
      result = result.times(z);
    } else {
      // A&S 4.4.26
      // asin(z) = -i*log(z+sqrt(1-z)*sqrt(1+z))
      // or, since log(iz) = log(z) +i*pi/2,
      // asin(z) = pi/2 - i*log(z+sqrt(z+1)*sqrt(z-1))
      DoubleComplex w = ((z.im < 0) ? negative(z) : z);
      DoubleComplex sqzp1 = sqrt(plus(w,1.0));
      if (sqzp1.im < 0.0)
        sqzp1 = negative(sqzp1);
      DoubleComplex sqzm1 = sqrt(minus(w,1.0));
      result = log(plus(w,times(sqzp1,sqzm1)));

      double rx = result.re;
      result.re = 0.5*Math.PI + result.im;
      result.im = -rx;
    }

    if (result.re > 0.5*Math.PI) {
      result.re = Math.PI - result.re;
      result.im = -result.im;
    }
    if (result.re < -0.5*Math.PI) {
      result.re = -Math.PI - result.re;
      result.im = -result.im;
    }
    if (z.im < 0) {
      result.re = -result.re;
      result.im = -result.im;
    }
    return result;
  }

  /** 
   *  Returns the inverse cosine (arc cosine) of a DoubleComplex,
   *  with branch cuts outside the interval [-1,1] along the
   *  real axis.
   *  @param  z  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to
   *      inverse (arc) cosine of the argument.
   *      The real part of the result is in the interval [0,pi].
   */
  public static DoubleComplex acos(DoubleComplex z) {
    DoubleComplex  result = new DoubleComplex();
    double r = abs(z);

    if (java.lang.Double.isInfinite(z.re) && java.lang.Double.isNaN(z.im)) {
      result.re = java.lang.Double.NaN;
      result.im = java.lang.Double.NEGATIVE_INFINITY;
    } else if (java.lang.Double.isInfinite(r)) {
      result.re = Math.atan2(Math.abs(z.im),z.re);
      result.im = z.im*java.lang.Double.NEGATIVE_INFINITY;
    } else if (r == 0) {
      result.re = Math.PI/2;
      result.im = -z.im;
    } else {
      result = minus(Math.PI/2,asin(z));
    }
    return result;
  }

  /** 
   * Returns the inverse tangent (arc tangent) of a DoubleComplex,
   * with branch cuts outside the interval [-i,i] along the
   * imaginary axis.
   * @param  z  A DoubleComplex object.
   * @return  A newly constructed DoubleComplex initialized to
   *      inverse (arc) tangent of the argument.
   *      Its real part is in the interval [-pi/2,pi/2].
   */
  public static DoubleComplex atan(DoubleComplex z) {
    DoubleComplex  result = new DoubleComplex();
    double  r = abs(z);

    if (java.lang.Double.isInfinite(r)) {
      double  pi2 = 0.5*Math.PI;
      double im = (java.lang.Double.isNaN(z.im) ? 0 : z.im);
      result.re = (z.re<0 ? -pi2 : pi2);
      result.im = (im<0 ? -1 : 1)/java.lang.Double.POSITIVE_INFINITY;
      if (java.lang.Double.isNaN(z.re))  result.re = z.re;
    } else if (java.lang.Double.isNaN(r)) {
      result.re = result.im = java.lang.Double.NaN;
      if (z.im == 0)  result.im = z.im;
    } else if (r < 1.82501e-08) {
      // sqrt(3.0*dmach(3)) = 1.82501e-08
      result.re = z.re;
      result.im = z.im;
    } else if (r < 0.1) {
      DoubleComplex z2 = times(z,z);
      // -0.4343*log(dmach(3))+1 = 17
      for (int k = 0;  k < 17;  k++) {
        DoubleComplex temp = times(z2,result);
        int twoi = 2*(17-k) - 1;
        result.re = 1.0/twoi - temp.re;
        result.im = -temp.im;
      }
      result = result.times(z);
    } else if (r < 9.0072e+15) {
      // 1.0/dmach(3) = 9.0072e+15
      double r2 = r*r;
      result.re = 0.5*Math.atan2(2*z.re,1.0-r2);
      result.im = 0.25*Math.log((r2+2*z.im+1)/(r2-2*z.im+1));
    } else {
      result.re = ((z.re < 0.0) ? -0.5*Math.PI : 0.5*Math.PI);
    }
    return result;
  }

  /** 
   * Returns the hyperbolic sine of a DoubleComplex. 
   * @param  z  A DoubleComplex object.
   * @return  A newly constructed DoubleComplex initialized to hyperbolic
   *      sine of the argument.
   */
  public static DoubleComplex sinh(DoubleComplex z) {
    double  coshx = Sfun.cosh(z.re);
    double  sinhx = Sfun.sinh(z.re);
    double  cosy  = Math.cos(z.im);
    double  siny  = Math.sin(z.im);
    boolean infiniteX = java.lang.Double.isInfinite(coshx);
    boolean infiniteY = java.lang.Double.isInfinite(z.im);
    DoubleComplex result;

    if (z.im == 0) {
      result = new DoubleComplex(Sfun.sinh(z.re));
    } else {
      // A&S 4.5.49
      result = new DoubleComplex(sinhx*cosy, coshx*siny);
      if (infiniteY) {
        result.im = java.lang.Double.NaN;
        if (z.re == 0)  result.re = 0;
      }
      if (infiniteX) {
        result.re = z.re*cosy;
        result.im = z.re*siny;
        if (z.im == 0)  result.im = 0;
        if (infiniteY) result.re = z.im;
      }
    }
    return result;
  }

  /** 
   * Returns the hyperbolic cosh of a DoubleComplex. 
   * @param  z  A DoubleComplex object.
   * @return  A newly constructed DoubleComplex initialized to
   *      the hyperbolic cosine of the argument.
   */
  public static DoubleComplex cosh(DoubleComplex z) {
    if (z.im == 0) {
      return new DoubleComplex(Sfun.cosh(z.re));
    }
    
    double  coshx = Sfun.cosh(z.re);
    double  sinhx = Sfun.sinh(z.re);
    double  cosy  = Math.cos(z.im);
    double  siny  = Math.sin(z.im);
    boolean infiniteX = java.lang.Double.isInfinite(coshx);
    boolean infiniteY = java.lang.Double.isInfinite(z.im);

    // A&S 4.5.50
    DoubleComplex result = new DoubleComplex(coshx*cosy, sinhx*siny);
    if (infiniteY)   result.re = java.lang.Double.NaN;
    if (z.re == 0) {
      result.im = 0;
    } else if (infiniteX) {
      result.re = z.re*cosy;
      result.im = z.re*siny;
      if (z.im == 0)  result.im = 0;
      if (java.lang.Double.isNaN(z.im)) {
        result.re = z.re;
      } else if (infiniteY) {
        result.re = z.im;
      }
    }
    return result;
  }

  /** 
   * Returns the hyperbolic tanh of a DoubleComplex. 
   * @param  z  A DoubleComplex object.
   * @return  A newly constructed DoubleComplex initialized to
   *      the hyperbolic tangent of the argument.
   */
  public static DoubleComplex tanh(DoubleComplex z) {
    double  sinh2x = Sfun.sinh(2*z.re);
    
    if (z.im == 0) {
      return new DoubleComplex(Sfun.tanh(z.re));
    } else if (sinh2x == 0) {
      return new DoubleComplex(0,Math.tan(z.im));
    }

    double  cosh2x = Sfun.cosh(2*z.re);
    double  cos2y  = Math.cos(2*z.im);
    double  sin2y  = Math.sin(2*z.im);
    boolean infiniteX = java.lang.Double.isInfinite(cosh2x);

    // Workaround for bug in JDK 1.2beta4
    if (java.lang.Double.isInfinite(z.im) || java.lang.Double.isNaN(z.im)) {
      cos2y = sin2y = java.lang.Double.NaN;  
    }

    if (infiniteX)
      return new DoubleComplex(z.re > 0 ? 1 : -1);

    // A&S 4.5.51
    double den = (cosh2x + cos2y);
    return new DoubleComplex(sinh2x/den, sin2y/den);
  }
  
  /** 
   *  Returns the DoubleComplex z raised to the x power,
   *  with a branch cut for the first parameter (z) along the
   *  negative real axis.
   *  @param  z  A DoubleComplex object.
   *  @param  x  A double value.
   *  @return  A newly constructed DoubleComplex initialized to
   *           z to the power x.
   */
  public static DoubleComplex pow(DoubleComplex z, double x) {
    double  absz = abs(z);
    DoubleComplex result = new DoubleComplex();
    
    if (absz == 0.0) {
      result = z;
    } else {
      double a = argument(z);
      double e = Math.pow(absz, x);
      result.re = e*Math.cos(x*a);
      result.im = e*Math.sin(x*a);
    }
    return result;
  }

  /** 
   *  Returns the inverse hyperbolic sine (arc sinh) of a DoubleComplex,
   *  with a branch cuts outside the interval [-i,i].
   *  @param  z  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to
   *      inverse (arc) hyperbolic sine of the argument.
   *      Its imaginary part is in the interval [-i*pi/2,i*pi/2].
   */
  public static DoubleComplex asinh(DoubleComplex z) {
    // asinh(z) = i*asin(-i*z)
    DoubleComplex miz = new DoubleComplex(z.im,-z.re); 
    DoubleComplex result = asin(miz);
    double rx = result.im;
    result.im = result.re;
    result.re = -rx;
    return result;
  }
  
  /** 
   *  Returns the inverse hyperbolic cosine (arc cosh) of a DoubleComplex,
   *  with a branch cut at values less than one along the real axis.
   *  @param  z  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized to
   *      inverse (arc) hyperbolic cosine of the argument.
   *      The real part of the result is non-negative and its
   *      imaginary part is in the interval [-i*pi,i*pi].
   */
  public static DoubleComplex acosh(DoubleComplex z) {
    DoubleComplex result = acos(z);
    double rx = -result.im;
    result.im = result.re;
    result.re = rx;
    if (result.re < 0 || isNegZero(result.re)) {
      result.re = -result.re;
      result.im = -result.im;    
    }
    return result;
  }


  /**
   *  Returns true is x is a negative zero.
   */
  private static boolean isNegZero(double x) {
    return (java.lang.Double.doubleToLongBits(x) == negZeroBits);
  }

  /** 
   *  Returns the inverse hyperbolic tangent (arc tanh) of a DoubleComplex,
   *  with a branch cuts outside the interval [-1,1] on the real axis.
   *  @param  z  A DoubleComplex object.
   *  @return  A newly constructed DoubleComplex initialized to
   *      inverse (arc) hyperbolic tangent of the argument.
   *      The imaginary part of the result is in the interval
   *      [-i*pi/2,i*pi/2].
   */
  public static DoubleComplex atanh(DoubleComplex z) {
    // atanh(z) = i*atan(-i*z)
    DoubleComplex miz = new DoubleComplex(z.im,-z.re); 
    DoubleComplex result = atan(miz);
    double rx = result.im;
    result.im = result.re;
    result.re = -rx;
    return result;

  }
  
  /** 
   *  Returns the DoubleComplex x raised to the DoubleComplex y power. 
   *  @param  x  A DoubleComplex object.
   *  @param  y  A DoubleComplex object.
   *  @return A newly constructed DoubleComplex initialized
   *      to x<SUP><FONT SIZE="1">y</FONT></SUP><FONT SIZE="3">.
   */
  public static DoubleComplex pow(DoubleComplex x, DoubleComplex y) {
    return exp(times(y,log(x)));
  }

  /** 
   *  Returns a String representation for the specified DoubleComplex. 
   *  @return A String representation for this object.
   */
  public String toString() {
    if (im == 0.0)
      return String.valueOf(re);

    if (re == 0.0)
      return String.valueOf(im) + suffix;

    String sign = (im < 0.0) ? "" : "+";
    return (String.valueOf(re) + sign + String.valueOf(im) + suffix);
  }

  /** 
   *  Parses a string into a DoubleComplex.
   *  @param  s  The string to be parsed.
   *  @return A newly constructed DoubleComplex initialized to the
   *          value represented by the string argument.
   *  @exception NumberFormatException If the string does not contain
   *             a parsable DoubleComplex number.
   *  @exception NullPointerException  If the input argument is null.
   */
  public static DoubleComplex valueOf(String s) throws NumberFormatException {
    String  input = s.trim();
    int    iBeginNumber = 0;
    DoubleComplex z = new DoubleComplex();
    int    state = 0;
    int    sign = 1;
    boolean  haveRealPart = false;

    /*
     * state values
     *  0  Initial State
     *  1  After Initial Sign
     *  2  In integer part
     *  3  In fractional part
     *  4  In exponential part (after 'e' but fore sign or digits)
     *  5  In exponential digits
     */
    for (int k = 0;  k < input.length();  k++) {
      
      char ch = input.charAt(k);

      switch (ch) {

      case '0': case '1': case '2': case '3': case '4':
      case '5': case '6': case '7': case '8': case '9':
        if (state == 0  ||  state == 1) {
          state = 2;
        } else if (state == 4) {
          state = 5;
        }
        break;

      case '-':
      case '+':
        sign = ((ch=='+') ? 1 : -1);
        if (state == 0) {
          state = 1;
        } else if (state == 4) {
          state = 5;
        } else {
          if (!haveRealPart) {
            // have the real part of the number
            z.re = java.lang.Double.valueOf(input.substring(
              iBeginNumber,k)).doubleValue();
            haveRealPart = true;
            // perpare to part the imaginary part
            iBeginNumber = k;
            state = 1;
          } else {
            throw new NumberFormatException(input);
          }
        }
        break;

      case '.':
        if (state == 0  ||  state == 1  ||  state == 2)
          state = 3;
        else
          throw new NumberFormatException(input);
        break;
   
      case 'i': case 'I':
      case 'j': case 'J':
        if (k+1 != input.length()) {
          throw new NumberFormatException(input);
        } else if (state == 0  ||  state == 1) {
          z.im = sign;
          return z;
        } else if (state == 2  ||  state == 3  ||  state == 5) {
          z.im = java.lang.Double.valueOf(
            input.substring(iBeginNumber,k)).doubleValue();
          return z;
        } else {
          throw new NumberFormatException(input);
        }
          

        case 'e': case 'E': case 'd': case 'D':
        if (state == 2  ||  state == 3) {
          state = 4;
        } else {
          throw new NumberFormatException(input);
        }
        break;

        default:
          throw new NumberFormatException(input);
      }
      
    }

    if (!haveRealPart) {
      z.re = java.lang.Double.valueOf(input).doubleValue();
      return z;
    } else {
      throw new NumberFormatException(input);
    }
  }

  /**
   * This is the holder inner class for inout and out arguments for
   * type <code>DoubleComplex</code>.
   */
  public static class Holder {
    private SIDL.DoubleComplex d_obj;

    /**
     * Create a holder class with a null holdee object.
     */
    public Holder() {
      d_obj = null;
    }

    /**
     * Create a holder with the specified object.
     */
    public Holder(SIDL.DoubleComplex obj) {
      d_obj = obj;
    }

    /**
     * Set the value of the holdee object.
     */
    public void set(SIDL.DoubleComplex obj) {
      d_obj = obj;
    }

    /**
     * Get the value of the holdee object.
     */
    public SIDL.DoubleComplex get() {
      return d_obj;
    }
  }

  /**
   * Define a one dimensional array of type <code>SIDL.DoubleComplex</code>
   * for the SIDL Java run-time.  Many of these methods will throw
   * array index exceptions if the specified indices are out of bounds.
   */
  public static class Array extends gov.llnl.sidl.BaseArray {
    /*
     * Register all native JNI routines for this class.
     */
    static {
      gov.llnl.sidl.BaseClass._registerNatives("SIDL.DoubleComplex");
    }

    /**
     * Construct an empty array object.  This array object must be allocated
     * with <code>realllocate</code> before any actions are performed on the
     * array data.
     */
    public Array() {
      super();
    }

    /**
     * Create an array using an IOR array pointer.  The pointer value
     * may be zero (representing null).
     */
    protected Array(long array, boolean owner) {
      super(array, owner);
    }

    /**
     * Create an array with the specified lower and upper bounds.  The
     * upper bounds are inclusive.  An array out of bounds exception is
     * thrown if the array bounds or dimension are invalid.
     */
    public Array(int dim, int[] lower, int[] upper) {
      super();
      reallocate(dim, lower, upper);
    }

    /**
     * Native routine to get the dimension of the current array.  This
     * routine assumes that the array has already been initialized.  If
     * the array has not been initialized, then horrible things may happen.
     */
    public native int _dim();

    /**
     * Native routine to fetch the specified lower bound of the array.  The
     * specified array dimension must be between zero and the array dimension
     * minus one.  Invalid values will have unpredictable (but almost certainly
     * bad) results.
     */
    public native int _lower(int dim);

    /**
     * Native routine to fetch the specified upper bound of the array.  The
     * specified array dimension must be between zero and the array dimension
     * minus one.  Invalid values will have unpredictable (but almost certainly
     * bad) results.
     */
    public native int _upper(int dim);

    /**
     * Native routine to fetch the specified value from the array.  The
     * specified array index/indices must be lie between the array lower
     * upper bounds (inclusive).  Invalid indices will have unpredictable
     * (but almost certainly bad) results.
     */
    public native SIDL.DoubleComplex _get(
      int i, int j, int k, int l);

    /**
     * Native routine to set the specified value in the array.  The
     * specified array index/indices must be lie between the array lower
     * upper bounds (inclusive).  Invalid indices will have unpredictable
     * (but almost certainly bad) results.
     */
    public native void _set(
      int i, int j, int k, int l, SIDL.DoubleComplex value);

    /**
     * Native routine to destroy (deallocate) the current array data.
     */
    public native void _destroy();

    /**
     * Native routine to reallocate data in the array.  The specified array
     * dimension and indices must match and be within valid ranges (e.g., the
     * upper bounds must be greater than or equal to lowe rbounds.  Invalid
     * indices will have unpredictable (but almost certainly bad) results.
     * This routine will deallocate the existing array data if it is not null.
     */
    public native void _reallocate(int dim, int[] lower, int[] upper);
  }

  /**
   * Define a one dimensional array of type <code>SIDL.DoubleComplex</code>.
   * This array representation is used for SIDL arrays since it requires
   * no copies to go between Java and SIDL.  Explicit copies may be made
   * of the array by calling the appropriate <code>get</code> and
   * <code>set</code> methods.
   */
  public static class Array1 extends Array {
    /**
     * Create an empty one dimensional array.  The array will need to be
     * initialized before use.
     */
    public Array1() {
      super();
    }

    /**
     * Create a one dimensional array directly using the SIDL pointer
     * and owner flag.  This constructor should only be called by the
     * SIDL runtime.
     */
    protected Array1(long array, boolean owner) {
      super(array, owner);
    }

    /**
     * Create a one dimensional array using the specified lower and upper
     * bounds (where both bounds are inclusive).  This constructor will throw
     * an array bounds out of range exception if the array bounds are invalid.
     */
    public Array1(int l0, int u0) {
      super(1, new int[] { l0 }, new int[] { u0 });
    }

    /**
     * Create a one dimenstional array of the specified size, with the lower
     * index starting at zero.  This constructor will throw an array bounds
     * out of range exception if the array bounds are invalid.
     */
    public Array1(int s0) {
      super(1, new int[] { 0 }, new int[] { s0-1 });
    }

    /**
     * Create a one dimensional array using the specified Java array.  The
     * lower bound(s) of the constructed array will start at zero.  An array
     * index out of range exception will be thrown if the bounds are invalid.
     */
    public Array1(SIDL.DoubleComplex[] array) {
      super();
      set(array);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public SIDL.DoubleComplex _get(int i) {
      return _get(i, 0, 0, 0);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public SIDL.DoubleComplex get(int i) {
      checkBounds(i);
      return _get(i, 0, 0, 0);
    }

    /**
     * Reallocate array data using the specifed lower and upper bounds.  The
     * upper bound is inclusive.  Previous array data will be freed.
     */
    public void reallocate(int l0, int u0) {
      reallocate(1, new int[] { l0 }, new int[] { u0 });
    }

    /**
     * Set the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>set</code> instead.
     */
    public void _set(int i, SIDL.DoubleComplex value) {
      _set(i, 0, 0, 0, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, SIDL.DoubleComplex value) {
      checkBounds(i);
      _set(i, 0, 0, 0, value);
    }

    /**
     * Convert the SIDL array into a Java array.  This method will copy
     * the SIDL array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the SIDL array is
     * empty (null), then a null Java array will be returned.
     */
    public SIDL.DoubleComplex[] get() {
      SIDL.DoubleComplex[] array = null;
      if (!isNull()) {
        checkDimension(1);
        int l0 = _lower(0);
        int u0 = _upper(0);
        array = new SIDL.DoubleComplex[u0-l0+1];
        for (int i = l0; i <= u0; i++) {
          array[i-l0] = _get(i);
        }
      }
      return array;
    }

    /**
     * Set the value of the SIDL array from the Java array.  This method
     * will copy the Java array values into the SIDL array, reallocating
     * the memory of the SIDL array as necessary.  The resulting SIDL array
     * will start with a zero lower bound.  If the Java array is null, then
     * the SIDL array will be null, as well.  Note that multidimensional Java
     * arrays must not be ragged; that is, all sub-arrays in a particular
     * dimension must have the same size.  Otherwise, some data may be missed
     * or this method may throw an array index out of bounds exception.
     */
    public void set(SIDL.DoubleComplex[] array) {
      if (array == null) {
        destroy();
      } else {
        int s0 = array.length-1;
        reallocate(0, s0);
        for (int i = 0; i <= s0; i++) {
          _set(i, array[i]);
        }
      }
    }
  }

  /**
   * Define a two dimensional array of type <code>SIDL.DoubleComplex</code>.
   * This array representation is used for SIDL arrays since it requires
   * no copies to go between Java and SIDL.  Explicit copies may be made
   * of the array by calling the appropriate <code>get</code> and
   * <code>set</code> methods.
   */
  public static class Array2 extends Array {
    /**
     * Create an empty two dimensional array.  The array will need
     * to be initialized before use.
     */
    public Array2() {
      super();
    }

    /**
     * Create a two dimensional array directly using the SIDL pointer
     * and owner flag.  This constructor should only be called by the
     * SIDL runtime.
     */
    protected Array2(long array, boolean owner) {
      super(array, owner);
    }

    /**
     * Create a two dimensional array using the specified lower and upper
     * bounds (where both bounds are inclusive).  This constructor will throw
     * an array bounds out of range exception if the array bounds are invalid.
     */
    public Array2(int l0, int l1, int u0, int u1) {
      super(2, new int[] { l0, l1 }, new int[] { u0, u1 });
    }

    /**
     * Create a two dimenstional array of the specified size, with the lower
     * index starting at zero.  This constructor will throw an array bounds
     * out of range exception if the array bounds are invalid.
     */
    public Array2(int s0, int s1) {
      super(2, new int[] { 0, 0 }, new int[] { s0-1, s1-1 });
    }

    /**
     * Create a two dimensional array using the specified Java array.  The
     * lower bound(s) of the constructed array will start at zero.  An array
     * index out of range exception will be thrown if the bounds are invalid.
     */
    public Array2(SIDL.DoubleComplex[][] array) {
      super();
      set(array);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public SIDL.DoubleComplex _get(int i, int j) {
      return _get(i, j, 0, 0);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public SIDL.DoubleComplex get(int i, int j) {
      checkBounds(i, j);
      return _get(i, j, 0, 0);
    }

    /**
     * Reallocate array data using the specifed lower and upper bounds.  The
     * upper bound is inclusive.  Previous array data will be freed.
     */
    public void reallocate(int l0, int l1, int u0, int u1) {
      reallocate(2, new int[] { l0, l1 }, new int[] { u0, u1 });
    }

    /**
     * Set the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>set</code> instead.
     */
    public void _set(int i, int j, SIDL.DoubleComplex value) {
      _set(i, j, 0, 0, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, int j, SIDL.DoubleComplex value) {
      checkBounds(i, j);
      _set(i, j, 0, 0, value);
    }

    /**
     * Convert the SIDL array into a Java array.  This method will copy
     * the SIDL array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the SIDL array is
     * empty (null), then a null Java array will be returned.
     */
    public SIDL.DoubleComplex[][] get() {
      SIDL.DoubleComplex[][] array = null;
      if (!isNull()) {
        checkDimension(2);
        int l0 = _lower(0);
        int u0 = _upper(0);
        int l1 = _lower(1);
        int u1 = _upper(1);
        array = new SIDL.DoubleComplex[u0-l0+1][];
        for (int i = l0; i <= u0; i++) {
          array[i] = new SIDL.DoubleComplex[u1-l1+1];
          for (int j = l1; j <= u1; j++) {
            array[i-l0][j-l1] = _get(i, j);
          }
        }
      }
      return array;
    }

    /**
     * Set the value of the SIDL array from the Java array.  This method
     * will copy the Java array values into the SIDL array, reallocating
     * the memory of the SIDL array as necessary.  The resulting SIDL array
     * will start with a zero lower bound.  If the Java array is null, then
     * the SIDL array will be null, as well.  Note that multidimensional Java
     * arrays must not be ragged; that is, all sub-arrays in a particular
     * dimension must have the same size.  Otherwise, some data may be missed
     * or this method may throw an array index out of bounds exception.
     */
    public void set(SIDL.DoubleComplex[][] array) {
      if (array == null) {
        destroy();
      } else {
        int s0 = array.length-1;
        int s1 = array[0].length-1;
        reallocate(0, 0, s0, s1);
        for (int i = 0; i <= s0; i++) {
          for (int j = 0; j <= s1; j++) {
            _set(i, j, array[i][j]);
          }
        }
      }
    }
  }

  /**
   * Define a three dimensional array of type <code>SIDL.DoubleComplex</code>.
   * This array representation is used for SIDL arrays since it requires
   * no copies to go between Java and SIDL.  Explicit copies may be made
   * of the array by calling the appropriate <code>get</code> and
   * <code>set</code> methods.
   */
  public static class Array3 extends Array {
    /**
     * Create an empty three dimensional array.  The array will need
     * to be initialized before use.
     */
    public Array3() {
      super();
    }

    /**
     * Create a three dimensional array directly using the SIDL pointer
     * and owner flag.  This constructor should only be called by the
     * SIDL runtime.
     */
    protected Array3(long array, boolean owner) {
      super(array, owner);
    }

    /**
     * Create a three dimensional array using the specified lower and upper
     * bounds (where both bounds are inclusive).  This constructor will throw
     * an array bounds out of range exception if the array bounds are invalid.
     */
    public Array3(int l0, int l1, int l2, int u0, int u1, int u2) {
      super(3, new int[] { l0, l1, l2 }, new int[] { u0, u1, u2 });
    }

    /**
     * Create a three dimenstional array of the specified size, with the lower
     * index starting at zero.  This constructor will throw an array bounds out
     * of range exception if the array bounds are invalid.
     */
    public Array3(int s0, int s1, int s2) {
      super(3, new int[] { 0, 0, 0 }, new int[] { s0-1, s1-1, s2-1 });
    }

    /**
     * Create a three dimensional array using the specified Java array.  The
     * lower bound(s) of the constructed array will start at zero.  An array
     * index out of range exception will be thrown if the bounds are invalid.
     */
    public Array3(SIDL.DoubleComplex[][][] array) {
      super();
      set(array);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public SIDL.DoubleComplex _get(int i, int j, int k) {
      return _get(i, j, k, 0);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public SIDL.DoubleComplex get(int i, int j, int k) {
      checkBounds(i, j, k);
      return _get(i, j, k, 0);
    }

    /**
     * Reallocate array data using the specifed lower and upper bounds.  The
     * upper bound is inclusive.  Previous array data will be freed.
     */
    public void reallocate(int l0, int l1, int l2, int u0, int u1, int u2) {
      reallocate(3, new int[] { l0, l1, l2 }, new int[] { u0, u1, u2 });
    }

    /**
     * Set the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>set</code> instead.
     */
    public void _set(int i, int j, int k, SIDL.DoubleComplex value) {
      _set(i, j, k, 0, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, int j, int k, SIDL.DoubleComplex value) {
      checkBounds(i, j, k);
      _set(i, j, k, 0, value);
    }

    /**
     * Convert the SIDL array into a Java array.  This method will copy
     * the SIDL array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the SIDL array is
     * empty (null), then a null Java array will be returned.
     */
    public SIDL.DoubleComplex[][][] get() {
      SIDL.DoubleComplex[][][] array = null;
      if (!isNull()) {
        checkDimension(3);
        int l0 = _lower(0);
        int u0 = _upper(0);
        int l1 = _lower(1);
        int u1 = _upper(1);
        int l2 = _lower(2);
        int u2 = _upper(2);
        array = new SIDL.DoubleComplex[u0-l0+1][][];
        for (int i = l0; i <= u0; i++) {
          array[i] = new SIDL.DoubleComplex[u1-l1+1][];
          for (int j = l1; j <= u1; j++) {
            array[i][j] = new SIDL.DoubleComplex[u2-l2+1];
            for (int k = l2; k <= u2; k++) {
              array[i-l0][j-l1][k-l2] = _get(i, j, k);
            }
          }
        }
      }
      return array;
    }

    /**
     * Set the value of the SIDL array from the Java array.  This method
     * will copy the Java array values into the SIDL array, reallocating
     * the memory of the SIDL array as necessary.  The resulting SIDL array
     * will start with a zero lower bound.  If the Java array is null, then
     * the SIDL array will be null, as well.  Note that multidimensional Java
     * arrays must not be ragged; that is, all sub-arrays in a particular
     * dimension must have the same size.  Otherwise, some data may be missed
     * or this method may throw an array index out of bounds exception.
     */
    public void set(SIDL.DoubleComplex[][][] array) {
      if (array == null) {
        destroy();
      } else {
        int s0 = array.length-1;
        int s1 = array[0].length-1;
        int s2 = array[0][0].length-1;
        reallocate(0, 0, 0, s0, s1, s2);
        for (int i = 0; i <= s0; i++) {
          for (int j = 0; j <= s1; j++) {
            for (int k = 0; k <= s1; k++) {
              _set(i, j, k, array[i][j][k]);
            }
          }
        }
      }
    }
  }

  /**
   * Define a four dimensional array of type <code>SIDL.DoubleComplex</code>.
   * This array representation is used for SIDL arrays since it requires
   * no copies to go between Java and SIDL.  Explicit copies may be made
   * of the array by calling the appropriate <code>get</code> and
   * <code>set</code> methods.
   */
  public static class Array4 extends Array {
    /**
     * Create an empty four dimensional array.  The array will need to be
     * initialized before use.
     */
    public Array4() {
      super();
    }

    /**
     * Create a four dimensional array directly using the SIDL pointer
     * and owner flag.  This constructor should only be called by the
     * SIDL runtime.
     */
    protected Array4(long array, boolean owner) {
      super(array, owner);
    }

    /**
     * Create a four dimensional array using the specified lower and upper
     * bounds (where both bounds are inclusive).  This constructor will throw
     * an array bounds out of range exception if the array bounds are invalid.
     */
    public Array4(int l0, int l1, int l2, int l3,
                  int u0, int u1, int u2, int u3) {
      super(4, new int[] { l0, l1, l2, l3 }, new int[] { u0, u1, u2, u3 });
    }

    /**
     * Create a four dimenstional array of the specified size, with the lower
     * index starting at zero.  This constructor will throw an array bounds out
     * of range exception if the array bounds are invalid.
     */
    public Array4(int s0, int s1, int s2, int s3) {
      super(4, new int[] { 0, 0, 0, 0 }, new int[] { s0-1, s1-1, s2-1 });
    }

    /**
     * Create a four dimensional array using the specified Java array.  The
     * lower bound(s) of the constructed array will start at zero.  An array
     * index out of range exception will be thrown if the bounds are invalid.
     */
    public Array4(SIDL.DoubleComplex[][][][] array) {
      super();
      set(array);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public SIDL.DoubleComplex _get(int i, int j, int k, int l) {
      return _get(i, j, k, l);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public SIDL.DoubleComplex get(int i, int j, int k, int l) {
      checkBounds(i, j, k, l);
      return _get(i, j, k, l);
    }

    /**
     * Reallocate array data using the specifed lower and upper bounds.  The
     * upper bound is inclusive.  Previous array data will be freed.
     */
    public void reallocate(int l0, int l1, int l2, int l3,
                           int u0, int u1, int u2, int u3) {
      reallocate(4, new int[] { l0, l1, l2, l3 }, new int[] { u0, u1, u2, l3 });
    }

    /**
     * Set the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>set</code> instead.
     */
    public void _set(int i, int j, int k, int l, SIDL.DoubleComplex value) {
      _set(i, j, k, l, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, int j, int k, int l, SIDL.DoubleComplex value) {
      checkBounds(i, j, k, l);
      _set(i, j, k, l, value);
    }

    /**
     * Convert the SIDL array into a Java array.  This method will copy
     * the SIDL array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the SIDL array is
     * empty (null), then a null Java array will be returned.
     */
    public SIDL.DoubleComplex[][][][] get() {
      SIDL.DoubleComplex[][][][] array = null;
      if (!isNull()) {
        checkDimension(4);
        int l0 = _lower(0);
        int u0 = _upper(0);
        int l1 = _lower(1);
        int u1 = _upper(1);
        int l2 = _lower(2);
        int u2 = _upper(2);
        int l3 = _lower(3);
        int u3 = _upper(3);
        array = new SIDL.DoubleComplex[u0-l0+1][][][];
        for (int i = l0; i <= u0; i++) {
          array[i] = new SIDL.DoubleComplex[u1-l1+1][][];
          for (int j = l1; j <= u1; j++) {
            array[i][j] = new SIDL.DoubleComplex[u2-l2+1][];
            for (int k = l2; k <= u2; k++) {
              array[i][j][k] = new SIDL.DoubleComplex[u3-l3+1];
              for (int l = l3; l <= u3; l++) {
                array[i-l0][j-l1][k-l2][l-l3] = _get(i, j, k, l);
              }
            }
          }
        }
      }
      return array;
    }

    /**
     * Set the value of the SIDL array from the Java array.  This method
     * will copy the Java array values into the SIDL array, reallocating
     * the memory of the SIDL array as necessary.  The resulting SIDL array
     * will start with a zero lower bound.  If the Java array is null, then
     * the SIDL array will be null, as well.  Note that multidimensional Java
     * arrays must not be ragged; that is, all sub-arrays in a particular
     * dimension must have the same size.  Otherwise, some data may be missed
     * or this method may throw an array index out of bounds exception.
     */
    public void set(SIDL.DoubleComplex[][][][] array) {
      if (array == null) {
        destroy();
      } else {
        int s0 = array.length-1;
        int s1 = array[0].length-1;
        int s2 = array[0][0].length-1;
        int s3 = array[0][0][0].length-1;
        reallocate(0, 0, 0, 0, s0, s1, s2, s3);
        for (int i = 0; i <= s0; i++) {
          for (int j = 0; j <= s1; j++) {
            for (int k = 0; k <= s1; k++) {
              for (int l = 0; l <= s2; l++) {
                _set(i, j, k, l, array[i][j][k][l]);
              }
            }
          }
        }
      }
    }
  }
}
