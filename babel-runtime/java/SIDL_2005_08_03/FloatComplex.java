/*
 * -------------------------------------------------------------------------
 * $Id$
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
 * FloatComplex.  Holder and array inner classes have been added, as well.
 */

package SIDL;
 
import SIDL.Sfun;
import java.lang.Float;
import java.lang.String;

/**
 * This class implements complex numbers. It provides the basic operations
 * (addition, subtraction, multiplication, division) as well as a set of
 * complex functions.
 *
 * The binary operations have the form, where op is <code>plus</code>,
 * <code>minus</code>, <code>times</code> or <code>over</code>.
 * <pre>
 * public static FloatComplex op(FloatComplex x, FloatComplex y)   // x op y
 * public static FloatComplex op(FloatComplex x, float y)    // x op y
 * public static FloatComplex op(float x, FloatComplex y)    // x op y
 * public FloatComplex op(FloatComplex y)                     // this op y
 * public FloatComplex op(float y)                      // this op y
 * public FloatComplex opReverse(float x)               // x op this
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
public class FloatComplex implements java.io.Serializable, Cloneable {
  /**  
   *  @serial Real part of the FloatComplex.
   */
  private float re;

  /**
   *  @serial Imaginary part of the FloatComplex.
   */
  private float im;

  /**
   *  Serialization ID
   */
  static final long serialVersionUID = -633126172485117692L;

  /**
   *  String used in converting FloatComplex to String.
   *  Default is "i", but sometimes "j" is desired.
   *  Note that this is set for the class, not for
   *  a particular instance of a FloatComplex.
   */
  public static String suffix = "i";

  private final static float PI = (float) Math.PI;

  private final static int negZeroBits =
    Float.floatToIntBits(1.0f/Float.NEGATIVE_INFINITY);

  /** 
   *  Constructs a FloatComplex equal to the argument.
   *  @param  z  A FloatComplex object
   *      If z is null then a NullPointerException is thrown.
   */
  public FloatComplex(FloatComplex z) {
    re = z.re;
    im = z.im;
  }

  /** 
   *  Constructs a FloatComplex with real and imaginary parts given
   *  by the input arguments.
   *  @param re A float value equal to the real part of the
   *            FloatComplex object.
   *  @param im A float value equal to the imaginary part of
   *            the FloatComplex object.
   */
  public FloatComplex(float re, float im) {
    this.re = re;
    this.im = im;
  }

  /** 
   *  Constructs a FloatComplex with a zero imaginary part. 
   *  @param re A float value equal to the real part of FloatComplex object.
   */
  public FloatComplex(float re) {
    this.re = re;
    this.im = 0.0f;
  }

  /**
   *  Constructs a FloatComplex equal to zero.
   */
  public FloatComplex() {
    re = 0.0f;
    im = 0.0f;
  }

  /** 
   *  Tests if this is a complex Not-a-Number (NaN) value. 
   *  @return  True if either component of the FloatComplex object is NaN;
   *  false, otherwise. 
   */
  private boolean isNaN() {
    return (Float.isNaN(re) || Float.isNaN(im));
  }
  
  /** 
   *  Compares with another FloatComplex. 
   *  <p><em>Note: To be useful in hashtables this method
   *  considers two NaN float values to be equal. This
   *  is not according to IEEE specification.</em>
   *  @param  z  A FloatComplex object.
   *  @return True if the real and imaginary parts of this object
   *      are equal to their counterparts in the argument; false, otherwise.
   */
  public boolean equals(FloatComplex z) {
    if (isNaN() && z.isNaN()) {
      return true;
    } else {
      return (re == z.re  &&  im == z.im);
    }
  }

  /**
   *  Compares this object against the specified object.
   *  <p><em>Note: To be useful in hashtables this method
   *  considers two NaN float values to be equal. This
   *  is not according to IEEE specification</em>
   *  @param  obj  The object to compare with.
   *  @return True if the objects are the same; false otherwise.
   */
  public boolean equals(Object obj) {
    if (obj == null) {
      return false;
    } else if (obj instanceof FloatComplex) {
      return equals((FloatComplex)obj);
    } else {
      return false;
    }
  }

  /**
   *  Returns a hashcode for this FloatComplex.
   *  @return  A hash code value for this object. 
   */
  public int hashCode() {
    int re_bits = Float.floatToIntBits(re);
    int im_bits = Float.floatToIntBits(im);
    return re_bits^im_bits;
  }

  /**
   * Set the real and imaginary parts of the FloatComplex object.
   */
  public void set(float real, float imag) {
    re = real;
    im = imag;
  }

  /** 
   *  Returns the real part of a FloatComplex object. 
   *  @return  The real part of z.
   */
  public float real() {
    return re;
  }

  /** 
   *  Returns the imaginary part of a FloatComplex object. 
   *  @param  z  A FloatComplex object.
   *  @return  The imaginary part of z.
   */
  public float imag() {
    return im;
  }
  
  /** 
   *  Returns the real part of a FloatComplex object. 
   *  @param  z  A FloatComplex object.
   *  @return  The real part of z.
   */
  public static float real(FloatComplex z) {
    return z.re;
  }

  /** 
   *  Returns the imaginary part of a FloatComplex object. 
   *  @param  z  A FloatComplex object.
   *  @return  The imaginary part of z.
   */
  public static float imag(FloatComplex z) {
    return z.im;
  }

  /** 
   *  Returns the negative of a FloatComplex object, -z. 
   *  @param  z  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to
   *      the negative of the argument.
   */
  public static FloatComplex negative(FloatComplex z) {
    return new FloatComplex(-z.re, -z.im);
  }
  
  /** 
   *  Returns the complex conjugate of a FloatComplex object.
   *  @param  z  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to
   *          complex conjugate of z.
   */
  public static FloatComplex conjugate(FloatComplex z) {
    return new FloatComplex(z.re, -z.im);
  }
  
  /** 
   *  Returns the sum of two FloatComplex objects, x+y.
   *  @param  x  A FloatComplex object.
   *  @param  y  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to x+y.
   */
  public static FloatComplex plus(FloatComplex x, FloatComplex y) {
    return new FloatComplex(x.re+y.re, x.im+y.im);
  }

  /** 
   *  Returns the sum of a FloatComplex and a float, x+y. 
   *  @param  x  A FloatComplex object.
   *  @param  y  A float value.
   *  @return A newly constructed FloatComplex initialized to x+y.
   */
  public static FloatComplex plus(FloatComplex x, float y) {
    return new FloatComplex(x.re+y, x.im);
  }

  /** 
   *  Returns the sum of a float and a FloatComplex, x+y. 
   *  @param  x  A float value.
   *  @param  y  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to x+y.
   */
  public static FloatComplex plus(float x, FloatComplex y) {
    return new FloatComplex(x+y.re, y.im);
  }

  /** 
   *  Returns the sum of this FloatComplex and another FloatComplex, this+y. 
   *  @param  y  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to this+y.
   */
  public FloatComplex plus(FloatComplex y) {
    return new FloatComplex(re+y.re, im+y.im);
  }

  /** 
   *  Returns the sum of this FloatComplex a float, this+y. 
   *  @param  y  A float value.
   *  @return A newly constructed FloatComplex initialized to this+y.
   */
  public FloatComplex plus(float y) {
    return new FloatComplex(re+y, im);
  }
  
  /** 
   *  Returns the sum of this FloatComplex and a float, x+this. 
   *  @param  x  A float value.
   *  @return A newly constructed FloatComplex initialized to x+this.
   */
  public FloatComplex plusReverse(float x) {
    return new FloatComplex(re+x, im);
  }

  /** 
   *  Returns the difference of two FloatComplex objects, x-y.
   *  @param  x  A FloatComplex object.
   *  @param  y  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to x-y.
   */
  public static FloatComplex minus(FloatComplex x, FloatComplex y) {
    return new FloatComplex(x.re-y.re, x.im-y.im);
  }

  /** 
   *  Returns the difference of a FloatComplex object and a float, x-y. 
   *  @param  x  A FloatComplex object.
   *  @param  y  A float value.
   *  @return A newly constructed FloatComplex initialized to x-y.
   */
  public static FloatComplex minus(FloatComplex x, float y) {
    return new FloatComplex(x.re-y, x.im);
  }
  
  /** 
   *  Returns the difference of a float and a FloatComplex object, x-y. 
   *  @param  x  A float value.
   *  @param  y  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to x-y..
   */
  public static FloatComplex minus(float x, FloatComplex y) {
    return new FloatComplex(x-y.re, -y.im);
  }

  /** 
   *  Returns the difference of this FloatComplex object and
   *  another FloatComplex object, this-y. 
   *  @param  y  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to this-y.
   */
  public FloatComplex minus(FloatComplex y) {
    return new FloatComplex(re-y.re, im-y.im);
  }

  /** 
   *  Subtracts a float from this FloatComplex and returns the difference,
   *  this-y.
   *  @param  y  A float value.
   *  @return A newly constructed FloatComplex initialized to this-y.
   */
  public FloatComplex minus(float y) {
    return new FloatComplex(re-y, im);
  }

  /** 
   *  Returns the difference of this FloatComplex object and a float, this-y.
   *  @param  y  A float value.
   *  @return A newly constructed FloatComplex initialized to x-this.
   */
  public FloatComplex minusReverse(float x) {
    return new FloatComplex(x-re, -im);
  }

  /** 
   *  Returns the product of two FloatComplex objects, x*y. 
   *  @param  x  A FloatComplex object.
   *  @param  y  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to x*y.
   */
  public static FloatComplex times(FloatComplex x, FloatComplex y) {
    FloatComplex t = new FloatComplex( x.re*y.re-x.im*y.im,
                                       x.re*y.im+x.im*y.re);
    if (Float.isNaN(t.re) && Float.isNaN(t.im)) timesNaN(x, y, t);
    return t;
  }

  /*
   *  Returns sign(b)*|a|.
   */
  private static float copysign(float a, float b) {
    float abs = Math.abs(a);
    return ((b < 0) ? -abs : abs);
  }

  /**
   *  Recovers infinities when computed x*y = NaN+i*NaN.
   *  This code is not part of times(), so that times
   *  could be inlined by an optimizing compiler.
   *  <p>
   *  This algorithm is adapted from the C9x Annex G:
   *  "IEC 559-compatible complex arithmetic."
   *  @param  x  First FloatComplex operand.
   *  @param  y  Second FloatComplex operand.
   *  @param  t  The product x*y, computed without regard to NaN.
   *        The real and/or the imaginary part of t is
   *        expected to be NaN.
   *  @return  The corrected product of x*y.
   */
  private static void timesNaN(FloatComplex x,
                               FloatComplex y,
                               FloatComplex t) {
    boolean  recalc = false;
    float  a = x.re;
    float  b = x.im;
    float  c = y.re;
    float  d = y.im;

    if (Float.isInfinite(a) || Float.isInfinite(b)) {
      // x is infinite
      a = copysign(Float.isInfinite(a) ? 1.0f : 0.0f, a);
      b = copysign(Float.isInfinite(b) ? 1.0f : 0.0f, b);
      if (Float.isNaN(c))  c = copysign(0.0f, c);
      if (Float.isNaN(d))  d = copysign(0.0f, d);
      recalc = true;
    }

    if (Float.isInfinite(c) || Float.isInfinite(d)) {
      // x is infinite
      a = copysign(Float.isInfinite(c) ? 1.0f : 0.0f, c);
      b = copysign(Float.isInfinite(d) ? 1.0f : 0.0f, d);
      if (Float.isNaN(a))  a = copysign(0.0f, a);
      if (Float.isNaN(b))  b = copysign(0.0f, b);
      recalc = true;
    }

    if (!recalc) {
      if (Float.isInfinite(a*c) || Float.isInfinite(b*d) ||
        Float.isInfinite(a*d) || Float.isInfinite(b*c)) {
        // Change all NaNs to 0
        if (Float.isNaN(a))  a = copysign(0.0f, a);
        if (Float.isNaN(b))  b = copysign(0.0f, b);
        if (Float.isNaN(c))  c = copysign(0.0f, c);
        if (Float.isNaN(d))  d = copysign(0.0f, d);
        recalc = true;
      }
    }

    if (recalc) {
      t.re = Float.POSITIVE_INFINITY * (a*c - b*d);
      t.im = Float.POSITIVE_INFINITY * (a*d + b*c);
    }
  }

  /** 
   *  Returns the product of a FloatComplex object and a float, x*y. 
   *  @param  x  A FloatComplex object.
   *  @param  y  A float value.
   *  @return  A newly constructed FloatComplex initialized to x*y.
   */
  public static FloatComplex times(FloatComplex x, float y) {
    return new FloatComplex(x.re*y, x.im*y);
  }

  /** 
   *  Returns the product of a float and a FloatComplex object, x*y. 
   *  @param  x  A float value.
   *  @param  y  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to x*y.
   */
  public static FloatComplex times(float x, FloatComplex y) {
    return new FloatComplex(x*y.re, x*y.im);
  }

  /** 
   * Returns the product of this FloatComplex object and another
   * FloatComplex object, this*y. 
   * @param  y  A FloatComplex object.
   * @return  A newly constructed FloatComplex initialized to this*y.
   */
  public FloatComplex times(FloatComplex y) {
    return times(this,y);
  }

  /** 
   *  Returns the product of this FloatComplex object and a float, this*y.
   *  @param  y  A float value.
   *  @return A newly constructed FloatComplex initialized to this*y.
   */
  public FloatComplex times(float y) {
    return new FloatComplex(re*y, im*y);
  }

  /** 
   *  Returns the product of a float and this FloatComplex, x*this. 
   *  @param  y  A float value.
   *  @return A newly constructed FloatComplex initialized to x*this.
   */
  public FloatComplex timesReverse(float x) {
    return new FloatComplex(x*re, x*im);
  }

  private static boolean isFinite(float x) {
    return !(Float.isInfinite(x) || Float.isNaN(x));
  }

  /** 
   *  Returns FloatComplex object divided by a FloatComplex object, x/y. 
   *  @param  x  The numerator, a FloatComplex object.
   *  @param  y  The denominator, a FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to x/y.
   */
  public static FloatComplex over(FloatComplex x, FloatComplex y) {
    float  a = x.re;
    float  b = x.im;
    float  c = y.re;
    float  d = y.im;

    float scale = Math.max(Math.abs(c), Math.abs(d));
    boolean isScaleFinite = isFinite(scale);
    if (isScaleFinite) {
      c /= scale;
      d /= scale;
    }

    float den = c*c + d*d;
    FloatComplex z = new FloatComplex((a*c+b*d)/den, (b*c-a*d)/den);
    
    if (isScaleFinite) {
      z.re /= scale;
      z.im /= scale;
    }

    // Recover infinities and zeros computed as NaN+iNaN.
    if (Float.isNaN(z.re) && Float.isNaN(z.im)) {
      if (den == 0.0f  && (!Float.isNaN(a) || !Float.isNaN(b))) {
        float s = copysign(Float.POSITIVE_INFINITY, c);
        z.re = s * a;
        z.im = s * b;
      
      } else if ((Float.isInfinite(a) || Float.isInfinite(b)) &&
        isFinite(c) && isFinite(d)) {
        a = copysign(Float.isInfinite(a) ? 1.0f : 0.0f, a);
        b = copysign(Float.isInfinite(b) ? 1.0f : 0.0f, b);
        z.re = Float.POSITIVE_INFINITY * (a*c + b*d);
        z.im = Float.POSITIVE_INFINITY * (b*c - a*d);
      
      } else if (Float.isInfinite(scale)  &&
        isFinite(a) && isFinite(b)) {
        c = copysign(Float.isInfinite(c) ? 1.0f : 0.0f, c);
        d = copysign(Float.isInfinite(d) ? 1.0f : 0.0f, d);
        z.re = 0.0f * (a*c + b*d);
        z.im = 0.0f * (b*c - a*d);
      }
    }
    return z;
  }

  /** 
   *  Returns FloatComplex object divided by a float, x/y.
   *  @param  x  The numerator, a FloatComplex object.
   *  @param  y  The denominator, a float.
   *  @return A newly constructed FloatComplex initialized to x/y.
   */
  public static FloatComplex over(FloatComplex x, float y) {
    return new FloatComplex(x.re/y, x.im/y);
  }

  /** 
   *  Returns a float divided by a FloatComplex object, x/y. 
   *  @param  x  A float value.
   *  @param  y  The denominator, a FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to x/y.
   */
  public static FloatComplex over(float x, FloatComplex y) {
    return y.overReverse(x);
  }

  /** 
   *  Returns this FloatComplex object divided by another
   *  FloatComplex object, this/y. 
   *  @param  y  The denominator, a FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to x/y.
   */
  public FloatComplex over(FloatComplex y) {
    return over(this, y);
  }

  /** 
   *  Returns this FloatComplex object divided by float, this/y. 
   *  @param  y  The denominator, a float.
   *  @return  A newly constructed FloatComplex initialized to x/y.
   */
  public FloatComplex over(float y) {
    return over(this, y);
  }

  /** 
   *  Returns a float dividied by this FloatComplex object, x/this. 
   *  @param  x  The numerator, a float.
   *  @return A newly constructed FloatComplex initialized to x/this.
   */
  public FloatComplex overReverse(float x) {
    float        den;
    float        t;
    FloatComplex z;

    if (Math.abs(re) > Math.abs(im)) {
      t = im / re;
      den = re + im*t;
      z = new FloatComplex(x/den, -x*t/den);
    } else {
      t = re / im;
      den = im + re*t;
      z = new FloatComplex(x*t/den, -x/den);
    }
    return z;
  }

  /** 
   *  Returns the absolute value (modulus) of a FloatComplex, |z|. 
   *  @param  z  A FloatComplex object.
   *  @return A float value equal to the absolute value of the argument.
   */
  public static float abs(FloatComplex z) {
    float x = Math.abs(z.re);
    float y = Math.abs(z.im);
    
    if (Float.isInfinite(x) || Float.isInfinite(y))
      return Float.POSITIVE_INFINITY;
    
    if (x + y == 0.0f) {
      return 0.0f;
    } else if (x > y) {
      y /= x;
      return x * (float) Math.sqrt(1.0f+y*y);
    } else {
      x /= y;
      return y * (float) Math.sqrt(x*x+1.0f);
    }
  }

  /** 
   *  Returns the argument (phase) of a FloatComplex, in radians,
   *  with a branch cut along the negative real axis.
   *  @param  z  A FloatComplex object.
   *  @return A float value equal to the argument (or phase) of
   *          a FloatComplex.  It is in the interval [-pi,pi].
   */
  public static float argument(FloatComplex z) {
    return (float) Math.atan2(z.im, z.re);
  }
  
  /** 
   *  Returns the square root of a FloatComplex,
   *  with a branch cut along the negative real axis.
   *  @param  z  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized
   *          to square root of z. Its real part is non-negative.
   */
  public static FloatComplex sqrt(FloatComplex z) {
    FloatComplex  result = new FloatComplex();

    if (Float.isInfinite(z.im)) {
      result.re = Float.POSITIVE_INFINITY;
      result.im = z.im;
    } else if (Float.isNaN(z.re)) {
      result.re = result.im = Float.NaN;
    } else if (Float.isNaN(z.im)) {
      if (Float.isInfinite(z.re)) {
        if (z.re > 0) {
          result.re = z.re;
          result.im = z.im;
        } else {
          result.re = z.im;
          result.im = Float.POSITIVE_INFINITY;
        }
      } else {
        result.re = result.im = Float.NaN;
      }
    } else {
      // Numerically correct version of formula 3.7.27
      // in the NBS Hanbook, as suggested by Pete Stewart.
      float t = abs(z);
    
      if (Math.abs(z.re) <= Math.abs(z.im)) {
        // No cancellation in these formulas
        result.re = (float) Math.sqrt(0.5*(t+z.re));
        result.im = (float) Math.sqrt(0.5*(t-z.re));
      } else {
        // Stable computation of the above formulas
        if (z.re > 0) {
          result.re = t + z.re;
          result.im = Math.abs(z.im) * (float) Math.sqrt(0.5/result.re);
          result.re = (float) Math.sqrt(0.5*result.re);
        } else {
          result.im = t - z.re;
          result.re = Math.abs(z.im) * (float) Math.sqrt(0.5/result.im);
          result.im = (float) Math.sqrt(0.5*result.im);
        }
      }
      if (z.im < 0)
        result.im = -result.im;
    }
    return result;
  }

  /** 
   *  Returns the exponential of a FloatComplex z, exp(z).
   *  @param  z  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to exponential
   *      of the argument. 
   */
  public static FloatComplex exp(FloatComplex z) {
    FloatComplex result = new FloatComplex();
    
    float r = (float) Math.exp(z.re);

    float cosa = (float) Math.cos(z.im);
    float sina = (float) Math.sin(z.im);
    if (Float.isInfinite(z.im) || Float.isNaN(z.im) || Math.abs(cosa)>1) {
      cosa = sina = Float.NaN;
    }

    if (Float.isInfinite(z.re) || Float.isInfinite(r)) {
      if (z.re < 0) {
        r = 0;
        if (Float.isInfinite(z.im)  ||  Float.isNaN(z.im)) {
          cosa = sina = 0;
        } else {
          cosa /= Float.POSITIVE_INFINITY;
          sina /= Float.POSITIVE_INFINITY;
        }
      } else {
        r = z.re;
        if (Float.isNaN(z.im)) cosa = 1;
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
   *  Returns the logarithm of a FloatComplex z,
   *  with a branch cut along the negative real axis.
   *  @param  z  A FloatComplex object.
   *  @return Newly constructed FloatComplex initialized to logarithm of
   *          the argument. Its imaginary part is in the interval [-i*pi,i*pi].
   */
  public static FloatComplex log(FloatComplex z) {
    FloatComplex  result = new FloatComplex();

    if (Float.isNaN(z.re)) {
      result.re = result.im = z.re;
      if (Float.isInfinite(z.im))
        result.re = Float.POSITIVE_INFINITY;
    } else if (Float.isNaN(z.im)) {
      result.re = result.im = z.im;
      if (Float.isInfinite(z.re))
        result.re = Float.POSITIVE_INFINITY;
    } else {
      result.re = (float) Math.log(abs(z));
      result.im = argument(z);
    }
    return result;
  }

  /** 
   *  Returns the sine of a FloatComplex. 
   *  @param  z  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to
   *          sine of the argument.
   */
  public static FloatComplex sin(FloatComplex z) {
    // sin(z) = -i*sinh(i*z)
    FloatComplex iz = new FloatComplex(-z.im,z.re);
    FloatComplex s = sinh(iz);
    float re = s.im;
    s.im = -s.re;
    s.re = re;
    return s;
  }

  /** 
   *  Returns the cosine of a FloatComplex. 
   *  @param  z  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to
   *          cosine of the argument.
   */
  public static FloatComplex cos(FloatComplex z) {
    // cos(z) = cosh(i*z)
    return cosh(new FloatComplex(-z.im,z.re));
  }

  /** 
   *  Returns the tangent of a FloatComplex. 
   *  @param  z  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized
   *      to tangent of the argument.
   */
  public static FloatComplex tan(FloatComplex z) {
    // tan = -i*tanh(i*z)
    FloatComplex iz = new FloatComplex(-z.im,z.re);
    FloatComplex s = tanh(iz);
    float re = s.im;
    s.im = -s.re;
    s.re = re;
    return s;
  }

  /** 
   *  Returns the inverse sine (arc sine) of a FloatComplex,
   *  with branch cuts outside the interval [-1,1] along the
   *  real axis.
   *  @param  z  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to inverse
   *      (arc) sine of the argument. The real part of the
   *      result is in the interval [-pi/2,+pi/2].
   */
  public static FloatComplex asin(FloatComplex z) {
    FloatComplex  result = new FloatComplex();

    float r = abs(z);

    if (Float.isInfinite(r)) {
      boolean infiniteX = Float.isInfinite(z.re);
      boolean infiniteY = Float.isInfinite(z.im);
      if (infiniteX) {
        float  pi2 = 0.5f * PI;
        result.re = (z.re>0 ? pi2 : -pi2);
        if (infiniteY) result.re /= 2;
      } else if (infiniteY) {
        result.re = z.re/Float.POSITIVE_INFINITY;
      }
      if (Float.isNaN(z.im)) {
        result.im = -z.re;
        result.re = z.im;
      } else {
        result.im = z.im*Float.POSITIVE_INFINITY;
      }
      return result;
    } else if (Float.isNaN(r)) {
      result.re = result.im = Float.NaN;
      if (z.re == 0.0f)  result.re = z.re;
    } else if (r < 2.58095e-08) {
      // sqrt(6.0*dmach(3)) = 2.58095e-08
      result.re = z.re;
      result.im = z.im;
    } else if (z.re == 0.0f) {
      result.re = 0.0f;
      result.im = (float) Sfun.asinh((double) z.im);
    } else if (r <= 0.1f) {
      FloatComplex z2 = times(z,z);
      //log(eps)/log(rmax) = 8 where rmax = 0.1
      for (int i = 1;  i <= 8;  i++) {
        float twoi = 2.0f*(8.0f-i) + 1.0f;
        result = times(times(result,z2),twoi/(twoi+1.0f));
        result.re += 1.0f/twoi;
      }
      result = result.times(z);
    } else {
      // A&S 4.4.26
      // asin(z) = -i*log(z+sqrt(1-z)*sqrt(1+z))
      // or, since log(iz) = log(z) +i*pi/2,
      // asin(z) = pi/2 - i*log(z+sqrt(z+1)*sqrt(z-1))
      FloatComplex w = ((z.im < 0.0f) ? negative(z) : z);
      FloatComplex sqzp1 = sqrt(plus(w,1.0f));
      if (sqzp1.im < 0.0f)
        sqzp1 = negative(sqzp1);
      FloatComplex sqzm1 = sqrt(minus(w,1.0f));
      result = log(plus(w,times(sqzp1,sqzm1)));

      float rx = result.re;
      result.re = 0.5f * PI + result.im;
      result.im = -rx;
    }

    if (result.re > 0.5f * PI) {
      result.re = PI - result.re;
      result.im = -result.im;
    }
    if (result.re < -0.5f * PI) {
      result.re = -PI - result.re;
      result.im = -result.im;
    }
    if (z.im < 0.0f) {
      result.re = -result.re;
      result.im = -result.im;
    }
    return result;
  }

  /** 
   *  Returns the inverse cosine (arc cosine) of a FloatComplex,
   *  with branch cuts outside the interval [-1,1] along the
   *  real axis.
   *  @param  z  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to
   *      inverse (arc) cosine of the argument.
   *      The real part of the result is in the interval [0,pi].
   */
  public static FloatComplex acos(FloatComplex z) {
    FloatComplex  result = new FloatComplex();
    float r = abs(z);

    if (Float.isInfinite(z.re) && Float.isNaN(z.im)) {
      result.re = Float.NaN;
      result.im = Float.NEGATIVE_INFINITY;
    } else if (Float.isInfinite(r)) {
      result.re = (float) Math.atan2(Math.abs(z.im),z.re);
      result.im = z.im*Float.NEGATIVE_INFINITY;
    } else if (r == 0.0f) {
      result.re = PI/2.0f;
      result.im = -z.im;
    } else {
      result = minus(PI/2.0f,asin(z));
    }
    return result;
  }

  /** 
   * Returns the inverse tangent (arc tangent) of a FloatComplex,
   * with branch cuts outside the interval [-i,i] along the
   * imaginary axis.
   * @param  z  A FloatComplex object.
   * @return  A newly constructed FloatComplex initialized to
   *      inverse (arc) tangent of the argument.
   *      Its real part is in the interval [-pi/2,pi/2].
   */
  public static FloatComplex atan(FloatComplex z) {
    FloatComplex  result = new FloatComplex();
    float  r = abs(z);

    if (Float.isInfinite(r)) {
      float  pi2 = 0.5f*PI;
      float im = (Float.isNaN(z.im) ? 0 : z.im);
      result.re = (z.re<0 ? -pi2 : pi2);
      result.im = (im<0 ? -1 : 1)/Float.POSITIVE_INFINITY;
      if (Float.isNaN(z.re))  result.re = z.re;
    } else if (Float.isNaN(r)) {
      result.re = result.im = Float.NaN;
      if (z.im == 0)  result.im = z.im;
    } else if (r < 1.82501e-08) {
      // sqrt(3.0*dmach(3)) = 1.82501e-08
      result.re = z.re;
      result.im = z.im;
    } else if (r < 0.1) {
      FloatComplex z2 = times(z,z);
      // -0.4343*log(dmach(3))+1 = 17
      for (int k = 0;  k < 17;  k++) {
        FloatComplex temp = times(z2,result);
        int twoi = 2*(17-k) - 1;
        result.re = 1.0f/twoi - temp.re;
        result.im = -temp.im;
      }
      result = result.times(z);
    } else if (r < 9.0072e+15) {
      // 1.0/dmach(3) = 9.0072e+15
      float r2 = r*r;
      result.re = 0.5f * (float) Math.atan2(2*z.re,1.0-r2);
      result.im = 0.25f * (float) Math.log((r2+2*z.im+1)/(r2-2*z.im+1));
    } else {
      result.re = ((z.re < 0.0f) ? -0.5f*PI : 0.5f*PI);
    }
    return result;
  }

  /** 
   * Returns the hyperbolic sine of a FloatComplex. 
   * @param  z  A FloatComplex object.
   * @return  A newly constructed FloatComplex initialized to hyperbolic
   *      sine of the argument.
   */
  public static FloatComplex sinh(FloatComplex z) {
    float  coshx = (float) Sfun.cosh(z.re);
    float  sinhx = (float) Sfun.sinh(z.re);
    float  cosy  = (float) Math.cos(z.im);
    float  siny  = (float) Math.sin(z.im);
    boolean infiniteX = Float.isInfinite(coshx);
    boolean infiniteY = Float.isInfinite(z.im);
    FloatComplex result;

    if (z.im == 0) {
      result = new FloatComplex((float) Sfun.sinh(z.re));
    } else {
      // A&S 4.5.49
      result = new FloatComplex(sinhx*cosy, coshx*siny);
      if (infiniteY) {
        result.im = Float.NaN;
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
   * Returns the hyperbolic cosh of a FloatComplex. 
   * @param  z  A FloatComplex object.
   * @return  A newly constructed FloatComplex initialized to
   *      the hyperbolic cosine of the argument.
   */
  public static FloatComplex cosh(FloatComplex z) {
    if (z.im == 0) {
      return new FloatComplex((float) Sfun.cosh(z.re));
    }
    
    float  coshx = (float) Sfun.cosh(z.re);
    float  sinhx = (float) Sfun.sinh(z.re);
    float  cosy  = (float) Math.cos(z.im);
    float  siny  = (float) Math.sin(z.im);
    boolean infiniteX = Float.isInfinite(coshx);
    boolean infiniteY = Float.isInfinite(z.im);

    // A&S 4.5.50
    FloatComplex result = new FloatComplex(coshx*cosy, sinhx*siny);
    if (infiniteY)   result.re = Float.NaN;
    if (z.re == 0) {
      result.im = 0;
    } else if (infiniteX) {
      result.re = z.re*cosy;
      result.im = z.re*siny;
      if (z.im == 0)  result.im = 0;
      if (Float.isNaN(z.im)) {
        result.re = z.re;
      } else if (infiniteY) {
        result.re = z.im;
      }
    }
    return result;
  }

  /** 
   * Returns the hyperbolic tanh of a FloatComplex.
   * @param  z  A FloatComplex object.
   * @return  A newly constructed FloatComplex initialized to
   *          the hyperbolic tangent of the argument.
   */
  public static FloatComplex tanh(FloatComplex z) {
    float  sinh2x = (float) Sfun.sinh(2*z.re);
    
    if (z.im == 0.0f) {
      return new FloatComplex((float) Sfun.tanh(z.re));
    } else if (sinh2x == 0.0f) {
      return new FloatComplex(0.0f, (float) Math.tan(z.im));
    }

    float  cosh2x = (float) Sfun.cosh(2.0*z.re);
    float  cos2y  = (float) Math.cos(2.0*z.im);
    float  sin2y  = (float) Math.sin(2.0*z.im);
    boolean infiniteX = Float.isInfinite(cosh2x);

    // Workaround for bug in JDK 1.2beta4
    if (Float.isInfinite(z.im) || Float.isNaN(z.im)) {
      cos2y = sin2y = Float.NaN;  
    }

    if (infiniteX)
      return new FloatComplex(z.re > 0.0f ? 1.0f : -1.0f);

    // A&S 4.5.51
    float den = (cosh2x + cos2y);
    return new FloatComplex(sinh2x/den, sin2y/den);
  }
  
  /** 
   *  Returns the FloatComplex z raised to the x power,
   *  with a branch cut for the first parameter (z) along the
   *  negative real axis.
   *  @param  z  A FloatComplex object.
   *  @param  x  A float value.
   *  @return  A newly constructed FloatComplex initialized to
   *           z to the power x.
   */
  public static FloatComplex pow(FloatComplex z, float x) {
    float  absz = abs(z);
    FloatComplex result = new FloatComplex();
    
    if (absz == 0.0) {
      result = z;
    } else {
      float a = argument(z);
      float e = (float) Math.pow(absz, x);
      result.re = e * ((float) Math.cos(x*a));
      result.im = e * ((float) Math.sin(x*a));
    }
    return result;
  }

  /** 
   *  Returns the inverse hyperbolic sine (arc sinh) of a FloatComplex,
   *  with a branch cuts outside the interval [-i,i].
   *  @param  z  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to
   *      inverse (arc) hyperbolic sine of the argument.
   *      Its imaginary part is in the interval [-i*pi/2,i*pi/2].
   */
  public static FloatComplex asinh(FloatComplex z) {
    // asinh(z) = i*asin(-i*z)
    FloatComplex miz = new FloatComplex(z.im,-z.re); 
    FloatComplex result = asin(miz);
    float rx = result.im;
    result.im = result.re;
    result.re = -rx;
    return result;
  }
  
  /** 
   *  Returns the inverse hyperbolic cosine (arc cosh) of a FloatComplex,
   *  with a branch cut at values less than one along the real axis.
   *  @param  z  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized to
   *      inverse (arc) hyperbolic cosine of the argument.
   *      The real part of the result is non-negative and its
   *      imaginary part is in the interval [-i*pi,i*pi].
   */
  public static FloatComplex acosh(FloatComplex z) {
    FloatComplex result = acos(z);
    float rx = -result.im;
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
  private static boolean isNegZero(float x) {
    return (Float.floatToIntBits(x) == negZeroBits);
  }

  /** 
   *  Returns the inverse hyperbolic tangent (arc tanh) of a FloatComplex,
   *  with a branch cuts outside the interval [-1,1] on the real axis.
   *  @param  z  A FloatComplex object.
   *  @return  A newly constructed FloatComplex initialized to
   *      inverse (arc) hyperbolic tangent of the argument.
   *      The imaginary part of the result is in the interval
   *      [-i*pi/2,i*pi/2].
   */
  public static FloatComplex atanh(FloatComplex z) {
    // atanh(z) = i*atan(-i*z)
    FloatComplex miz = new FloatComplex(z.im,-z.re); 
    FloatComplex result = atan(miz);
    float rx = result.im;
    result.im = result.re;
    result.re = -rx;
    return result;

  }
  
  /** 
   *  Returns the FloatComplex x raised to the FloatComplex y power. 
   *  @param  x  A FloatComplex object.
   *  @param  y  A FloatComplex object.
   *  @return A newly constructed FloatComplex initialized
   *      to x<SUP><FONT SIZE="1">y</FONT></SUP><FONT SIZE="3">.
   */
  public static FloatComplex pow(FloatComplex x, FloatComplex y) {
    return exp(times(y,log(x)));
  }

  /** 
   *  Returns a String representation for the specified FloatComplex. 
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
   *  Parses a string into a FloatComplex.
   *  @param  s  The string to be parsed.
   *  @return A newly constructed FloatComplex initialized to the
   *          value represented by the string argument.
   *  @exception NumberFormatException If the string does not contain
   *             a parsable FloatComplex number.
   *  @exception NullPointerException  If the input argument is null.
   */
  public static FloatComplex valueOf(String s) throws NumberFormatException {
    String  input = s.trim();
    int    iBeginNumber = 0;
    FloatComplex z = new FloatComplex();
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
            z.re = Float.valueOf(input.substring(
              iBeginNumber,k)).floatValue();
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
          z.im = Float.valueOf(input.substring(iBeginNumber,k)).floatValue();
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
      z.re = Float.valueOf(input).floatValue();
      return z;
    } else {
      throw new NumberFormatException(input);
    }
  }

  /**
   * This is the holder inner class for inout and out arguments for
   * type <code>FloatComplex</code>.
   */
  public static class Holder {
    private SIDL.FloatComplex d_obj;

    /**
     * Create a holder class with a null holdee object.
     */
    public Holder() {
      d_obj = null;
    }

    /**
     * Create a holder with the specified object.
     */
    public Holder(SIDL.FloatComplex obj) {
      d_obj = obj;
    }

    /**
     * Set the value of the holdee object.
     */
    public void set(SIDL.FloatComplex obj) {
      d_obj = obj;
    }

    /**
     * Get the value of the holdee object.
     */
    public SIDL.FloatComplex get() {
      return d_obj;
    }
  }

  /**
   * Define a one dimensional array of type <code>SIDL.FloatComplex</code>
   * for the SIDL Java run-time.  Many of these methods will throw
   * array index exceptions if the specified indices are out of bounds.
   */
  public static class Array extends gov.llnl.sidl.BaseArray {
    /*
     * Register all native JNI routines for this class.
     */
    static {
      gov.llnl.sidl.BaseClass._registerNatives("SIDL.FloatComplex");
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
    public native SIDL.FloatComplex _get(
      int i, int j, int k, int l);

    /**
     * Native routine to set the specified value in the array.  The
     * specified array index/indices must be lie between the array lower
     * upper bounds (inclusive).  Invalid indices will have unpredictable
     * (but almost certainly bad) results.
     */
    public native void _set(
      int i, int j, int k, int l, SIDL.FloatComplex value);

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
   * Define a one dimensional array of type <code>SIDL.FloatComplex</code>.
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
    public Array1(SIDL.FloatComplex[] array) {
      super();
      set(array);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public SIDL.FloatComplex _get(int i) {
      return _get(i, 0, 0, 0);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public SIDL.FloatComplex get(int i) {
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
    public void _set(int i, SIDL.FloatComplex value) {
      _set(i, 0, 0, 0, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, SIDL.FloatComplex value) {
      checkBounds(i);
      _set(i, 0, 0, 0, value);
    }

    /**
     * Convert the SIDL array into a Java array.  This method will copy
     * the SIDL array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the SIDL array is
     * empty (null), then a null Java array will be returned.
     */
    public SIDL.FloatComplex[] get() {
      SIDL.FloatComplex[] array = null;
      if (!isNull()) {
        checkDimension(1);
        int l0 = _lower(0);
        int u0 = _upper(0);
        array = new SIDL.FloatComplex[u0-l0+1];
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
    public void set(SIDL.FloatComplex[] array) {
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
   * Define a two dimensional array of type <code>SIDL.FloatComplex</code>.
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
    public Array2(SIDL.FloatComplex[][] array) {
      super();
      set(array);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public SIDL.FloatComplex _get(int i, int j) {
      return _get(i, j, 0, 0);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public SIDL.FloatComplex get(int i, int j) {
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
    public void _set(int i, int j, SIDL.FloatComplex value) {
      _set(i, j, 0, 0, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, int j, SIDL.FloatComplex value) {
      checkBounds(i, j);
      _set(i, j, 0, 0, value);
    }

    /**
     * Convert the SIDL array into a Java array.  This method will copy
     * the SIDL array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the SIDL array is
     * empty (null), then a null Java array will be returned.
     */
    public SIDL.FloatComplex[][] get() {
      SIDL.FloatComplex[][] array = null;
      if (!isNull()) {
        checkDimension(2);
        int l0 = _lower(0);
        int u0 = _upper(0);
        int l1 = _lower(1);
        int u1 = _upper(1);
        array = new SIDL.FloatComplex[u0-l0+1][];
        for (int i = l0; i <= u0; i++) {
          array[i] = new SIDL.FloatComplex[u1-l1+1];
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
    public void set(SIDL.FloatComplex[][] array) {
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
   * Define a three dimensional array of type <code>SIDL.FloatComplex</code>.
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
    public Array3(SIDL.FloatComplex[][][] array) {
      super();
      set(array);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public SIDL.FloatComplex _get(int i, int j, int k) {
      return _get(i, j, k, 0);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public SIDL.FloatComplex get(int i, int j, int k) {
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
    public void _set(int i, int j, int k, SIDL.FloatComplex value) {
      _set(i, j, k, 0, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, int j, int k, SIDL.FloatComplex value) {
      checkBounds(i, j, k);
      _set(i, j, k, 0, value);
    }

    /**
     * Convert the SIDL array into a Java array.  This method will copy
     * the SIDL array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the SIDL array is
     * empty (null), then a null Java array will be returned.
     */
    public SIDL.FloatComplex[][][] get() {
      SIDL.FloatComplex[][][] array = null;
      if (!isNull()) {
        checkDimension(3);
        int l0 = _lower(0);
        int u0 = _upper(0);
        int l1 = _lower(1);
        int u1 = _upper(1);
        int l2 = _lower(2);
        int u2 = _upper(2);
        array = new SIDL.FloatComplex[u0-l0+1][][];
        for (int i = l0; i <= u0; i++) {
          array[i] = new SIDL.FloatComplex[u1-l1+1][];
          for (int j = l1; j <= u1; j++) {
            array[i][j] = new SIDL.FloatComplex[u2-l2+1];
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
    public void set(SIDL.FloatComplex[][][] array) {
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
   * Define a four dimensional array of type <code>SIDL.FloatComplex</code>.
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
    public Array4(SIDL.FloatComplex[][][][] array) {
      super();
      set(array);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public SIDL.FloatComplex _get(int i, int j, int k, int l) {
      return _get(i, j, k, l);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public SIDL.FloatComplex get(int i, int j, int k, int l) {
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
    public void _set(int i, int j, int k, int l, SIDL.FloatComplex value) {
      _set(i, j, k, l, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, int j, int k, int l, SIDL.FloatComplex value) {
      checkBounds(i, j, k, l);
      _set(i, j, k, l, value);
    }

    /**
     * Convert the SIDL array into a Java array.  This method will copy
     * the SIDL array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the SIDL array is
     * empty (null), then a null Java array will be returned.
     */
    public SIDL.FloatComplex[][][][] get() {
      SIDL.FloatComplex[][][][] array = null;
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
        array = new SIDL.FloatComplex[u0-l0+1][][][];
        for (int i = l0; i <= u0; i++) {
          array[i] = new SIDL.FloatComplex[u1-l1+1][][];
          for (int j = l1; j <= u1; j++) {
            array[i][j] = new SIDL.FloatComplex[u2-l2+1][];
            for (int k = l2; k <= u2; k++) {
              array[i][j][k] = new SIDL.FloatComplex[u3-l3+1];
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
    public void set(SIDL.FloatComplex[][][][] array) {
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
