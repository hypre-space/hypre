//
// File:	Float.java
// Package:	SIDL
// Release:	$Name: V1-9-0b $
// Revision:	$Revision: 1.4 $
// Modified:	$Date: 2003/04/07 21:44:19 $
// Description:	holder and array classes for built-in data types
//
// Copyright (c) 2000-2001, The Regents of the University of Calfornia.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the Components Team <components@llnl.gov>
// UCRL-CODE-2002-054
// All rights reserved.
// 
// This file is part of Babel. For more information, see
// http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
// for Our Notice and the LICENSE file for the GNU Lesser General Public
// License.
// 
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License (as published by
// the Free Software Foundation) version 2.1 dated February 1999.
// 
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
// conditions of the GNU Lesser General Public License for more details.
// 
// You should have recieved a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

package SIDL;

/**
 * Class <code>Float</code> contains inner classes that
 * provide holder and array support for standard Java primitive
 * types.
 */
public class Float {
  /**
   * This is the holder inner class for inout and out arguments for
   * type <code>Float</code>.
   */
  public static class Holder {
    private float d_obj;

    /**
     * Create a holder class with an empty holdee object.
     */
    public Holder() {
      d_obj = 0.0f;
    }

    /**
     * Create a holder with the specified object.
     */
    public Holder(float obj) {
      d_obj = obj;
    }

    /**
     * Set the value of the holdee object.
     */
    public void set(float obj) {
      d_obj = obj;
    }

    /**
     * Get the value of the holdee object.
     */
    public float get() {
      return d_obj;
    }
  }

  /**
   * Define a one dimensional array of type <code>float</code>
   * for the SIDL Java run-time.  Many of these methods will throw
   * array index exceptions if the specified indices are out of bounds.
   */
  public static class Array extends gov.llnl.sidl.BaseArray {
    /*
     * Register all native JNI routines for this class.
     */
    static {
      gov.llnl.sidl.BaseClass._registerNatives("SIDL.Float");
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
    public native float _get(
      int i, int j, int k, int l);

    /**
     * Native routine to set the specified value in the array.  The
     * specified array index/indices must be lie between the array lower
     * upper bounds (inclusive).  Invalid indices will have unpredictable
     * (but almost certainly bad) results.
     */
    public native void _set(
      int i, int j, int k, int l, float value);

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
   * Define a one dimensional array of type <code>float</code>.
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
    public Array1(float[] array) {
      super();
      set(array);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public float _get(int i) {
      return _get(i, 0, 0, 0);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public float get(int i) {
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
    public void _set(int i, float value) {
      _set(i, 0, 0, 0, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, float value) {
      checkBounds(i);
      _set(i, 0, 0, 0, value);
    }

    /**
     * Convert the SIDL array into a Java array.  This method will copy
     * the SIDL array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the SIDL array is
     * empty (null), then a null Java array will be returned.
     */
    public float[] get() {
      float[] array = null;
      if (!isNull()) {
        checkDimension(1);
        int l0 = _lower(0);
        int u0 = _upper(0);
        array = new float[u0-l0+1];
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
    public void set(float[] array) {
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
   * Define a two dimensional array of type <code>float</code>.
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
    public Array2(float[][] array) {
      super();
      set(array);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public float _get(int i, int j) {
      return _get(i, j, 0, 0);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public float get(int i, int j) {
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
    public void _set(int i, int j, float value) {
      _set(i, j, 0, 0, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, int j, float value) {
      checkBounds(i, j);
      _set(i, j, 0, 0, value);
    }

    /**
     * Convert the SIDL array into a Java array.  This method will copy
     * the SIDL array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the SIDL array is
     * empty (null), then a null Java array will be returned.
     */
    public float[][] get() {
      float[][] array = null;
      if (!isNull()) {
        checkDimension(2);
        int l0 = _lower(0);
        int u0 = _upper(0);
        int l1 = _lower(1);
        int u1 = _upper(1);
        array = new float[u0-l0+1][];
        for (int i = l0; i <= u0; i++) {
          array[i] = new float[u1-l1+1];
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
    public void set(float[][] array) {
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
   * Define a three dimensional array of type <code>float</code>.
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
    public Array3(float[][][] array) {
      super();
      set(array);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public float _get(int i, int j, int k) {
      return _get(i, j, k, 0);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public float get(int i, int j, int k) {
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
    public void _set(int i, int j, int k, float value) {
      _set(i, j, k, 0, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, int j, int k, float value) {
      checkBounds(i, j, k);
      _set(i, j, k, 0, value);
    }

    /**
     * Convert the SIDL array into a Java array.  This method will copy
     * the SIDL array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the SIDL array is
     * empty (null), then a null Java array will be returned.
     */
    public float[][][] get() {
      float[][][] array = null;
      if (!isNull()) {
        checkDimension(3);
        int l0 = _lower(0);
        int u0 = _upper(0);
        int l1 = _lower(1);
        int u1 = _upper(1);
        int l2 = _lower(2);
        int u2 = _upper(2);
        array = new float[u0-l0+1][][];
        for (int i = l0; i <= u0; i++) {
          array[i] = new float[u1-l1+1][];
          for (int j = l1; j <= u1; j++) {
            array[i][j] = new float[u2-l2+1];
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
    public void set(float[][][] array) {
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
   * Define a four dimensional array of type <code>float</code>.
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
    public Array4(float[][][][] array) {
      super();
      set(array);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public float _get(int i, int j, int k, int l) {
      return _get(i, j, k, l);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public float get(int i, int j, int k, int l) {
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
    public void _set(int i, int j, int k, int l, float value) {
      _set(i, j, k, l, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, int j, int k, int l, float value) {
      checkBounds(i, j, k, l);
      _set(i, j, k, l, value);
    }

    /**
     * Convert the SIDL array into a Java array.  This method will copy
     * the SIDL array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the SIDL array is
     * empty (null), then a null Java array will be returned.
     */
    public float[][][][] get() {
      float[][][][] array = null;
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
        array = new float[u0-l0+1][][][];
        for (int i = l0; i <= u0; i++) {
          array[i] = new float[u1-l1+1][][];
          for (int j = l1; j <= u1; j++) {
            array[i][j] = new float[u2-l2+1][];
            for (int k = l2; k <= u2; k++) {
              array[i][j][k] = new float[u3-l3+1];
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
    public void set(float[][][][] array) {
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
