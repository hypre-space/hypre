//
// File:	Boolean.java
// Package:	sidl
// Copyright:	(c) 2000-2001 The Regents of the University of California
// Revision:	$Revision$
// Modified:	$Date$
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


package sidl;

/**
 * Class <code>Boolean</code> contains inner classes that
 * provide holder and array support for standard Java primitive
 * types.
 */
public class Boolean {
  /**
   * This is the holder inner class for inout and out arguments for
   * type <code>Boolean</code>.
   */
  public static class Holder {
    private boolean d_obj;

    /**
     * Create a holder class with an empty holdee object.
     */
    public Holder() {
      d_obj = false;
    }

    /**
     * Create a holder with the specified object.
     */
    public Holder(boolean obj) {
      d_obj = obj;
    }

    /**
     * Set the value of the holdee object.
     */
    public void set(boolean obj) {
      d_obj = obj;
    }

    /**
     * Get the value of the holdee object.
     */
    public boolean get() {
      return d_obj;
    }
  }

  /**
   * Define a one dimensional array of type <code>boolean</code>
   * for the sidl Java run-time.  Many of these methods will throw
   * array index exceptions if the specified indices are out of bounds.
   */
  public static class Array extends gov.llnl.sidl.BaseArray {
    /*
     * Register all native JNI routines for this class.
     */
    static {
      gov.llnl.sidl.BaseClass._registerNatives("sidl.Boolean");
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
     * If isRow is true, the array will be in Row order
     */
    public Array(int dim, int[] lower, int[] upper, boolean isRow) {
      super();
      reallocate(dim, lower, upper, isRow);
    }

    /**
     * Native routine to fetch the specified value from the array.  The
     * specified array index/indices must be lie between the array lower
     * upper bounds (inclusive).  Invalid indices will have unpredictable
     * (but almost certainly bad) results.
     */
    public native boolean _get(
      int i, int j, int k, int l, int m, int n, int o);

    /**
     * Native routine to set the specified value in the array.  The
     * specified array index/indices must be lie between the array lower
     * upper bounds (inclusive).  Invalid indices will have unpredictable
     * (but almost certainly bad) results.
     */
    public native void _set(
      int i, int j, int k, int l, int m, int n, int o, boolean value);

    /**
     * Native routine to reallocate data in the array.  The specified array
     * dimension and indices must match and be within valid ranges (e.g., the
     * upper bounds must be greater than or equal to lower bounds.)  Invalid
     * indices will have unpredictable (but almost certainly bad) results.
     * This routine will deallocate the existing array data if it is not null.
     */
    public native void _reallocate(int dim, int[] lower, int[] upper, boolean isRow);
  
    /**
     * Slice returns an array that is <= the orignial array.  It shares
     * data with the orginal array.  
     * dimen gives the number of dimensions in the result array
     * numElem array gives the number of elements in each dimension
     * srcStart gives the array index to start the result array at
     * srcStride gives the stride of the result array's elements over
     * the original array's elements.
     * See the Babel user's manual for more information.
     */
     public native Array _slice(int dimen, int[] numElem, int[] srcStart,					int[] srcStride, int[] newStart);  	

  /**
   * Method smartCopy returns a a copy of a borrowed array, or  	 
   * increments the reference count of an array that manages it's 
   * own data.  Useful if you wish to keep a copy of an incoming array
   */
   //public native gov.llnl.sidl.BaseArray _smartCopy();

   /**
    * Method Copy copies the elements of 'this' to an already existing 
    * array of the same size.  NOT LIKE clone()!!
    */
    public native void _copy(Array dest);
 

    /**
     *  Casts this array to an array of a defined dimension and returns 
     *  the resulting array.  (You might want to deallocate the original 
     *  array.  
     *  Argument dimen determines what dimension array to cast this
     *  array to.  
     */

  public Array _dcast() {
    try{ 
      int dimen = _dim();
      sidl.Boolean.Array ret = null;
      switch (dimen) {
      case 1: 

	ret = (Array) new sidl.Boolean.Array1(get_ior_pointer(),true);
        _addRef();
        return ret;
      case 2:
        ret = (Array) new sidl.Boolean.Array2(get_ior_pointer(),true);
        _addRef();
        return ret;
      case 3:
        ret = (Array) new sidl.Boolean.Array3(get_ior_pointer(),true);
        _addRef();
        return ret;
      case 4:
        ret = (Array) new sidl.Boolean.Array4(get_ior_pointer(),true);
         _addRef();
        return ret;
      case 5:
        ret = (Array) new sidl.Boolean.Array5(get_ior_pointer(),true);
        _addRef();
        return ret;
      case 6:
        ret = (Array) new sidl.Boolean.Array6(get_ior_pointer(),true);
         _addRef();
        return ret;
      case 7:
        ret = (Array) new sidl.Boolean.Array7(get_ior_pointer(),true);
         _addRef();
        return ret;
      default:
        return null;
      }
    } catch (Exception ex) {
      return null;	
    }

  } 

    public static class Holder {
      private Array d_obj;

    /**
     * Create a holder class with an empty holdee object.
     */
    public Holder() {
      d_obj = null;
    }

    /**
     * Create a holder with the specified object.
     */
    public Holder(Array obj) {
      d_obj = obj;
    }

    /**
     * Set the value of the holdee object.
     */
    public void set(Array obj) {
      d_obj = obj;
    }

    /**
     * Get the value of the holdee object.
     */
    public Array get() {
      return d_obj;
    }
  }

}
  /**
   * Define a one dimensional array of type <code>boolean</code>.
   * This array representation is used for sidl arrays since it requires
   * no copies to go between Java and sidl.  Explicit copies may be made
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
     * Create a one dimensional array directly using the sidl pointer
     * and owner flag.  This constructor should only be called by the
     * sidl runtime.
     */
    protected Array1(long array, boolean owner) {
      super(array, owner);
    }

    /**
     * Create a one dimensional array using the specified lower and upper
     * bounds (where both bounds are inclusive).  This constructor will throw
     * an array bounds out of range exception if the array bounds are invalid.
     */
    public Array1(int l0, int u0, boolean isRow) {
      super(1, new int[] { l0 }, new int[] { u0 }, isRow);
    }

    /**
     * Create a one dimenstional array of the specified size, with the lower
     * index starting at zero.  This constructor will throw an array bounds
     * out of range exception if the array bounds are invalid.
     */
    public Array1(int s0, boolean isRow) {
      super(1, new int[] { 0 }, new int[] { s0-1 }, isRow);
    }

    /**
     * Create a one dimensional array using the specified Java array.  The
     * lower bound(s) of the constructed array will start at zero.  An array
     * index out of range exception will be thrown if the bounds are invalid.
     */
    public Array1(boolean[] array) {
      super();
      fromArray(array);
    }

    /**
     * Routine gets length of the array
     */
    public int length() {
	return super._length(0);
    }
     
    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public boolean _get(int i) {
      return _get(i, 0, 0, 0, 0, 0, 0);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public boolean get(int i) {
      checkBounds(i);
      return _get(i, 0, 0, 0, 0, 0, 0);
    }

    /**
     * Reallocate array data using the specifed lower and upper bounds.  The
     * upper bound is inclusive.  Previous array data will be freed.
     */
    public void reallocate(int l0, int u0, boolean isRow) {
      reallocate(1, new int[] { l0 }, new int[] { u0 }, isRow);
    }

  /**
   * Method smartCopy returns a a copy of a borrowed array, or  	
   * increments the reference count of an array that manages it's
   * own data.  Useful if you wish to keep a copy of an incoming array
   */
   public Array1 smartCopy() {
     return (Array1) ((Array)_smartCopy())._dcast();
   }

   /**
    * Method Copy copies the elements of 'this' to an already existing 
    * array of the same size.
    */
    public void copy(Array1 dest) {
      _copy((Array) dest);	
    }

    /**
     * Set the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>set</code> instead.
     */
    public void _set(int i, boolean value) {
      _set(i, 0, 0, 0, 0, 0, 0,value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, boolean value) {
      checkBounds(i);
      _set(i, 0, 0, 0, 0, 0, 0, value);
    }

    /**
     * Convert the sidl array into a Java array.  This method will copy
     * the sidl array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the sidl array is
     * empty (null), then a null Java array will be returned.
     */
    public boolean[] toArray() {
      boolean[] array = null;
      if (!isNull()) {
        checkDimension(1);
        int l0 = _lower(0);
        int u0 = _upper(0);
        array = new boolean[u0-l0+1];
        for (int i = l0; i <= u0; i++) {
          array[i-l0] = _get(i);
        }
      }
      return array;
    }

    /**
     * Set the value of the sidl array from the Java array.  This method
     * will copy the Java array values into the sidl array, reallocating
     * the memory of the sidl array as necessary.  The resulting sidl array
     * will start with a zero lower bound.  If the Java array is null, then
     * the sidl array will be null, as well.  Note that multidimensional Java
     * arrays must not be ragged; that is, all sub-arrays in a particular
     * dimension must have the same size.  Otherwise, some data may be missed
     * or this method may throw an array index out of bounds exception.
     */
    public void fromArray(boolean[] array) {
      if (array == null) {
        destroy();
      } else {
        int s0 = array.length-1;
        reallocate(0, s0, _isRowOrder());
        for (int i = 0; i <= s0; i++) {
          _set(i, array[i]);
        }
      }
    }


    public static class Holder {
      private Array1 d_obj;

    /**
     * Create a holder class with an empty holdee object.
     */
    public Holder() {
      d_obj = null;
    }

    /**
     * Create a holder with the specified object.
     */
    public Holder(Array1 obj) {
      d_obj = obj;
    }

    /**
     * Set the value of the holdee object.
     */
    public void set(Array1 obj) {
      d_obj = obj;
    }

    /**
     * Get the value of the holdee object.
     */
    public Array1 get() {
      return d_obj;
    }
  }
  }

  /**
   * Define a two dimensional array of type <code>boolean</code>.
   * This array representation is used for sidl arrays since it requires
   * no copies to go between Java and sidl.  Explicit copies may be made
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
     * Create a two dimensional array directly using the sidl pointer
     * and owner flag.  This constructor should only be called by the
     * sidl runtime.
     */
    protected Array2(long array, boolean owner) {
      super(array, owner);
    }

    /**
     * Create a two dimensional array using the specified lower and upper
     * bounds (where both bounds are inclusive).  This constructor will throw
     * an array bounds out of range exception if the array bounds are invalid.
     */
    public Array2(int l0, int l1, int u0, int u1, boolean isRow) {
      super(2, new int[] { l0, l1 }, new int[] { u0, u1 }, isRow);
    }

    /**
     * Create a two dimenstional array of the specified size, with the lower
     * index starting at zero.  This constructor will throw an array bounds
     * out of range exception if the array bounds are invalid.
     */
    public Array2(int s0, int s1, boolean isRow) {
      super(2, new int[] { 0, 0 }, new int[] { s0-1, s1-1 }, isRow);
    }

    /**
     * Create a two dimensional array using the specified Java array.  The
     * lower bound(s) of the constructed array will start at zero.  An array
     * index out of range exception will be thrown if the bounds are invalid.
     */
    public Array2(boolean[][] array) {
      super();
      fromArray(array);
    }

    /**
     * Routine gets length of the array in the specified dimension
     */
    public int length(int dim) {
	return super._length(dim);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public boolean _get(int i, int j) {
      return _get(i, j, 0, 0, 0, 0, 0);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public boolean get(int i, int j) {
      checkBounds(i, j);
      return _get(i, j, 0, 0, 0, 0, 0);
    }

    /**
     * Reallocate array data using the specifed lower and upper bounds.  The
     * upper bound is inclusive.  Previous array data will be freed.
     */
    public void reallocate(int l0, int l1, int u0, int u1, boolean isRow) {
      reallocate(2, new int[] { l0, l1 }, new int[] { u0, u1 }, isRow);
    }

    /**
     * Method smartCopy returns a a copy of a borrowed array, or  	
     * increments the reference count of an array that manages it's
     * own data.  Useful if you wish to keep a copy of an incoming array
     */
     public Array2 smartCopy() {
       return (Array2) ((Array)_smartCopy())._dcast();
     }

   /**
    * Method Copy copies the elements of 'this' to an already existing 
    * array of the same size.
    */
    public void copy(Array2 dest) {
      _copy((Array) dest);	
    }

    /**
     * Set the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>set</code> instead.
     */
    public void _set(int i, int j, boolean value) {
      _set(i, j, 0, 0, 0, 0, 0, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, int j, boolean value) {
      checkBounds(i, j);
      _set(i, j, 0, 0, 0, 0, 0, value);
    }


    /**
     * Convert the sidl array into a Java array.  This method will copy
     * the sidl array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the sidl array is
     * empty (null), then a null Java array will be returned.
     */
    public boolean[][] toArray() {
      boolean[][] array = null;
      if (!isNull()) {
        checkDimension(2);
        int l0 = _lower(0);
        int u0 = _upper(0);
        int l1 = _lower(1);
        int u1 = _upper(1);
        array = new boolean[u0-l0+1][];
        for (int i = l0; i <= u0; i++) {
          array[i] = new boolean[u1-l1+1];
          for (int j = l1; j <= u1; j++) {
            array[i-l0][j-l1] = _get(i, j);
          }
        }
      }
      return array;
    }

    /**
     * Set the value of the sidl array from the Java array.  This method
     * will copy the Java array values into the sidl array, reallocating
     * the memory of the sidl array as necessary.  The resulting sidl array
     * will start with a zero lower bound.  If the Java array is null, then
     * the sidl array will be null, as well.  Note that multidimensional Java
     * arrays must not be ragged; that is, all sub-arrays in a particular
     * dimension must have the same size.  Otherwise, some data may be missed
     * or this method may throw an array index out of bounds exception.
     */
    public void fromArray(boolean[][] array) {
      if (array == null) {
        destroy();
      } else {
        int s0 = array.length-1;
        int s1 = array[0].length-1;
        reallocate(0, 0, s0, s1, _isRowOrder());
        for (int i = 0; i <= s0; i++) {
          for (int j = 0; j <= s1; j++) {
            _set(i, j, array[i][j]);
          }
        }
      }
    }

    public static class Holder {
      private Array2 d_obj;

    /**
     * Create a holder class with an empty holdee object.
     */
    public Holder() {
      d_obj = null;
    }

    /**
     * Create a holder with the specified object.
     */
    public Holder(Array2 obj) {
      d_obj = obj;
    }

    /**
     * Set the value of the holdee object.
     */
    public void set(Array2 obj) {
      d_obj = obj;
    }

    /**
     * Get the value of the holdee object.
     */
    public Array2 get() {
      return d_obj;
    }
    }
  }

  /**
   * Define a three dimensional array of type <code>boolean</code>.
   * This array representation is used for sidl arrays since it requires
   * no copies to go between Java and sidl.  Explicit copies may be made
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
     * Create a three dimensional array directly using the sidl pointer
     * and owner flag.  This constructor should only be called by the
     * sidl runtime.
     */
    protected Array3(long array, boolean owner) {
      super(array, owner);
    }

    /**
     * Create a three dimensional array using the specified lower and upper
     * bounds (where both bounds are inclusive).  This constructor will throw
     * an array bounds out of range exception if the array bounds are invalid.
     */
    public Array3(int l0, int l1, int l2, int u0, int u1, int u2, boolean isRow) {
      super(3, new int[] { l0, l1, l2 }, new int[] { u0, u1, u2 }, isRow);
    }

    /**
     * Create a three dimenstional array of the specified size, with the lower
     * index starting at zero.  This constructor will throw an array bounds out
     * of range exception if the array bounds are invalid.
     */
    public Array3(int s0, int s1, int s2, boolean isRow) {
      super(3, new int[] { 0, 0, 0 }, new int[] { s0-1, s1-1, s2-1 }, isRow);
    }

    /**
     * Create a three dimensional array using the specified Java array.  The
     * lower bound(s) of the constructed array will start at zero.  An array
     * index out of range exception will be thrown if the bounds are invalid.
     */
    public Array3(boolean[][][] array) {
      super();
      fromArray(array);
    }

    /**
     * Routine gets length of the array in the specified dimension
     */
    public int length(int dim) {
	return super._length(dim);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public boolean _get(int i, int j, int k) {
      return _get(i, j, k, 0, 0, 0, 0);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public boolean get(int i, int j, int k) {
      checkBounds(i, j, k);
      return _get(i, j, k, 0, 0, 0, 0);
    }

    /**
     * Reallocate array data using the specifed lower and upper bounds.  The
     * upper bound is inclusive.  Previous array data will be freed.
     */
    public void reallocate(int l0, int l1, int l2, int u0, int u1, int u2,
	boolean isRow) {
      reallocate(3, new int[] { l0, l1, l2 }, new int[] { u0, u1, u2 }, isRow);
    }

    /**
     * Method smartCopy returns a a copy of a borrowed array, or  	
     * increments the reference count of an array that manages it's
     * own data.  Useful if you wish to keep a copy of an incoming array
     */
     public Array3 smartCopy() {
       return (Array3) ((Array)_smartCopy())._dcast();
     }

   /**
    * Method Copy copies the elements of 'this' to an already existing 
    * array of the same size.
    */
    public void copy(Array3 dest) {
      _copy((Array) dest);	
    }

    /**
     * Set the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>set</code> instead.
     */
    public void _set(int i, int j, int k, boolean value) {
      _set(i, j, k, 0, 0, 0, 0, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, int j, int k, boolean value) {
      checkBounds(i, j, k);
      _set(i, j, k, 0, 0, 0, 0, value);
    }

    /**
     * Convert the sidl array into a Java array.  This method will copy
     * the sidl array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the sidl array is
     * empty (null), then a null Java array will be returned.
     */
    public boolean[][][] toArray() {
      boolean[][][] array = null;
      if (!isNull()) {
        checkDimension(3);
        int l0 = _lower(0);
        int u0 = _upper(0);
        int l1 = _lower(1);
        int u1 = _upper(1);
        int l2 = _lower(2);
        int u2 = _upper(2);
        array = new boolean[u0-l0+1][][];
        for (int i = l0; i <= u0; i++) {
          array[i] = new boolean[u1-l1+1][];
          for (int j = l1; j <= u1; j++) {
            array[i][j] = new boolean[u2-l2+1];
            for (int k = l2; k <= u2; k++) {
              array[i-l0][j-l1][k-l2] = _get(i, j, k);
            }
          }
        }
      }
      return array;
    }

    /**
     * Set the value of the sidl array from the Java array.  This method
     * will copy the Java array values into the sidl array, reallocating
     * the memory of the sidl array as necessary.  The resulting sidl array
     * will start with a zero lower bound.  If the Java array is null, then
     * the sidl array will be null, as well.  Note that multidimensional Java
     * arrays must not be ragged; that is, all sub-arrays in a particular
     * dimension must have the same size.  Otherwise, some data may be missed
     * or this method may throw an array index out of bounds exception.
     */
    public void fromArray(boolean[][][] array) {
      if (array == null) {
        destroy();
      } else {
        int s0 = array.length-1;
        int s1 = array[0].length-1;
        int s2 = array[0][0].length-1;
        reallocate(0, 0, 0, s0, s1, s2, _isRowOrder());
        for (int i = 0; i <= s0; i++) {
          for (int j = 0; j <= s1; j++) {
            for (int k = 0; k <= s1; k++) {
              _set(i, j, k, array[i][j][k]);
            }
          }
        }
      }
    }

    public static class Holder {
      private Array3 d_obj;

    /**
     * Create a holder class with an empty holdee object.
     */
    public Holder() {
      d_obj = null;
    }

    /**
     * Create a holder with the specified object.
     */
    public Holder(Array3 obj) {
      d_obj = obj;
    }

    /**
     * Set the value of the holdee object.
     */
    public void set(Array3 obj) {
      d_obj = obj;
    }

    /**
     * Get the value of the holdee object.
     */
    public Array3 get() {
      return d_obj;
    }
  }

  }

  /**
   * Define a four dimensional array of type <code>boolean</code>.
   * This array representation is used for sidl arrays since it requires
   * no copies to go between Java and sidl.  Explicit copies may be made
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
     * Create a four dimensional array directly using the sidl pointer
     * and owner flag.  This constructor should only be called by the
     * sidl runtime.
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
                  int u0, int u1, int u2, int u3, boolean isRow) {
      super(4, new int[] { l0, l1, l2, l3 }, new int[] { u0, u1, u2, u3 },
	    isRow);
    }

    /**
     * Create a four dimenstional array of the specified size, with the lower
     * index starting at zero.  This constructor will throw an array bounds out
     * of range exception if the array bounds are invalid.
     */
    public Array4(int s0, int s1, int s2, int s3, boolean isRow) {
      super(4, new int[] { 0, 0, 0, 0 }, new int[] { s0-1, s1-1, s2-1, s3-1 },
	isRow);
    }

    /**
     * Create a four dimensional array using the specified Java array.  The
     * lower bound(s) of the constructed array will start at zero.  An array
     * index out of range exception will be thrown if the bounds are invalid.
     */
    public Array4(boolean[][][][] array) {
      super();
      fromArray(array);
    }

    /**
     * Routine gets length of the array in the specified dimension
     */
    public int length(int dim) {
	return super._length(dim);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public boolean _get(int i, int j, int k, int l) {
      return _get(i, j, k, l, 0, 0, 0);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public boolean get(int i, int j, int k, int l) {
      checkBounds(i, j, k, l);
      return _get(i, j, k, l, 0, 0, 0);
    }

    /**
     * Reallocate array data using the specifed lower and upper bounds.  The
     * upper bound is inclusive.  Previous array data will be freed.
     */
    public void reallocate(int l0, int l1, int l2, int l3,
                           int u0, int u1, int u2, int u3, boolean isRow) {
      reallocate(4, new int[] { l0, l1, l2, l3 }, new int[] { u0, u1, u2, l3 },
      isRow);
    }

    /**
     * Method smartCopy returns a a copy of a borrowed array, or  	
     * increments the reference count of an array that manages it's
     * own data.  Useful if you wish to keep a copy of an incoming array
     */
     public Array4 smartCopy() {
       return (Array4) ((Array)_smartCopy())._dcast();
     }

   /**
    * Method Copy copies the elements of 'this' to an already existing 
    * array of the same size.
    */
    public void copy(Array4 dest) {
      _copy((Array) dest);	
    }

    /**
     * Set the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>set</code> instead.
     */
    public void _set(int i, int j, int k, int l, boolean value) {
      _set(i, j, k, l, 0, 0, 0, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, int j, int k, int l, boolean value) {
      checkBounds(i, j, k, l);
      _set(i, j, k, l, 0, 0, 0, value);
    }

    /**
     * Convert the sidl array into a Java array.  This method will copy
     * the sidl array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the sidl array is
     * empty (null), then a null Java array will be returned.
     */
    public boolean[][][][] toArray() {
      boolean[][][][] array = null;
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
        array = new boolean[u0-l0+1][][][];
        for (int i = l0; i <= u0; i++) {
          array[i] = new boolean[u1-l1+1][][];
          for (int j = l1; j <= u1; j++) {
            array[i][j] = new boolean[u2-l2+1][];
            for (int k = l2; k <= u2; k++) {
              array[i][j][k] = new boolean[u3-l3+1];
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
     * Set the value of the sidl array from the Java array.  This method
     * will copy the Java array values into the sidl array, reallocating
     * the memory of the sidl array as necessary.  The resulting sidl array
     * will start with a zero lower bound.  If the Java array is null, then
     * the sidl array will be null, as well.  Note that multidimensional Java
     * arrays must not be ragged; that is, all sub-arrays in a particular
     * dimension must have the same size.  Otherwise, some data may be missed
     * or this method may throw an array index out of bounds exception.
     */
    public void fromArray(boolean[][][][] array) {
      if (array == null) {
        destroy();
      } else {
        int s0 = array.length-1;
        int s1 = array[0].length-1;
        int s2 = array[0][0].length-1;
        int s3 = array[0][0][0].length-1;
        reallocate(0, 0, 0, 0, s0, s1, s2, s3, _isRowOrder());
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

    public static class Holder {
      private Array4 d_obj;

    /**
     * Create a holder class with an empty holdee object.
     */
    public Holder() {
      d_obj = null;
    }

    /**
     * Create a holder with the specified object.
     */
    public Holder(Array4 obj) {
      d_obj = obj;
    }

    /**
     * Set the value of the holdee object.
     */
    public void set(Array4 obj) {
      d_obj = obj;
    }

    /**
     * Get the value of the holdee object.
     */
    public Array4 get() {
      return d_obj;
    }
  }

  }


 /**
   * Define a five dimensional array of type <code>boolean</code>.
   * This array representation is used for sidl arrays since it requires
   * no copies to go between Java and sidl.  Explicit copies may be made
   * of the array by calling the appropriate <code>get</code> and
   * <code>set</code> methods.
   */
  public static class Array5 extends Array {
    /**
     * Create an empty four dimensional array.  The array will need to be
     * initialized before use.
     */
    public Array5() {
      super();
    }

    /**
     * Create a five dimensional array directly using the sidl pointer
     * and owner flag.  This constructor should only be called by the
     * sidl runtime.
     */
    protected Array5(long array, boolean owner) {
      super(array, owner);
    }

    /**
     * Create a five dimensional array using the specified lower and upper
     * bounds (where both bounds are inclusive).  This constructor will throw
     * an array bounds out of range exception if the array bounds are invalid.
     */
      public Array5(int l0, int l1, int l2, int l3, int l4,
                  int u0, int u1, int u2, int u3, int u4, boolean isRow) {
      super(5, new int[] { l0, l1, l2, l3, l4}, 
	       new int[] { u0, u1, u2, u3, u4 }, isRow);
    }

    /**
     * Create a five dimenstional array of the specified size, with the lower
     * index starting at zero.  This constructor will throw an array bounds out
     * of range exception if the array bounds are invalid.
     */
    public Array5(int s0, int s1, int s2, int s3, int s4, boolean isRow) {
      super(5, new int[] { 0, 0, 0, 0, 0 }, 
	new int[] { s0-1, s1-1, s2-1, s3-1, s4-1 }, isRow);
    }

    /**
     * Create a five dimensional array using the specified Java array.  The
     * lower bound(s) of the constructed array will start at zero.  An array
     * index out of range exception will be thrown if the bounds are invalid.
     */
    public Array5(boolean[][][][][] array) {
      super();
      fromArray(array);
    }

    /**
     * Routine gets length of the array in the specified dimension
     */
    public int length(int dim) {
	return super._length(dim);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public boolean _get(int i, int j, int k, int l, int m) {
      return _get(i, j, k, l, m, 0, 0);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public boolean get(int i, int j, int k, int l, int m) {
      checkBounds(i, j, k, l, m);
      return _get(i, j, k, l, m, 0, 0);
    }

    /**
     * Reallocate array data using the specifed lower and upper bounds.  The
     * upper bound is inclusive.  Previous array data will be freed.
     */
    public void reallocate(int l0, int l1, int l2, int l3, int l4,
                           int u0, int u1, int u2, int u3, int u4,
			   boolean isRow ) {
      reallocate(5, new int[] { l0, l1, l2, l3, l4 }, new int[] { u0, u1, u2, u3, u4 }, isRow);
    }

    /**
     * Method smartCopy returns a a copy of a borrowed array, or  	
     * increments the reference count of an array that manages it's
     * own data.  Useful if you wish to keep a copy of an incoming array
     */
     public Array5 smartCopy() {
       return (Array5) ((Array)_smartCopy())._dcast();
    }

   /**
    * Method Copy copies the elements of 'this' to an already existing 
    * array of the same size.
    */
    public void copy(Array5 dest) {
      _copy((Array) dest);	
    }

    /**
     * Set the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>set</code> instead.
     */
    public void _set(int i, int j, int k, int l, int m, boolean value) {
      _set(i, j, k, l, m, 0, 0, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, int j, int k, int l, int m, boolean value) {
      checkBounds(i, j, k, l, m);
      _set(i, j, k, l, m, 0, 0, value);
    }

    /**
     * Convert the sidl array into a Java array.  This method will copy
     * the sidl array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the sidl array is
     * empty (null), then a null Java array will be returned.
     */
    public boolean[][][][][] toArray() {
      boolean[][][][][] array = null;
      if (!isNull()) {
        checkDimension(5);
        int l0 = _lower(0);
        int u0 = _upper(0);
        int l1 = _lower(1);
        int u1 = _upper(1);
        int l2 = _lower(2);
        int u2 = _upper(2);
        int l3 = _lower(3);
        int u3 = _upper(3);
	int l4 = _lower(4);
        int u4 = _upper(4);
        array = new boolean[u0-l0+1][][][][];
        for (int i = l0; i <= u0; i++) {
          array[i] = new boolean[u1-l1+1][][][];
          for (int j = l1; j <= u1; j++) {
            array[i][j] = new boolean[u2-l2+1][][];
            for (int k = l2; k <= u2; k++) {
              array[i][j][k] = new boolean[u3-l3+1][];
              for (int l = l3; l <= u3; l++) {
		array[i][j][k][l] = new boolean[u4-l4+1];
		for (int m = l4; m <= u4; m++) {
		  array[i-l0][j-l1][k-l2][l-l3][m-l4] = _get(i, j, k, l, m);
		}
              }
            }
          }
        }
      }
      return array;
    }

    /**
     * Set the value of the sidl array from the Java array.  This method
     * will copy the Java array values into the sidl array, reallocating
     * the memory of the sidl array as necessary.  The resulting sidl array
     * will start with a zero lower bound.  If the Java array is null, then
     * the sidl array will be null, as well.  Note that multidimensional Java
     * arrays must not be ragged; that is, all sub-arrays in a particular
     * dimension must have the same size.  Otherwise, some data may be missed
     * or this method may throw an array index out of bounds exception.
     */
    public void fromArray(boolean[][][][][] array) {
      if (array == null) {
        destroy();
      } else {
        int s0 = array.length-1;
        int s1 = array[0].length-1;
        int s2 = array[0][0].length-1;
        int s3 = array[0][0][0].length-1;
	int s4 = array[0][0][0][0].length-1;
        reallocate(0, 0, 0, 0, 0, s0, s1, s2, s3, s4, _isRowOrder());
        for (int i = 0; i <= s0; i++) {
          for (int j = 0; j <= s1; j++) {
            for (int k = 0; k <= s2; k++) {
              for (int l = 0; l <= s3; l++) {
		for (int m = 0; m <= s4; m++) {
                _set(i, j, k, l, m, array[i][j][k][l][m]);
		}
              }
            }
          }
        }
      }
    }

    public static class Holder {
      private Array5 d_obj;

    /**
     * Create a holder class with an empty holdee object.
     */
    public Holder() {
      d_obj = null;
    }

    /**
     * Create a holder with the specified object.
     */
    public Holder(Array5 obj) {
      d_obj = obj;
    }

    /**
     * Set the value of the holdee object.
     */
    public void set(Array5 obj) {
      d_obj = obj;
    }

    /**
     * Get the value of the holdee object.
     */
    public Array5 get() {
      return d_obj;
    }
  }

  }


/**
   * Define a six dimensional array of type <code>boolean</code>.
   * This array representation is used for sidl arrays since it requires
   * no copies to go between Java and sidl.  Explicit copies may be made
   * of the array by calling the appropriate <code>get</code> and
   * <code>set</code> methods.
   */
  public static class Array6 extends Array {
    /**
     * Create an empty six dimensional array.  The array will need to be
     * initialized before use.
     */
    public Array6() {
      super();
    }

    /**
     * Create a six dimensional array directly using the sidl pointer
     * and owner flag.  This constructor should only be called by the
     * sidl runtime.
     */
    protected Array6(long array, boolean owner) {
      super(array, owner);
    }

    /**
     * Create a six dimensional array using the specified lower and upper
     * bounds (where both bounds are inclusive).  This constructor will throw
     * an array bounds out of range exception if the array bounds are invalid.
     */
      public Array6(int l0, int l1, int l2, int l3, int l4, int l5,
                  int u0, int u1, int u2, int u3, int u4, int u5, boolean isRow) {
      super(6, new int[] { l0, l1, l2, l3, l4, l5}, new int[] { u0, u1, u2, u3, u4, u5 }, isRow);
    }

    /**
     * Create a six dimenstional array of the specified size, with the lower
     * index starting at zero.  This constructor will throw an array bounds out
     * of range exception if the array bounds are invalid.
     */
    public Array6(int s0, int s1, int s2, int s3, int s4, int s5, boolean isRow) {
      super(6, new int[] { 0, 0, 0, 0, 0, 0 }, 
               new int[] { s0-1, s1-1, s2-1, s3-1, s4-1, s5-1 }, isRow);
    }

    /**
     * Create a six dimensional array using the specified Java array.  The
     * lower bound(s) of the constructed array will start at zero.  An array
     * index out of range exception will be thrown if the bounds are invalid.
     */
    public Array6(boolean[][][][][][] array) {
      super();
      fromArray(array);
    }

    /**
     * Routine gets length of the array in the specified dimension
     */
    public int length(int dim) {
	return super._length(dim);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public boolean _get(int i, int j, int k, int l, int m, int n) {
      return _get(i, j, k, l, m, n, 0);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public boolean get(int i, int j, int k, int l, int m, int n) {
      checkBounds(i, j, k, l, m, n);
      return _get(i, j, k, l, m, n, 0);
    }

    /**
     * Reallocate array data using the specifed lower and upper bounds.  The
     * upper bound is inclusive.  Previous array data will be freed.
     */
      public void reallocate(int l0, int l1, int l2, int l3, int l4, int l5,
                           int u0, int u1, int u2, int u3, int u4, int u5,
			    boolean isRow) {
      reallocate(6, new int[] { l0, l1, l2, l3, l4, l5 }, new int[] { u0, u1, u2, u3, u4, u5 }, isRow);
    }

   /**
    * Method smartCopy returns a a copy of a borrowed array, or  	
    * increments the reference count of an array that manages it's
    * own data.  Useful if you wish to keep a copy of an incoming array
    */
    public Array6 smartCopy() {
      return (Array6) ((Array)_smartCopy())._dcast();
    }

   /**
    * Method Copy copies the elements of 'this' to an already existing 
    * array of the same size.
    */
    public void copy(Array6 dest) {
      _copy((Array) dest);	
    }

    /**
     * Set the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>set</code> instead.
     */
    public void _set(int i, int j, int k, int l, int m, int n, boolean value) {
      _set(i, j, k, l, m, n, 0, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, int j, int k, int l, int m, int n, boolean value) {
      checkBounds(i, j, k, l, m, n);
      _set(i, j, k, l, m, n, 0, value);
    }

    /**
     * Convert the sidl array into a Java array.  This method will copy
     * the sidl array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the sidl array is
     * empty (null), then a null Java array will be returned.
     */
    public boolean[][][][][][] toArray() {
      boolean[][][][][][] array = null;
      if (!isNull()) {
        checkDimension(6);
        int l0 = _lower(0);
        int u0 = _upper(0);
        int l1 = _lower(1);
        int u1 = _upper(1);
        int l2 = _lower(2);
        int u2 = _upper(2);
        int l3 = _lower(3);
        int u3 = _upper(3);
	int l4 = _lower(4);
        int u4 = _upper(4);
	int l5 = _lower(5);
        int u5 = _upper(5);
        array = new boolean[u0-l0+1][][][][][];
        for (int i = l0; i <= u0; i++) {
	    array[i] = new boolean[u1-l1+1][][][][];
          for (int j = l1; j <= u1; j++) {
	      array[i][j] = new boolean[u2-l2+1][][][];
            for (int k = l2; k <= u2; k++) {
		array[i][j][k] = new boolean[u3-l3+1][][];
              for (int l = l3; l <= u3; l++) {
		  array[i][j][k][l] = new boolean[u4-l4+1][];
		for (int m = l4; m <= u4; m++) {
		    array[i][j][k][l][m] = new boolean[u4-l4+1];
		    for (int n = l5; n <= u5; n++) {
		      array[i-l0][j-l1][k-l2][l-l3][m-l4][n-l5] = _get(i, j, k, l, m, n);
		  }
		}
	      }
	    }
	  }
	}
      }
      return array;
    }

    /**
     * Set the value of the sidl array from the Java array.  This method
     * will copy the Java array values into the sidl array, reallocating
     * the memory of the sidl array as necessary.  The resulting sidl array
     * will start with a zero lower bound.  If the Java array is null, then
     * the sidl array will be null, as well.  Note that multidimensional Java
     * arrays must not be ragged; that is, all sub-arrays in a particular
     * dimension must have the same size.  Otherwise, some data may be missed
     * or this method may throw an array index out of bounds exception.
     */
    public void fromArray(boolean[][][][][][] array) {
      if (array == null) {
        destroy();
      } else {
        int s0 = array.length-1;
        int s1 = array[0].length-1;
        int s2 = array[0][0].length-1;
        int s3 = array[0][0][0].length-1;
	int s4 = array[0][0][0][0].length-1;
	int s5 = array[0][0][0][0][0].length-1;
        reallocate(0, 0, 0, 0, 0, 0, s0, s1, s2, s3, s4, s5, _isRowOrder());
        for (int i = 0; i <= s0; i++) {
          for (int j = 0; j <= s1; j++) {
            for (int k = 0; k <= s2; k++) {
              for (int l = 0; l <= s3; l++) {
		for (int m = 0; m <= s4; m++) { 
		  for (int n = 0; n <= s5; n++) {
                     _set(i, j, k, l, m, n, array[i][j][k][l][m][n]);
		  }
		}
              }
            }
          }
        }
      }
    }

    public static class Holder {
      private Array6 d_obj;

    /**
     * Create a holder class with an empty holdee object.
     */
    public Holder() {
      d_obj = null;
    }

    /**
     * Create a holder with the specified object.
     */
    public Holder(Array6 obj) {
      d_obj = obj;
    }

    /**
     * Set the value of the holdee object.
     */
    public void set(Array6 obj) {
      d_obj = obj;
    }

    /**
     * Get the value of the holdee object.
     */
    public Array6 get() {
      return d_obj;
    }
  }
  }



/**
   * Define a seven dimensional array of type <code>boolean</code>.
   * This array representation is used for sidl arrays since it requires
   * no copies to go between Java and sidl.  Explicit copies may be made
   * of the array by calling the appropriate <code>get</code> and
   * <code>set</code> methods.
   */
  public static class Array7 extends Array {
    /**
     * Create an empty seven dimensional array.  The array will need to be
     * initialized before use.
     */
    public Array7() {
      super();
    }

    /**
     * Create a seven dimensional array directly using the sidl pointer
     * and owner flag.  This constructor should only be called by the
     * sidl runtime.
     */
    protected Array7(long array, boolean owner) {
      super(array, owner);
    }

    /**
     * Create a seven dimensional array using the specified lower and upper
     * bounds (where both bounds are inclusive).  This constructor will throw
     * an array bounds out of range exception if the array bounds are invalid.
     */
      public Array7(int l0, int l1, int l2, int l3, int l4, int l5, int l6,
                  int u0, int u1, int u2, int u3, int u4, int u5, int u6,
		  boolean isRow) {
      super(7, new int[] { l0, l1, l2, l3, l4, l5, l6}, 
	       new int[] { u0, u1, u2, u3, u4, u5, u6 }, isRow);
    }

    /**
     * Create a seven dimenstional array of the specified size, with the lower
     * index starting at zero.  This constructor will throw an array bounds out
     * of range exception if the array bounds are invalid.
     */
    public Array7(int s0, int s1, int s2, int s3, int s4, int s5, 
		  int s6, boolean isRow) {
      super(7, new int[] {0,0,0,0,0,0,0}, 
	       new int[] { s0-1, s1-1, s2-1, s3-1, s4-1, s5-1, s6-1}, isRow);
    }

    /**
     * Create a seven dimensional array using the specified Java array.  The
     * lower bound(s) of the constructed array will start at zero.  An array
     * index out of range exception will be thrown if the bounds are invalid.
     */
    public Array7(boolean[][][][][][][] array) {
      super();
      fromArray(array);
    }

    /**
     * Routine gets length of the array in the specified dimension
     */
    public int length(int dim) {
	return super._length(dim);
    }

    /**
     * Get the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>get</code> instead.
     */
    public boolean _get(int i, int j, int k, int l, int m, int n, int o) {
      return super._get(i, j, k, l, m, n, o);
    }

    /**
     * Get the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public boolean get(int i, int j, int k, int l, int m, int n, int o) {
      checkBounds(i, j, k, l, m, n, o);
      return _get(i, j, k, l, m, n, o);
    }

    /**
     * Reallocate array data using the specifed lower and upper bounds.  The
     * upper bound is inclusive.  Previous array data will be freed.
     */
      public void reallocate(int l0, int l1, int l2, int l3, int l4, int l5, int l6,
                 int u0, int u1, int u2, int u3, int u4, int u5, int u6,
		 boolean isRow) {
      reallocate(7, new int[] {l0,l1,l2,l3,l4,l5,l6}, new int[] { u0, u1, u2, u3, u4, u5, u6 }, isRow);
    }

    /**
     * Method smartCopy returns a a copy of a borrowed array, or  	
     * increments the reference count of an array that manages it's
     * own data.  Useful if you wish to keep a copy of an incoming array
     */
      public Array7 smartCopy() {
	return (Array7) ((Array)_smartCopy())._dcast();
      }

   /**
    * Method Copy copies the elements of 'this' to an already existing 
    * array of the same size.
    */
     public void copy(Array7 dest) {
	_copy((Array) dest);	
     }


    /**
     * Set the specified array element without bounds checking.  If the index
     * is invalid, then bad things may happen.  If you are unsure whether the
     * index is valid, then use <code>set</code> instead.
     */
    public void _set(int i, int j, int k, int l, int m, int n, int o, boolean value) {
      super._set(i, j, k, l, m, n, o, value);
    }

    /**
     * Set the specified array element with bounds checking.  If the index is
     * invalid, then an array index out of bounds exception will be thrown.
     */
    public void set(int i, int j, int k, int l, int m, int n, int o, boolean value) {
      checkBounds(i, j, k, l, m, n, o);
      _set(i, j, k, l, m, n, o, value);
    }

    /**
     * Convert the sidl array into a Java array.  This method will copy
     * the sidl array into the Java array.  The resulting Java array will
     * (obviously) start with a zero lower bound.  If the sidl array is
     * empty (null), then a null Java array will be returned.
     */
    public boolean[][][][][][][] toArray() {
      boolean[][][][][][][] array = null;
      if (!isNull()) {
        checkDimension(6);
        int l0 = _lower(0);
        int u0 = _upper(0);
        int l1 = _lower(1);
        int u1 = _upper(1);
        int l2 = _lower(2);
        int u2 = _upper(2);
        int l3 = _lower(3);
        int u3 = _upper(3);
	int l4 = _lower(4);
        int u4 = _upper(4);
	int l5 = _lower(5);
        int u5 = _upper(5);
	int l6 = _lower(6);
        int u6 = _upper(6);
        array = new boolean[u0-l0+1][][][][][][];
        for (int i = l0; i <= u0; i++) {
          array[i] = new boolean[u1-l1+1][][][][][];
          for (int j = l1; j <= u1; j++) {
            array[i][j] = new boolean[u2-l2+1][][][][];
            for (int k = l2; k <= u2; k++) {
              array[i][j][k] = new boolean[u3-l3+1][][][];
              for (int l = l3; l <= u3; l++) {
		array[i][j][k][l] = new boolean[u4-l4+1][][];
		for (int m = l4; m <= u4; m++) {
		  array[i][j][k][l][m] = new boolean[u4-l4+1][];
		  for (int n = l5; n <= u5; n++) {
		    array[i][j][k][l][m][n] = new boolean[u5-l5+1];
		    for (int o = l6; o <= u6; o++) {
		      array[i-l0][j-l1][k-l2][l-l3][m-l4][n-l5][o-l6] = _get(i, j, k, l, m, n, o);
			
		    }
		  }
		}
              }
            }
          }
        }
      }
      return array;
    }

    /**
     * Set the value of the sidl array from the Java array.  This method
     * will copy the Java array values into the sidl array, reallocating
     * the memory of the sidl array as necessary.  The resulting sidl array
     * will start with a zero lower bound.  If the Java array is null, then
     * the sidl array will be null, as well.  Note that multidimensional Java
     * arrays must not be ragged; that is, all sub-arrays in a particular
     * dimension must have the same size.  Otherwise, some data may be missed
     * or this method may throw an array index out of bounds exception.
     */
    public void fromArray(boolean[][][][][][][] array) {
      if (array == null) {
        destroy();
      } else {
        int s0 = array.length-1;
        int s1 = array[0].length-1;
        int s2 = array[0][0].length-1;
        int s3 = array[0][0][0].length-1;
	int s4 = array[0][0][0][0].length-1;
	int s5 = array[0][0][0][0][0].length-1;
	int s6 = array[0][0][0][0][0][0].length-1;
        reallocate(0, 0, 0, 0, 0, 0, 0, s0, s1, s2, s3, s4, s5, s6,
		    _isRowOrder());
        for (int i = 0; i <= s0; i++) {
          for (int j = 0; j <= s1; j++) {
            for (int k = 0; k <= s2; k++) {
              for (int l = 0; l <= s3; l++) {
		for (int m = 0; m <= s4; m++) { 
		  for (int n = 0; n <= s5; n++) {
		    for (int o = 0; o <= s6; o++) {
                     _set(i, j, k, l, m, n, o, array[i][j][k][l][m][n][o]);
		    }
		  }
		}
              }
            }
          }
        }
      }
    }

    public static class Holder {
      private Array7 d_obj;

    /**
     * Create a holder class with an empty holdee object.
     */
    public Holder() {
      d_obj = null;
    }

    /**
     * Create a holder with the specified object.
     */
    public Holder(Array7 obj) {
      d_obj = obj;
    }

    /**
     * Set the value of the holdee object.
     */
    public void set(Array7 obj) {
      d_obj = obj;
    }

    /**
     * Get the value of the holdee object.
     */
    public Array7 get() {
      return d_obj;
    }
  }

  }

}

