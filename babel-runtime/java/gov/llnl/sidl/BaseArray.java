//
// File:	BaseArray.java
// Package:	gov.llnl.sidl
// Release:	$Name$
// Revision:	$Revision$
// Modified:	$Date$
// Description:	base array for the sidl Java runtime system
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

package gov.llnl.sidl;

/**
 * Class <code>BaseArray</code> is the base array for all sidl Java arrays
 * in the run-time system.  This class provides basic support for bounds
 * checking and management of the IOR array pointer object.
 */
public abstract class BaseArray {
  protected long d_array;
  protected boolean d_owner;

  /**
   * Construct an empty array object.  This object must be allocated before
   * any actions are performed on the array data.
   */
  protected BaseArray() {
    d_array = 0;
    d_owner = true;
  }

  /**
   * Create an array using an IOR array pointer.  The pointer value may be
   * zero (representing null).  If the owner flag is true, then the array
   * will be deallocated when this object disappears.
   */
  protected BaseArray(long array, boolean owner) {
    d_array = array;
    d_owner = owner;
  }

  /**
   * Destroy existing array data (if present and owner) and assign the
   * new array pointer and owner.
   */
  protected void reset(long array, boolean owner) {
    destroy();
    d_array = array;
    d_owner = owner;
  }

  /**
   * Check whether the array referenced by this object is null.
   */
  public boolean isNull() {
    return d_array == 0;
  }

  /**
   * Abstract method to get the dimension of the array.  This method
   * will be implemented in subclasses as a native method.
   */
  public abstract int _dim();

  /**
   * Abstract method to fetch the specified lower bound of the array.
   * This method will be implemented in subclasses as a native method.
   * The specified array dimension must be between zero and the array
   * dimension minus one.  Invalid values will have unpredictable (but
   * almost certainly bad) results.
   */
  public abstract int _lower(int dim);

  /**
   * Abstract method to fetch the specified upper bound of the array.
   * This method will be implemented in subclasses as a native method.
   * The specified array dimension must be between zero and the array
   * dimension minus one.  Invalid values will have unpredictable (but
   * almost certainly bad) results.
   */
  public abstract int _upper(int dim);

  /**
   * Abstract method to fetch the stride of the specified dimen of the array.
   * This method will be implemented in subclasses as a native method.
   * The specified array dimension must be between zero and the array
   * dimension minus one.  Invalid values will have unpredictable (but
   * almost certainly bad) results.
   */
  public abstract int _stride(int dim);

  /**
   * Abstract method returns true if array is ColumnOrder.
   * This method will be implemented in subclasses as a native method.
   */
  public abstract boolean _isColumnOrder();

  /**
   * Abstract method returns true if array if RowOrder.
   * This method will be implemented in subclasses as a native method.
   */
  public abstract boolean _isRowOrder();

  /** 
   * Abstract method adds 1 to array's reference count.  Not for users   
   * but for internal babel stuff.
   */
  public abstract void _addRef();


  /** Deallocate deletes java's reference to the array (calls deleteRef)
   *  But does not (nessecarily) case the array to be GCed.
   */
  public abstract void _deallocate();

    
  /**
   * Abstract method to destroy the array.  This will be called by the
   * object finalizer if the array is owned by this object.
   */
  protected abstract void _destroy();

  /**
   * Abstract method to reallocate array data using the specified dimension,
   * lower bounds, and upper bounds.  This routine assumes that the dimension
   * and indices are valid.
   */
  protected abstract void _reallocate(int dim, int[] lower, int[] upper, boolean isRow);
 
  /**
   * Destroy the existing array and make it null.  This method deallocates
   * the IOR array reference if we are the owner and the reference is not
   * null.  The new array reference is null.
   */
  public void destroy() {
    if (d_owner && (d_array != 0)) {
      _destroy();
    }
    d_array = 0;
    d_owner = true;
  }
  
  /**
   *  Return the pointer to the implementation of the Array (A special 
   *  function for Object arrays, No touchie!
   *
   */
  public long get_ior_pointer() {
    return d_array;
  }
  
  /**
   *  Set the pointer to the implementation of the Array (A special 
   *  function for Object arrays, No touchie!
   *
   */
  public void set_ior_pointer(long p) {
    d_array = p;
  }
  
  /**
   *  Return the array owner flag (A special  function for Object arrays, No touchie!
   */
  public boolean get_owner() {
    return d_owner;
  }

  /**
   *  Return the array owner flag (A special  function for Object arrays, No touchie!
   */
  public void set_owner(boolean p) {
    d_owner = p;
  }


  /**
   * The finalizer of this object deallocates the IOR array reference if
   * we are the owner and the referece is not null.
   */
  protected void finalize() throws Throwable {
    destroy();
  }

  /**
   * Reallocate array data using the specified dimension and lower and
   * upper bounds.  Old array data is deleted.  Each of the lower and upper
   * bounds arrays must contain <code>dim</code> elements.  Upper array
   * bounds are inclusive.  An array index out of bounds exception is thrown
   * if any of the indices are invalid.
   */
  public void reallocate(int dim, int[] lower, int[] upper, boolean isRow) {
      destroy();
    if ((lower == null) || (upper == null)) {
      throw new ArrayIndexOutOfBoundsException("Null array index argument");
    }
    if ((dim > lower.length) || (dim > upper.length)) {
      throw new ArrayIndexOutOfBoundsException("Array dimension mismatch");
    }
    for (int d = 0; d < dim; d++) {
      if (upper[d]+1 < lower[d]) { //An array with 0, -1 as bounds is an empty array.
        throw new ArrayIndexOutOfBoundsException("Upper bound less than lower");
      }
    }
    _reallocate(dim, lower, upper, isRow);
  }

  /**
   * Return the dimension of the array.  If the array is null, then the
   * dimension is zero.
   */
  public int dim() {
    return isNull() ? 0 : _dim();
  }

  /**
   * Throw a <code>NullPointerException</code> if the array is null.
   */
  protected void checkNullArray() {
    if (isNull()) {
      throw new NullPointerException("Array data has not been allocated");
    }
  }

  /**
   * Check that the array is equal to the specified rank.  If the array
   * ranks do not match, an <code>ArrayIndexOutOfBoundsException</code>
   * is thrown.  This routine assumes that the array is not null.
   */
  protected void checkDimension(int d) {
    if (d != _dim()) {
      throw new ArrayIndexOutOfBoundsException("Illegal array dimension : "+d);
    }
  }

  /**
   * Check that the index is valid for the specified dimension.  An
   * <code>ArrayIndexOutOfBoundsException</code> is thrown if the index
   * is out of bounds.  This routine assumes both that the array pointer
   * is not null and that the dimension argument is valid.
   */
  protected void checkIndexBounds(int i, int d) {
    if ((i < _lower(d)) || (i > _upper(d))) {
      throw new ArrayIndexOutOfBoundsException("Index "+d+" out of bounds: "+i);
    }
  }

  /**
   * Check that the index is valid for the array.  A null pointer exception
   * is thrown if the arry is null.  An array index out of bounds exception
   * is thrown if the index is out of bounds.
   */
  protected void checkBounds(int i) {
    checkNullArray();
    checkDimension(1);
    checkIndexBounds(i, 0);
  }

  /**
   * Check that the indices are valid for the array.  A null pointer
   * exception is thrown if the arry is null.  An array index out of
   * bounds exception is thrown if the index is out of bounds.
   */
  protected void checkBounds(int i, int j) {
    checkNullArray();
    checkDimension(2);
    checkIndexBounds(i, 0);
    checkIndexBounds(j, 1);
  }

  /**
   * Check that the indices are valid for the array.  A null pointer
   * exception is thrown if the arry is null.  An array index out of
   * bounds exception is thrown if the index is out of bounds.
   */
  protected void checkBounds(int i, int j, int k) {
    checkNullArray();
    checkDimension(3);
    checkIndexBounds(i, 0);
    checkIndexBounds(j, 1);
    checkIndexBounds(k, 2);
  }

  /**
   * Check that the indices are valid for the array.  A null pointer
   * exception is thrown if the arry is null.  An array index out of
   * bounds exception is thrown if the index is out of bounds.
   */
  protected void checkBounds(int i, int j, int k, int l) {
    checkNullArray();
    checkDimension(4);
    checkIndexBounds(i, 0);
    checkIndexBounds(j, 1);
    checkIndexBounds(k, 2);
    checkIndexBounds(l, 3);
  }


/**
   * Check that the indices are valid for the array.  A null pointer
   * exception is thrown if the arry is null.  An array index out of
   * bounds exception is thrown if the index is out of bounds.
   */
  protected void checkBounds(int i, int j, int k, int l, int m) {
    checkNullArray();
    checkDimension(5);
    checkIndexBounds(i, 0);
    checkIndexBounds(j, 1);
    checkIndexBounds(k, 2);
    checkIndexBounds(l, 3);
    checkIndexBounds(m, 4);
  }

/**
   * Check that the indices are valid for the array.  A null pointer
   * exception is thrown if the arry is null.  An array index out of
   * bounds exception is thrown if the index is out of bounds.
   */
  protected void checkBounds(int i, int j, int k, int l, int m, int n) {
    checkNullArray();
    checkDimension(6);
    checkIndexBounds(i, 0);
    checkIndexBounds(j, 1);
    checkIndexBounds(k, 2);
    checkIndexBounds(l, 3);
    checkIndexBounds(m, 4);
    checkIndexBounds(n, 5);
  }

/**
   * Check that the indices are valid for the array.  A null pointer
   * exception is thrown if the arry is null.  An array index out of
   * bounds exception is thrown if the index is out of bounds.
   */
  protected void checkBounds(int i, int j, int k, int l, int m, int n, int o) {
    checkNullArray();
    checkDimension(7);
    checkIndexBounds(i, 0);
    checkIndexBounds(j, 1);
    checkIndexBounds(k, 2);
    checkIndexBounds(l, 3);
    checkIndexBounds(m, 4);
    checkIndexBounds(n, 5);
    checkIndexBounds(o, 6);
  }

  /**
   * Return the lower index of the array corresponding to the specified
   * array dimension.  This routine will throw a null pointer exception
   * if the object is null or an array index out of bounds exception if
   * the specified array dimension is not valid.
   */
  public int lower(int dim) {
    checkNullArray();
    return _lower(dim);
  }

  /**
   * Return the upper index of the array corresponding to the specified
   * array dimension.  The array runs from the lower bound to the upper
   * bound, inclusive.  This routine will throw a null pointer exception
   * if the object is null or an array index out of bounds exception if
   * the specified array dimension is not valid.
   */
  public int upper(int dim) {
    checkNullArray();
    return _upper(dim);
  }

  /**
   * Return the stride of the array corresponding to the specified
   * array dimension.  This routine will throw a null pointer exception
   * if the object is null or an array index out of bounds exception if
   * the specified array dimension is not valid.
   */
  public int stride(int dim) {
    checkNullArray();
    return _stride(dim);
  }

} 
