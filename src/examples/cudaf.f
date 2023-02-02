!     Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
!     HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

! This file contains the module defining Fortran interfaces for the CUDA Runtime API
! Copied and modified from:
!    https://raw.githubusercontent.com/starkiller-astro/XNet/master/source/cudaf.f90

      module cudaf
         ! Interface to CUDA Runtime API

         use, intrinsic :: iso_c_binding, only: c_int

         implicit none

         integer(c_int), parameter :: cudaMemAttachGlobal = 1
         integer(c_int), parameter :: cudaMemAttachHost = 2
         integer(c_int), parameter :: cudaMemAttachSingle = 4

         !include "cudaDeviceProp.fh"

         interface

            integer(c_int) function
     1              cudaMallocManaged(dPtr, size, flags)
     1              bind(c, name="cudaMallocManaged")
               use, intrinsic :: iso_c_binding
               type(c_ptr), intent(out) :: dPtr
               integer(c_size_t), value :: size
               integer(c_int), value :: flags
            end function cudaMallocManaged

            integer(c_int) function
     1              cudaFree(dPtr)
     1              bind(c, name="cudaFree")
               use, intrinsic :: iso_c_binding
               type(c_ptr), value :: dPtr
            end function cudaFree

         end interface

         contains

         ! wrapper functions

         integer function
     1           device_malloc_managed(nbytes, dPtr) result(stat)
            use, intrinsic :: iso_c_binding, only: c_size_t, c_ptr
            use, intrinsic :: iso_fortran_env, only: int64
            integer(int64), intent(in) :: nbytes
            type(c_ptr), intent(inout) :: dPtr
            stat = cudaMallocManaged(dPtr, int(nbytes,c_size_t),
     1                               cudaMemAttachGlobal)
         end function device_malloc_managed
         !
         integer function
     1           device_free(dPtr) result(stat)
            use, intrinsic :: iso_c_binding, only: c_ptr
            type(c_ptr), intent(inout) :: dPtr
            stat = cudaFree(dPtr)
         end function device_free

      end module cudaf
