!     Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
!     HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

! This file contains the module defining Fortran interfaces for the CUDA Runtime API
! Copied and modified from:
!    https://raw.githubusercontent.com/starkiller-astro/XNet/master/source/cudaf.f90

      module cudaf
         ! Interface to CUDA Runtime API

         integer, parameter :: cudaMemAttachGlobal = 1
         integer, parameter :: cudaMemAttachHost = 2
         integer, parameter :: cudaMemAttachSingle = 4

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

      end module cudaf

