      double precision function dsign(a,b)
      real*8 a,b

      if (b .ge. 0.e0)       then
      dsign = dabs(a)
      return
      endif


      if (b .lt. 0.e0) then
         dsign =-dabs(a)
             return
      endif

      end 

         
