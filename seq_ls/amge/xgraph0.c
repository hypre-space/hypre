#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <stdio.h>
#include <ctype.h>

Window          myWindow;
Pixmap          myPixmap;
Display         *myDisplay;
Screen          *myScreen;
unsigned long   ColorMapPixels[256];
int             ColorStatus[256];
int             scale;

/**********************************************************************/
/*                                                                    */
/*          piecewise linear triangle multi grid package              */
/*                                                                    */
/*                  edition 7.1 - - - june, 1994                      */
/*                                                                    */
/**********************************************************************/
void xutl0_(int *ncolor, float red[], float  green[], float blue[] )
    {
        int winX, winY, winW, winH, bwidth, i, done;
        unsigned long valuemask;
        XSetWindowAttributes xswa;
        XEvent myEvent;
        KeySym myKeysym;
        XColor color;
        char text[10];

        if ( *ncolor > 0 ) {
            myDisplay= XOpenDisplay("");              
            myScreen = DefaultScreenOfDisplay(myDisplay);
            winX = XWidthOfScreen(myScreen);
            winY = XHeightOfScreen(myScreen);
	    /*            scale =((( winX*2 < winY*3 ) ? (winX*2)/3 : winY )*95)/100 ;*/
            scale =((( winX*2 < winY*3 ) ? (winX*2)/3 : winY )*60)/100 ;
            winW = (scale * 3)/2 ;
            winH = scale ;
            winX = (winX - winW) / 2 ;
            winY = (winY - winH) / 2 ;
            bwidth=2;
            xswa.event_mask = ExposureMask | KeyPressMask |
                StructureNotifyMask | ButtonPressMask;
            xswa.backing_store = Always;
            xswa.background_pixel = WhitePixelOfScreen(myScreen);
            valuemask = CWBackingStore | CWBackPixel | CWEventMask;
            myWindow = XCreateWindow(myDisplay, RootWindowOfScreen(myScreen),
                 winX, winY, winW, winH, bwidth, 
                 DefaultDepthOfScreen(myScreen), InputOutput, 
                 DefaultVisualOfScreen(myScreen), valuemask, &xswa);
            XChangeProperty(myDisplay, myWindow, XA_WM_NAME, XA_STRING, 8,
                PropModeReplace, "AMGe edition 0.0", 17);
            myPixmap = XCreatePixmap (myDisplay, myWindow, winW, winH,
                DefaultDepthOfScreen(myScreen));
            XSetForeground(myDisplay,DefaultGCOfScreen(myScreen),
                WhitePixelOfScreen(myScreen));
            XFillRectangle(myDisplay,myPixmap,DefaultGCOfScreen(myScreen),
                 0,0,winW,winH);
            for (i = 0; i < *ncolor; i++) {
                color.red = (unsigned short) ( red[i] * 65535 );
                color.green = (unsigned short) ( green[i] * 65535 );
                color.blue = (unsigned short) ( blue[i] * 65535 );
                color.flags = DoRed | DoGreen | DoBlue;
                ColorStatus[i] =  XAllocColor(myDisplay, 
                          DefaultColormapOfScreen(myScreen), 
                          &color );
                if ( ColorStatus[i] == 0 ) {
                    ColorMapPixels[i] = 
                        WhitePixelOfScreen(myScreen);   
                } else {
                    ColorMapPixels[i] = color.pixel;
                } 
            }  
            XMapRaised(myDisplay, myWindow);
            done = 0;
            while ( done == 0 ) {
                XNextEvent( myDisplay, &myEvent );
                switch (myEvent.type) {
                    case MapNotify:
                        done = 1;
                        break;
                    default:
                        break;
                }
            }
	    /*          XSync(myDisplay,1);                   may not be necessary */
	    /*          sleep(1);                       may not be necessary */
	}
	else {
            done = 0;
            while ( done == 0 ) {
                XNextEvent( myDisplay, &myEvent );
                switch (myEvent.type) {
                    case ButtonPress:
                        i = myEvent.xbutton.button;
                        if ( i == 2 || i == 3  ) done = 1;
                        break;
                    case KeyPress:
                        i = XLookupString ( &myEvent.xkey, text, 10, 
                            &myKeysym, 0 );
                        if( i == 1 && text[0] == 'q' ) done = 1;
                        break;
                    case Expose:
                        XCopyArea( myDisplay, myPixmap, myWindow,
                             DefaultGCOfScreen(myScreen),
                             0, 0, winW, winH, 0, 0 );
                        break;
                    default:
                        break;
                }
            }
            XFreePixmap( myDisplay, myPixmap );
            XCloseDisplay(myDisplay);
        }
    }
/**********************************************************************/
/*                                                                    */
/*            piecewise linear triangle multi grid package            */
/*                                                                    */
/*                  edition 8.1 - - - february, 1999                 */
/*                                                                    */
/**********************************************************************/
void xline0_( float x[], float y[], int *np, int *icolor)
{
        int i;
        XPoint *pp, *pts;

        pts = (XPoint *) malloc( *np * sizeof(XPoint) );
        pp = pts;

        for(i = 0; i < *np; i++)  {
                pp->x = (int) ( x[i] * scale);
                pp->y = (int) ( ( 1.- y[i] ) * scale );
                pp++ ;
        }
        XSetForeground(myDisplay, 
                DefaultGCOfScreen(myScreen), 
                ColorMapPixels[ *icolor - 1 ] );  
        XDrawLines( myDisplay, myWindow, 
                DefaultGCOfScreen(myScreen),
                pts, *np, CoordModeOrigin );
        XDrawLines( myDisplay, myPixmap, 
                DefaultGCOfScreen(myScreen),
                pts, *np, CoordModeOrigin );
        free(pts);
} 

/**********************************************************************/
/*                                                                    */
/*            piecewise linear triangle multi grid package            */
/*                                                                    */
/*                  edition 8.1 - - - february, 1999                 */
/*                                                                    */
/**********************************************************************/
void xfill0_( float x[], float y[], int *np, int *icolor)
{
        int i;
        XPoint *pp, *pts;

        pts = (XPoint *) malloc( ( *np + 1) * sizeof(XPoint) );
        pp = pts;

        for(i = 0; i < *np; i++)  {
                pp->x = (int) ( x[i] * scale );
                pp->y = (int) ( ( 1.- y[i] ) * scale );
                pp++ ;
        }
        pp->x = (int) ( x[0] * scale );
        pp->y = (int) ( ( 1.- y[0] ) * scale );

        XSetForeground(myDisplay, 
                DefaultGCOfScreen(myScreen), 
                ColorMapPixels[ *icolor - 1 ] );  
        XFillPolygon( myDisplay, myWindow, 
                DefaultGCOfScreen(myScreen),
                pts, *np, Nonconvex, CoordModeOrigin );
        XFillPolygon( myDisplay, myPixmap, 
                DefaultGCOfScreen(myScreen),
                pts, *np, Nonconvex, CoordModeOrigin );

/*      draw boundary to get rid of possible white splotches     */
  
        XDrawLines( myDisplay, myWindow, 
                DefaultGCOfScreen(myScreen),
                pts, *np + 1 , CoordModeOrigin );
        XDrawLines( myDisplay, myPixmap, 
                DefaultGCOfScreen(myScreen),
                pts, *np + 1 , CoordModeOrigin );
  
        free(pts);
} 











