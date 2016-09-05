<img src="https://s-media-cache-ak0.pinimg.com/736x/51/0d/f9/510df94355c8b3eee63c27b2b1307d94.jpg" height="150">

For centuries, humans have desired to augment their biological abilities. The plane, for flight. The car, for speed. And now...
the ability to see in 360 degrees, horizontal and vertical, and in infared.

I thought it'd be cool.

Using a helmet rig of gopro knockoffs and an oculus rift, i construct a 3d sphere-view from the camera inputs, and then project that only a 2d image.  
This image is then written to the oculus's display, allowng 360 degree vision.

####Does it work?
yes!

####Does it work in PRACTICE?
I dunno, its not finished.

####Greater than 6 cam setup
Since most cameras don't have the 180 vertical and horizontal fov requires to generate a cubemap, you'll usually want to use more cameras and combine their images to get a higher fov.  
This is done by hvaing each viewpoint have more than 1 camera accociated with it.  
When each camera in a viewpoint is read, the image is slid into the total output image for this cube's face, as a slice of the entire side.  
The only speed loss from this teqnique is the movement of the image (a slice) into the main face's image.  
  
###TODO:  
Get test 360 steriocopic videos (calibration with squares for shape, and real videos)    

fix issue with when you move your head, oculus has to snap it/or the 2d quad back, instead of just ignoring any movement  
-want the quad to be in a fixed position, and head movement/rotation to be ignored  
  
build hardware  
-need to select camera type  
-figure out housing and construction  
-can the cameras view infared?  
  
Should the rendering have speperate pipelines for left and right eyes? i.e if 1 eye comes in but not the other, only have to copy that projection back from gpu, and set that projection to the texture  