##VR Augmented Vision
For centuries, humans have desired to augment their biological abilities. The plane, for flight. The car, for speed. And now...
the ability to see in 360 degrees, horizontal and vertical, and in infared.

I thought it'd be cool.

Using a helmet rig of gopro knockoffs and an oculus rift, i construct a 3d sphere-view from the camera inputs, and then project that only a 2d image.  
This image is then written to the oculus's display, allowng 360 degree vision.

Does it work?  
yes!

Does it work in PRACTICE?  
I dunno, its not finished.

  
###TODO:  
Get test 360 steriocopic videos (calibration with squares for shape, and real videos)    

fix issue with when you move your head, oculus has to snap it/or the 2d quad back, instead of just ignoring any movement  
-want the quad to be in a fixed position, and head movement/rotation to be ignored  
  
Should the rendering have speperate pipelines for left and right eyes? i.e if 1 eye comes in but not the other, only have to copy that projection back from gpu, and set that projection to the texture  