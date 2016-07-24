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


###NOTES
CPU can, in theory, have less latency than GPU due to the GPU needing to be syncronized with the main thread (load images, call gpu, wait), instead of the asyncroncous cpu (simultainous and ongoing image loading and converting)  
However GPU has less latency most of the time because its much faster at converting the image  
  
TODO:  
fix issue with when you move your head, oculus has to snap it/or the 2d quad back, instead of just ignoring any movement  
-want the quad to be in a fixed position, and head movement/rotation to be ignored  