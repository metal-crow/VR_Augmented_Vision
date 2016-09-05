#ifndef SOURCE_H
#define SOURCE_H

#include <Windows.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>

#include "opencv2\core.hpp"
#include "opencv2\highgui.hpp"

#include "VRdisplay.h"

using namespace cv;
using namespace std;

/*-----Contants and settings------*/

#define GPU 1 //0 for CPU only, 1 for GPU
#define NUM_THREADS 7 //this specifies the number of thread this program can use
#define USE_VR 1

#define DEBUG_TIME 1

#define NUMBER_OF_VIEWPOINTS 6 //this is a constant 6, since we're formatting this as a cube map.

//resolution of each cube face (each side's combined cameras' resolutions)
const unsigned int CUBE_FACE_WIDTH = 1536;
const unsigned int CUBE_FACE_HEIGHT = 1536;

//desired size of output image
const unsigned int SCREEN_WIDTH = 1080;
const unsigned int SCREEN_HEIGHT = 1200;//TODO: since we want to cut of the tops of the poles, makes this slightly higher than actual, then crop


/*-----Structs and Enums----*/

//enum for each cube face
enum cubeface_names{
	top_view = 0,
	bottom_view = 1,
	front_view = 2,
	left_view = 3,
	right_view = 4,
	back_view = 5,
};

//THe part of a cube face a camera composes. A slice
typedef struct{
	cubeface_names cube_face;//the cube face this is a slice of
	unsigned int slice_width;
	unsigned int slice_height;
	unsigned int slice_loc_in_view_x;//x offset from left of cube face
	unsigned int slice_loc_in_view_y;//y offset from top of cube face
} CubeFace_Slice;

//Each camera (left and right are same camera to simplify). Composes multiple slices of a single cube face
typedef struct{
	VideoCapture cam_left;
	VideoCapture cam_right;
	unsigned char number_of_cube_faces;//the number of cube faces this camera is a slice of
	CubeFace_Slice* slices_of;//an array of the cube faces this camera is sliced into
} Camera_View;

typedef struct{
	Mat left;
	Mat right;
} Frame;

/*-----Global Variable declarations (defined in c file)------*/

//the image that is displayed to the user
extern Frame projected_frame;

//the image frames that make up the cube faces
extern Frame cube_faces[NUMBER_OF_VIEWPOINTS];

//the input cameras that are used to populate the cube_faces
extern unsigned char number_of_cameras;
extern Camera_View cameras[10];//TODO make this dynamic

#endif