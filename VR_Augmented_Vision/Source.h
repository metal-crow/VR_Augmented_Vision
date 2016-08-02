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

const unsigned char NUMBER_OF_VIEWPOINTS = 6; //should never be more than 256 viewpoints

//combined width of input images
const unsigned int CUBE_FACE_WIDTH = 1920;
const unsigned int TOTAL_WIDTH = CUBE_FACE_WIDTH * 4; //4 horizontal faces
const unsigned int CUBE_FACE_HEIGHT = 1080;
const unsigned int TOTAL_HEIGHT = CUBE_FACE_HEIGHT * 3; //3 vertical faces

//desired size of output image
const unsigned int SCREEN_WIDTH = 1080;
const unsigned int SCREEN_HEIGHT = 1200;//TODO: since we want to cut of the tops of the poles, makes this slightly higher than actual, then crop


/*-----Structs and Enums----*/

//enum for each ViewPoint
enum viewpoint_names{
	top_view = 0,
	bottom_view = 1,
	front_view = 2,
	left_view = 3,
	right_view = 4,
	back_view = 5,
};

typedef struct{
	Mat* frame_0;//frames used for frame buffer
	Mat* frame_1;
	unsigned char selected_frame;//frame current in use from buffer
} Frame_Pointer;

typedef struct{
	Frame_Pointer left;
	Frame_Pointer right;
} ViewPoint_Frame_Pointer;

//Handle each set of cameras, called a viewpoint. left and right eye cameras at a specific position.
typedef struct{
	VideoCapture left;
	VideoCapture right;
} ViewPoint;

typedef struct{
	Mat left;
	Mat right;
} Projected_Frame;

/*-----Global Variable declarations (defined in c file)------*/

//the image that is displayed to the user
extern Projected_Frame projected_frame;

//the set of input viewpoints
extern ViewPoint input_views[NUMBER_OF_VIEWPOINTS];

//array of pointers to each camera's frame
//each mat pointer is atomically updated to the new mat when a new frame comes in.
//eliminates case where mutex would make thread stop writing, which could lead to another frame, which has been updated, not be drawn
extern ViewPoint_Frame_Pointer viewpoint_frame_array[NUMBER_OF_VIEWPOINTS];

#endif