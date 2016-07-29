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

const unsigned char NUMBER_OF_CAMERAS = 6; //should never be more than 255 cameras

//combined width of input images
const unsigned int CUBE_FACE_WIDTH = 1100;
const unsigned int TOTAL_WIDTH = CUBE_FACE_WIDTH * 4; //4 horizontal faces
const unsigned int CUBE_FACE_HEIGHT = 450;
const unsigned int TOTAL_HEIGHT = CUBE_FACE_HEIGHT * 3; //3 vertical faces

//desired size of output image
const unsigned int SCREEN_WIDTH = 1920;
const unsigned int SCREEN_HEIGHT = 1200;//NOTE: since we want to cut of the tops of the poles, makes this slightly higher than actual, then crop


/*-----Structs and Enums----*/

//enum for each frame or camera in the frame_pointer or input_videos array
enum camera_names{
	top_frame = 0,
	bottom_frame = 1,
	front_frame = 2,
	left_frame = 3,
	right_frame = 4,
	back_frame = 5,
};

typedef struct{
	Mat* frame_0;//frames used for frame buffer
	Mat* frame_1;
	unsigned char selected_frame;//frame current in use from buffer
} Frame_Pointer;


/*-----Global Variable declarations (defined in c file)------*/

//write only
//the frame that is displayed to the user
extern Mat projected_frame;

//opencv's array of input videos
extern VideoCapture input_videos[NUMBER_OF_CAMERAS];

//array of pointers to each camera's frame
//each mat pointer is atomically updated to the new mat when a new frame comes in.
//eliminates case where mutex would make thread stop writing, which could lead to another frame, which has been updated, not be drawn
extern Frame_Pointer frame_array[NUMBER_OF_CAMERAS];

#endif