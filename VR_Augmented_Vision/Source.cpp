#include <Windows.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>

#include "opencv2\core.hpp"
#include "opencv2\highgui.hpp"

#include "kernal.h"

using namespace cv;
using namespace std;

#define location "C:/Users/Manganese/Desktop/"

//write only
//the frame that is displayed to the user
Mat projected_frame;

#define NUMBER_OF_CAMERAS 6
VideoCapture input_videos[NUMBER_OF_CAMERAS];

//enum for each frame or camera in the frame_pointer or input_videos array
enum camera_namesE{
	top_frame = 0,
	bottom_frame = 1,
	front_frame = 2,
	left_frame = 3,
	right_frame = 4,
	back_frame = 5,
} camera_names;

typedef struct{
	HANDLE lock;//mutex to atomically update the pointer
	Mat* frame;//pointer to newest frame
} Frame_Pointer;
//array of pointers to each camera's frame
//each mat pointer is atomically updated to the new mat when a new frame comes in.
//eliminates case where mutex would make thread stop writing, which could lead to another frame, which has been updated, not be drawn
Frame_Pointer frame_array[NUMBER_OF_CAMERAS];

//combined width of input images
const unsigned int cubeFaceWidth = 1100; 
const unsigned int totalWidth = cubeFaceWidth * 4; //4 horizontal faces
const unsigned int cubeFaceHeight = 450;
const unsigned int totalHeight = cubeFaceHeight * 3; //3 vertical faces

//desired size of output image
const unsigned int screenWidth = 1920;
const unsigned int screenHeight = 1200;

typedef struct{
	unsigned int x;
	unsigned int y;
	unsigned int width;
	unsigned int height;
} Thread_Screen;
DWORD WINAPI Project_to_Screen(void* input);

#define NUM_THREADS 6

int main(int argc, char** argv)
{
	//zero all starting frames and set up a mutex
	projected_frame = Mat::zeros(totalHeight, totalWidth, CV_8UC3);//CV_[The number of bits per item][Signed or Unsigned][Type Prefix]C[The channel number]
	for (unsigned int i = 0; i < NUMBER_OF_CAMERAS; ++i){
		Mat frame = Mat::zeros(cubeFaceHeight, cubeFaceWidth, CV_8UC3);
		frame_array[i].frame = &frame;//TODO smart pointer get deallocd? what am i doing wrong?
		frame_array[i].lock = CreateMutex(NULL,FALSE,NULL);
	}

	input_videos[top_frame].open(location"top.mp4");
	input_videos[bottom_frame].open(location"bottom.mp4");
	input_videos[front_frame].open(location"front.mp4");
	input_videos[left_frame].open(location"left.mp4");
	input_videos[right_frame].open(location"right.mp4");
	input_videos[back_frame].open(location"back.mp4");
	namedWindow("");

	//start n threads, to cover the entire screen area
	unsigned int per_thread_width = screenWidth / (NUM_THREADS/2);
	unsigned int per_thread_height = screenHeight / (NUM_THREADS/2);
	int x_offset = 0;
	int y_offset = 0;
	for (unsigned int i = 0; i < NUM_THREADS; ++i){
		Thread_Screen* thread_area = (Thread_Screen*)malloc(sizeof(Thread_Screen));
		thread_area->height = per_thread_height;
		thread_area->width = per_thread_width;
		thread_area->x = x_offset*per_thread_width;
		thread_area->y = y_offset*per_thread_height;

		CreateThread(
			NULL,                   // default security attributes
			0,                      // use default stack size  
			Project_to_Screen,       // thread function name
			thread_area,          // argument to thread function 
			0,                      // use default creation flags 
			NULL);   // returns the thread identifier 

		x_offset++;
		if (x_offset == NUM_THREADS/2)
			y_offset++;
	}

	while (1){
		//read any new frames from the cameras
		for (unsigned int i = 0; i < NUMBER_OF_CAMERAS; ++i){
			//if there is a new frame for a camera, get it
			//http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-grab
			if (input_videos[i].grab()){
				Mat next_frame;
				input_videos[i].retrieve(next_frame);
				//update the current frame's pointer to this new frame
				DWORD locked = WaitForSingleObject(frame_array[i].lock, 10);//10 ms timeout
				if (locked == WAIT_OBJECT_0){
					frame_array[i].frame = &next_frame;
				}
				ReleaseMutex(frame_array[i].lock);
			}
		}

		imshow("", projected_frame);
		waitKey(1);
		//imwrite(location"out.png", projected_frame);
	}

	return 0;
}

DWORD WINAPI Project_to_Screen(void* input){
	Thread_Screen* area = (Thread_Screen*)input;
	while (1){
		//http://stackoverflow.com/questions/34250742/converting-a-cubemap-into-equirectangular-panorama
		//inverse mapping
		//can do better with gpu shaders

		float u, v; //Normalised texture coordinates, from 0 to 1, starting at lower left corner
		float phi, theta; //Polar coordinates

		for (int j = area->y; j < area->y+area->height; j++)
		{
			//Rows start from the bottom
			v = 1 - ((float)j / totalHeight);
			theta = v * M_PI;

			for (int i = area->x; i < area->x+area->width; i++)//go along columns in inner loop for speed.
			{
				//convert x,y cartesian to u,v polar

				//Columns start from the left
				u = ((float)i / totalWidth);
				phi = u * 2 * M_PI;

				//convert polar to 3d vector
				float x, y, z; //Unit vector
				x = sin(phi) * sin(theta) * -1;
				y = cos(theta);
				z = cos(phi) * sin(theta) * -1;

				float xa, ya, za;
				float a;

				a = fmaxf(fmaxf(abs(x), abs(y)), abs(z));

				//Vector Parallel to the unit vector that lies on one of the cube faces
				xa = x / a;
				ya = y / a;
				za = z / a;

				Vec3b pixel;
				int xPixel, yPixel;
				int xOffset, yOffset;

				if (xa == 1)
				{
					//Right
					xPixel = (int)((((za + 1.0) / 2.0) - 1.0) * cubeFaceWidth);
					yPixel = (int)((((ya + 1.0) / 2.0)) * cubeFaceHeight);

					DWORD locked = WaitForSingleObject(frame_array[right_frame].lock, 10);
					if (locked == WAIT_OBJECT_0){
						pixel = (*frame_array[right_frame].frame).at<Vec3b>(abs(yPixel), abs(xPixel));
					}
					ReleaseMutex(frame_array[right_frame].lock);
				}
				else if (xa == -1)
				{
					//Left
					xPixel = (int)((((za + 1.0) / 2.0)) * cubeFaceWidth);
					yPixel = (int)((((ya + 1.0) / 2.0)) * cubeFaceHeight);

					DWORD locked = WaitForSingleObject(frame_array[left_frame].lock, 10);
					if (locked == WAIT_OBJECT_0){
						pixel = (*frame_array[left_frame].frame).at<Vec3b>(abs(yPixel), abs(xPixel));
					}
					ReleaseMutex(frame_array[left_frame].lock);
				}
				else if (ya == 1)
				{
					//Up
					xPixel = (int)((((xa + 1.0) / 2.0)) * cubeFaceWidth);
					yPixel = (int)((((za + 1.0) / 2.0) - 1.0) * cubeFaceHeight);

					DWORD locked = WaitForSingleObject(frame_array[top_frame].lock, 10);
					if (locked == WAIT_OBJECT_0){
						pixel = (*frame_array[top_frame].frame).at<Vec3b>(abs(yPixel), abs(xPixel));
					}
					ReleaseMutex(frame_array[top_frame].lock);
				}
				else if (ya == -1)
				{
					//Down
					xPixel = (int)((((xa + 1.0) / 2.0)) * cubeFaceWidth);
					yPixel = (int)((((za + 1.0) / 2.0)) * cubeFaceHeight);

					DWORD locked = WaitForSingleObject(frame_array[bottom_frame].lock, 10);
					if (locked == WAIT_OBJECT_0){
						pixel = (*frame_array[bottom_frame].frame).at<Vec3b>(abs(yPixel), abs(xPixel));
					}
					ReleaseMutex(frame_array[bottom_frame].lock);
				}
				else if (za == 1)
				{
					//Front
					xPixel = (int)((((xa + 1.0) / 2.0)) * cubeFaceWidth);
					yPixel = (int)((((ya + 1.0) / 2.0)) * cubeFaceHeight);

					DWORD locked = WaitForSingleObject(frame_array[front_frame].lock, 10);
					if (locked == WAIT_OBJECT_0){
						pixel = (*frame_array[front_frame].frame).at<Vec3b>(abs(yPixel), abs(xPixel));
					}
					ReleaseMutex(frame_array[front_frame].lock);
				}
				else if (za == -1)
				{
					//Back
					xPixel = (int)((((xa + 1.0) / 2.0) - 1.0) * cubeFaceWidth);
					yPixel = (int)((((ya + 1.0) / 2.0)) * cubeFaceHeight);

					DWORD locked = WaitForSingleObject(frame_array[back_frame].lock, 10);
					if (locked == WAIT_OBJECT_0){
						pixel = (*frame_array[back_frame].frame).at<Vec3b>(abs(yPixel), abs(xPixel));
					}
					ReleaseMutex(frame_array[back_frame].lock);
				}
				else
				{
					printf("Unknown face, something went wrong");
				}

				projected_frame.at<Vec3b>(j, i) = pixel;
			}
		}
	}
}