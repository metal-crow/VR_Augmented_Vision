#include <Windows.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>

#include "opencv2\core.hpp"
#include "opencv2\highgui.hpp"

using namespace cv;
using namespace std;

#define location "C:/Users/Manganese/Desktop/"

//write only
Mat projected_frame;
//read only
//TODO should i have a pointer to an old frame, then only when the new frame is created, swap out the pointer?
//eliminates case where mutex would make thread stop writing, which could lead to another frame, which has been updated, not be drawn
Mat top_frame;
Mat bottom_frame;
Mat front_frame;
Mat left_frame;
Mat right_frame;
Mat back_frame;

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
	//zero all starting frames
	projected_frame = Mat::zeros(totalHeight, totalWidth, CV_8UC3);//CV_[The number of bits per item][Signed or Unsigned][Type Prefix]C[The channel number]
	top_frame = Mat::zeros(cubeFaceHeight, cubeFaceWidth, CV_8UC3);
	bottom_frame = Mat::zeros(cubeFaceHeight, cubeFaceWidth, CV_8UC3);
	front_frame = Mat::zeros(cubeFaceHeight, cubeFaceWidth, CV_8UC3);
	left_frame = Mat::zeros(cubeFaceHeight, cubeFaceWidth, CV_8UC3);
	right_frame = Mat::zeros(cubeFaceHeight, cubeFaceWidth, CV_8UC3);
	back_frame = Mat::zeros(cubeFaceHeight, cubeFaceWidth, CV_8UC3);

	VideoCapture top(location"top.mp4");
	VideoCapture bottom(location"bottom.mp4");
	VideoCapture front(location"front.mp4");
	VideoCapture left(location"left.mp4");
	VideoCapture right(location"right.mp4");
	VideoCapture back(location"back.mp4");
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
		//if there is a new frame for a camera, get it
		//http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-grab
		//TODO see link
		if (top.grab())
			top.retrieve(top_frame);
		if (bottom.grab())
			bottom.retrieve(bottom_frame);
		if (front.grab())
			front.retrieve(front_frame);
		if (left.grab())
			left.retrieve(left_frame);
		if (right.grab())
			right.retrieve(right_frame);
		if (back.grab())
			back.retrieve(back_frame);

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

			for (int i = area->x; i < area->x+area->width; i++)
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

					pixel = right_frame.at<Vec3b>(abs(yPixel), abs(xPixel));
				}
				else if (xa == -1)
				{
					//Left
					xPixel = (int)((((za + 1.0) / 2.0)) * cubeFaceWidth);
					yPixel = (int)((((ya + 1.0) / 2.0)) * cubeFaceHeight);

					pixel = left_frame.at<Vec3b>(abs(yPixel), abs(xPixel));
				}
				else if (ya == 1)
				{
					//Up
					xPixel = (int)((((xa + 1.0) / 2.0)) * cubeFaceWidth);
					yPixel = (int)((((za + 1.0) / 2.0) - 1.0) * cubeFaceHeight);

					pixel = top_frame.at<Vec3b>(abs(yPixel), abs(xPixel));
				}
				else if (ya == -1)
				{
					//Down
					xPixel = (int)((((xa + 1.0) / 2.0)) * cubeFaceWidth);
					yPixel = (int)((((za + 1.0) / 2.0)) * cubeFaceHeight);

					pixel = bottom_frame.at<Vec3b>(abs(yPixel), abs(xPixel));
				}
				else if (za == 1)
				{
					//Front
					xPixel = (int)((((xa + 1.0) / 2.0)) * cubeFaceWidth);
					yPixel = (int)((((ya + 1.0) / 2.0)) * cubeFaceHeight);

					pixel = front_frame.at<Vec3b>(abs(yPixel), abs(xPixel));
				}
				else if (za == -1)
				{
					//Back
					xPixel = (int)((((xa + 1.0) / 2.0) - 1.0) * cubeFaceWidth);
					yPixel = (int)((((ya + 1.0) / 2.0)) * cubeFaceHeight);

					pixel = back_frame.at<Vec3b>(abs(yPixel), abs(xPixel));
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