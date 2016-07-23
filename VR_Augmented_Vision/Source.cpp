#include "Source.h"
#include "GPURender.h"
#include "CPURender.h"

#define location "C:/Users/Manganese/Desktop/"

//write only
//the frame that is displayed to the user
Mat projected_frame;

//opencv's array of input videos
VideoCapture input_videos[NUMBER_OF_CAMERAS];

//array of pointers to each camera's frame
//each mat pointer is atomically updated to the new mat when a new frame comes in.
//eliminates case where mutex would make thread stop writing, which could lead to another frame, which has been updated, not be drawn
Frame_Pointer frame_array[NUMBER_OF_CAMERAS];

int WINAPI WinMain(HINSTANCE hinst, HINSTANCE, LPSTR, int)
{
	// Allocate a console for this app
	AllocConsole();
	AttachConsole(GetCurrentProcessId());
	freopen("CON", "w", stdout);

	//init vr and mirror
#if USE_VR
	Initalize_VR(hinst, screenWidth, screenHeight);
#else
	namedWindow("");
#endif

	//load input video streams
	input_videos[top_frame].open(location"top.avi");
	input_videos[bottom_frame].open(location"bottom.avi");
	input_videos[front_frame].open(location"front.avi");
	input_videos[left_frame].open(location"left.avi");
	input_videos[right_frame].open(location"right.avi");
	input_videos[back_frame].open(location"back.avi");

	//run main render loop
#if GPU
	GPU_Render(hinst);
#else
	CPU_Render(hinst);
#endif
}