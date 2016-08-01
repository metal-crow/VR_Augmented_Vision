#include "Source.h"
#include "GPURender.h"
#include "CPURender.h"

#define location "C:/Users/Manganese/Desktop/"

//the image that is displayed to the user
Projected_Frame projected_frame;

//the set of input viewpoints
ViewPoint input_views[NUMBER_OF_VIEWPOINTS];

//array of pointers to each camera's frame
//each mat pointer is atomically updated to the new mat when a new frame comes in.
//eliminates case where mutex would make thread stop writing, which could lead to another frame, which has been updated, not be drawn
ViewPoint_Frame_Pointer viewpoint_frame_array[NUMBER_OF_VIEWPOINTS];

int WINAPI WinMain(HINSTANCE hinst, HINSTANCE, LPSTR, int)
{
	// Allocate a console for this app
	AllocConsole();
	AttachConsole(GetCurrentProcessId());
	freopen("CON", "w", stdout);

	//init vr and mirror
#if USE_VR
	Initalize_VR(hinst, SCREEN_WIDTH, SCREEN_HEIGHT);
#else
	namedWindow("");
#endif

	//load input video streams
	/*input_views[top_view].left.open(location"3D_20_LEFT.mp4");
	input_views[top_view].right.open(location"3D_20_RIGHT.mp4");
	input_views[bottom_view].left.open(location"3D_20_LEFT.mp4");
	input_views[bottom_view].right.open(location"3D_20_RIGHT.mp4");*/
	input_views[front_view].left.open(location"3D_20_LEFT.mp4");
	input_views[front_view].right.open(location"3D_20_RIGHT.mp4");
	/*input_views[left_view].left.open(location"3D_20_LEFT.mp4");
	input_views[left_view].right.open(location"3D_20_RIGHT.mp4");
	input_views[right_view].left.open(location"3D_20_LEFT.mp4");
	input_views[right_view].right.open(location"3D_20_RIGHT.mp4");
	input_views[back_view].left.open(location"3D_20_LEFT.mp4");
	input_views[back_view].right.open(location"3D_20_RIGHT.mp4");*/

	//run main render loop
#if GPU
	GPU_Render(hinst);
#else
	CPU_Render(hinst);
#endif
}