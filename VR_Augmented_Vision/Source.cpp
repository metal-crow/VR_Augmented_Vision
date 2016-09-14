#include "Source.h"
#include "GPURender.h"
#include "CPURender.h"

#define location "C:/Users/Manganese/Desktop/backyard sterio/"

//the image that is displayed to the user
Frame projected_frame;

//the image frames that make up the cube faces
Frame cube_faces[NUMBER_OF_VIEWPOINTS];

//the input cameras that are used to populate the cube_faces
unsigned char number_of_cameras = 10;
Camera_View cameras[10];//TODO make this dynamic

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

	//top
	cameras[0].cam_left.open(location"backyard_top_L.png.mp4");
	cameras[0].cam_right.open(location"backyard_top_R.png.mp4");
	cameras[0].number_of_cube_faces = 1;
	cameras[0].slices_of = (CubeFace_Slice*)calloc(sizeof(CubeFace_Slice), cameras[0].number_of_cube_faces);
	cameras[0].slices_of[0].cube_face = top_view;
	cameras[0].slices_of[0].slice_width = 1536;
	cameras[0].slices_of[0].slice_height = 1536;
	cameras[0].slices_of[0].slice_loc_in_view_x = 0;
	cameras[0].slices_of[0].slice_loc_in_view_y = 0;

	//bottom
	cameras[1].cam_left.open(location"backyard_bot_L.png.mp4");
	cameras[1].cam_right.open(location"backyard_bot_R.png.mp4");
	cameras[1].number_of_cube_faces = 1;
	cameras[1].slices_of = (CubeFace_Slice*)calloc(sizeof(CubeFace_Slice), cameras[1].number_of_cube_faces);
	cameras[1].slices_of[0].cube_face = bottom_view;
	cameras[1].slices_of[0].slice_width = 1536;
	cameras[1].slices_of[0].slice_height = 1536;
	cameras[1].slices_of[0].slice_loc_in_view_x = 0;
	cameras[1].slices_of[0].slice_loc_in_view_y = 0;

	cameras[2].cam_left.open(location"backyard_back_L.png.mp4");
	cameras[2].cam_right.open(location"backyard_back_R.png.mp4");
	cameras[2].number_of_cube_faces = 1;
	cameras[2].slices_of = (CubeFace_Slice*)calloc(sizeof(CubeFace_Slice), cameras[2].number_of_cube_faces);
	cameras[2].slices_of[0].cube_face = back_view;
	cameras[2].slices_of[0].slice_width = 1536;
	cameras[2].slices_of[0].slice_height = 1536;
	cameras[2].slices_of[0].slice_loc_in_view_x = 0;
	cameras[2].slices_of[0].slice_loc_in_view_y = 0;

	cameras[3].cam_left.open(location"backyard_front_L.png.mp4");
	cameras[3].cam_right.open(location"backyard_front_R.png.mp4");
	cameras[3].number_of_cube_faces = 1;
	cameras[3].slices_of = (CubeFace_Slice*)calloc(sizeof(CubeFace_Slice), cameras[3].number_of_cube_faces);
	cameras[3].slices_of[0].cube_face = front_view;
	cameras[3].slices_of[0].slice_width = 1536;
	cameras[3].slices_of[0].slice_height = 1536;
	cameras[3].slices_of[0].slice_loc_in_view_x = 0;
	cameras[3].slices_of[0].slice_loc_in_view_y = 0;

	cameras[4].cam_left.open(location"backyard_left_L.png.mp4");
	cameras[4].cam_right.open(location"backyard_left_R.png.mp4");
	cameras[4].number_of_cube_faces = 1;
	cameras[4].slices_of = (CubeFace_Slice*)calloc(sizeof(CubeFace_Slice), cameras[4].number_of_cube_faces);
	cameras[4].slices_of[0].cube_face = left_view;
	cameras[4].slices_of[0].slice_width = 1536;
	cameras[4].slices_of[0].slice_height = 1536;
	cameras[4].slices_of[0].slice_loc_in_view_x = 0;
	cameras[4].slices_of[0].slice_loc_in_view_y = 0;

	cameras[5].cam_left.open(location"backyard_right_L.png.mp4");
	cameras[5].cam_right.open(location"backyard_right_R.png.mp4");
	cameras[5].number_of_cube_faces = 1;
	cameras[5].slices_of = (CubeFace_Slice*)calloc(sizeof(CubeFace_Slice), cameras[5].number_of_cube_faces);
	cameras[5].slices_of[0].cube_face = right_view;
	cameras[5].slices_of[0].slice_width = 1536;
	cameras[5].slices_of[0].slice_height = 1536;
	cameras[5].slices_of[0].slice_loc_in_view_x = 0;
	cameras[5].slices_of[0].slice_loc_in_view_y = 0;

	bool prio_set = SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
	if (!prio_set){
		fprintf(stderr, "Unable to set process to REALTIME");
	}

	//initalize default output images
	projected_frame.left = Mat::zeros(SCREEN_HEIGHT, SCREEN_WIDTH, CV_8UC4);//CV_[The number of bits per item][Signed or Unsigned][Type Prefix]C[The channel number]
	projected_frame.right = Mat::zeros(SCREEN_HEIGHT, SCREEN_WIDTH, CV_8UC4);

	//run main render loop
#if GPU
	GPU_Render(hinst);
#else
	CPU_Render(hinst);
#endif
}