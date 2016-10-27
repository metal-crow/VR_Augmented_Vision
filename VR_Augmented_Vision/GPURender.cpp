#include "Source.h"

#include "kernal.h"
#include <assert.h>

//in between step for the gpu to copy the projected frame data to before its converted to a mat on the host
typedef struct{
	unsigned char* left;
	unsigned char* right;
} Projected_Frame_Raw;

Projected_Frame_Raw* projected_frame_data;

//set by the camera threads so we don't unnecicarily start the gpu every main loop tick with old data
//this allows faster reaction for an actual new frame, instead of if a new frame comes in just as the gpu finishes. Then we'd have to wait
volatile bool new_frame_grabbed = false;

//start inclusive, end exclusive
typedef struct {
	unsigned char start_cam_i;
	unsigned char end_cam_i;
} Camera_Thread_Watcher;

DWORD WINAPI Grab_Camera_Frame(void* views_responsible_p);
DWORD WINAPI VR_Render_Thread(void* null);

int GPU_Render(HINSTANCE hinst)
{
	//setup the frame buffer and selected frame, + other gpu variables
	projected_frame_data = (Projected_Frame_Raw*)allocate_frames(CUBE_FACE_WIDTH, CUBE_FACE_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT);
	if (projected_frame_data == NULL){
		return EXIT_FAILURE;
	}

	assert(number_of_cameras >= NUM_THREADS - 1);//can't currently handle cases this isnt true.	
	double cams_per_thread = number_of_cameras / (double)(NUM_THREADS - 1);
	//start threads to read new frames from the cameras
	for (unsigned char i = 0; i < NUM_THREADS-1; ++i){
		Camera_Thread_Watcher* cams_responsible = (Camera_Thread_Watcher*)malloc(sizeof(Camera_Thread_Watcher));
		cams_responsible->start_cam_i = round(i*cams_per_thread);
		cams_responsible->end_cam_i = round((i + 1)*cams_per_thread);

		CreateThread(NULL, 0, Grab_Camera_Frame, (void*)cams_responsible, 0, NULL);
	}

	//start thread to send images to oculus
	HANDLE vr_thread = CreateThread(NULL, 0, VR_Render_Thread, NULL, 0, NULL);
	SetThreadPriority(vr_thread, THREAD_PRIORITY_TIME_CRITICAL);

	while (1){
		//trigger a new projection generation
		if (new_frame_grabbed){
			new_frame_grabbed = false;

			cuda_run();//async run gpu projection

			read_projected_frame();//async copy from device to host memory

			projected_frame.left = Mat(SCREEN_HEIGHT, SCREEN_WIDTH, CV_8UC4, projected_frame_data->left);
			projected_frame.right = Mat(SCREEN_HEIGHT, SCREEN_WIDTH, CV_8UC4, projected_frame_data->right);
			#if USE_VR
				//set textures used for oculus display
				UpdateTexture(projected_frame.left.data, projected_frame.right.data);//2 ms
				//TODO add Main_VR_Render_Loop back here
			#endif
		}
	}

	return EXIT_SUCCESS;
}

//camera(s) are handled in this thread
//check if a new frame is available, and grab it, then copy to GPU
DWORD WINAPI Grab_Camera_Frame(void* cameras_responsible_p){
	Camera_Thread_Watcher cameras_responsible = *(Camera_Thread_Watcher*)cameras_responsible_p;

	while (1){
		//for all assigned cameras
		for (unsigned char cam_i = cameras_responsible.start_cam_i; cam_i<cameras_responsible.end_cam_i; ++cam_i){
			//if there is a new frame for a camera, get it
			//http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-grab
			//NOTE: cube_faces is stored in the GPU code, not used here

			if (cameras[cam_i].cam_left.grab()){
				Mat new_frame;
				cameras[cam_i].cam_left.retrieve(new_frame);
				//send the frame to the device for each slice it is a part of, where it is copied into gpu cube face.
				for (unsigned char cube_face = 0; cube_face < cameras[cam_i].number_of_cube_faces; ++cube_face){
					copy_new_frame(cameras[cam_i].slices_of[cube_face].cube_face, true, new_frame.data,
						cameras[cam_i].slices_of[cube_face].slice_loc_in_view_x, cameras[cam_i].slices_of[cube_face].slice_loc_in_view_y,
						cameras[cam_i].slices_of[cube_face].slice_width,         cameras[cam_i].slices_of[cube_face].slice_height);
				}
				new_frame_grabbed = true;
			}

			if (cameras[cam_i].cam_right.grab()){
				Mat new_frame;
				cameras[cam_i].cam_right.retrieve(new_frame);
				//send the frame to the device for each slice it is a part of, where it is copied into gpu cube face.
				for (unsigned char cube_face = 0; cube_face < cameras[cam_i].number_of_cube_faces; ++cube_face){
					copy_new_frame(cameras[cam_i].slices_of[cube_face].cube_face, false, new_frame.data,
						cameras[cam_i].slices_of[cube_face].slice_loc_in_view_x, cameras[cam_i].slices_of[cube_face].slice_loc_in_view_y,
						cameras[cam_i].slices_of[cube_face].slice_width,         cameras[cam_i].slices_of[cube_face].slice_height);
				}
				new_frame_grabbed = true;
			}
		}

		//we don't want to 100% cpu, and know that a camera wont have a new frame instantly available, so yield
		Sleep(10);
	}

	free(cameras_responsible_p);
}

//updates to the oculus are sent in this thread
//optimally this would not exist, and oculus would only update on a new frame
//however, currently that leads to issues (something something async timewarp is stupid)
DWORD WINAPI VR_Render_Thread(void* null){
	#if DEBUG_TIME
		long start_time = clock();
		unsigned long long submit_frame_counter = 0;
	#endif

	while (1){
		#if USE_VR
			Main_VR_Render_Loop();
		#else
			imshow("", projected_frame.left);
			waitKey(1);
			Sleep(10);
		#endif

		#if DEBUG_TIME
			submit_frame_counter++;
			if ((clock() - start_time) / CLOCKS_PER_SEC >= 1){
				printf("fps %d\n", submit_frame_counter);
				submit_frame_counter = 0;
				start_time = clock();
			}
		#endif
	}
}
