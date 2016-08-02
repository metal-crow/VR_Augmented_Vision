#include "Source.h"
#include "kernal.h"
#include <assert.h>

//in between step for the gpu to copy the projected frame datat to before its converted to a mat on the host
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
	unsigned char start_view_i;
	unsigned char end_view_i;
} Viewpoint_Thread_Watcher;

DWORD WINAPI Grab_Camera_Frame(void* views_responsible_p);
DWORD WINAPI VR_Render_Thread(void* null);

#undef NUM_THREADS
#define NUM_THREADS 2 //TEMP: higher fps using fewer camera watching threads

int GPU_Render(HINSTANCE hinst)
{
	//setup the frame buffer and selected frame, + other gpu variables
	projected_frame_data = (Projected_Frame_Raw*)allocate_frames(NUMBER_OF_VIEWPOINTS, CUBE_FACE_WIDTH, CUBE_FACE_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT);
	if (projected_frame_data == NULL){
		return EXIT_FAILURE;
	}

	assert(NUMBER_OF_VIEWPOINTS >= NUM_THREADS - 1);//can't currently handle cases this isnt true.	
	double views_per_thread = NUMBER_OF_VIEWPOINTS / (double)(NUM_THREADS - 1);
	//start threads to read new frames from the views
	for (unsigned char i = 0; i < NUM_THREADS-1; ++i){
		Viewpoint_Thread_Watcher* views_responsible = (Viewpoint_Thread_Watcher*)malloc(sizeof(Viewpoint_Thread_Watcher));
		views_responsible->start_view_i = round(i*views_per_thread);
		views_responsible->end_view_i = round((i + 1)*views_per_thread);

		CreateThread(NULL, 0, Grab_Camera_Frame, (void*)views_responsible, 0, NULL);
	}

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
				UpdateTexture(projected_frame.left.data, projected_frame.right.data);//2 ms
			#endif
		}
	}

	return EXIT_SUCCESS;
}

//camera(s) are handled in this thread
//check if a new frame is available, and grab it, then copy to GPU
DWORD WINAPI Grab_Camera_Frame(void* views_responsible_p){
	Viewpoint_Thread_Watcher views_responsible = *(Viewpoint_Thread_Watcher*)views_responsible_p;

	while (1){
		for (unsigned char view_num = views_responsible.start_view_i; view_num < views_responsible.end_view_i; ++view_num){
			//if there is a new frame for a camera, get it
			//http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-grab
			if (input_views[view_num].left.grab()){
				Mat next_frame;
				input_views[view_num].left.retrieve(next_frame);
				copy_new_frame(view_num, true, next_frame.data);//send frame to device memory
				new_frame_grabbed = true;
			}
			if (input_views[view_num].right.grab()){
				Mat next_frame;
				input_views[view_num].right.retrieve(next_frame);
				copy_new_frame(view_num, false, next_frame.data);
				new_frame_grabbed = true;
			}
		}

		//we don't want to 100% cpu, and know that a camera wont have a new frame instantly available, so yield
		Sleep(10);
	}
}

DWORD WINAPI VR_Render_Thread(void* null){
	#if DEBUG_TIME
		long start_time = clock();
		unsigned long long submit_frame_counter = 0;
	#endif

	while (1){
		//want to update vr headset regardless of new frame (something something async timewarp is stupid)
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
