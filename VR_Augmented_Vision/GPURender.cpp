#include "Source.h"
#include "kernal.h"
#include <assert.h>

//in between step for the gpu to copy the projected frame datat to before its converted to a mat on the host
unsigned char* projected_frame_data;

//set by the camera threads so we don't unnecicarily start the gpu every main loop tick with old data
//this allows faster reaction for an actual new frame, instead of if a new frame comes in just as the gpu finishes. Then we'd have to wait
volatile bool new_frame_grabbed = false;

//start inclusive, end exclusive
typedef struct {
	unsigned char start_cam_i;
	unsigned char end_cam_i;
} Camera_Thread_Watcher;

DWORD WINAPI Grab_Camera_Frame(void* camera_num_voidp);

int GPU_Render(HINSTANCE hinst)
{
	//setup the frame buffer and selected frame, + other gpu variables
	projected_frame_data = allocate_frames(NUMBER_OF_CAMERAS, CUBE_FACE_WIDTH, CUBE_FACE_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT);
	if (projected_frame_data == NULL){
		return EXIT_FAILURE;
	}

	assert(NUMBER_OF_CAMERAS >= NUM_THREADS - 1);//can't currently handle cases this isnt true.	
	double cameras_per_thread = NUMBER_OF_CAMERAS / (double)(NUM_THREADS - 1);
	//start threads to read new frames from the cameras
	for (unsigned char i = 0; i < NUM_THREADS-1; ++i){
		Camera_Thread_Watcher* cameras_responsible = (Camera_Thread_Watcher*)malloc(sizeof(Camera_Thread_Watcher));
		cameras_responsible->start_cam_i = round(i*cameras_per_thread);
		cameras_responsible->end_cam_i = round((i + 1)*cameras_per_thread);

		CreateThread(NULL, 0, Grab_Camera_Frame, (void*)cameras_responsible, 0, NULL);
	}

	#if DEBUG_TIME
		long start_time = clock();
		unsigned long long submit_frame_counter = 0;
	#endif

	while (1){
		//trigger a new projection generation
		if (new_frame_grabbed){
			new_frame_grabbed = false;

			cuda_run();//async run gpu projection

			read_projected_frame();//async copy from device to host memory

			projected_frame = Mat(SCREEN_HEIGHT, SCREEN_WIDTH, CV_8UC4, projected_frame_data);
			#if USE_VR
				UpdateTexture(projected_frame);//2 ms
			#endif
		}

		//want to update vr headset regardless of new frame (something something async timewarp is stupid)
		#if USE_VR
			Main_VR_Render_Loop();//TODO mirroring taking too much time?
		#else
			imshow("", projected_frame);
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

	return EXIT_SUCCESS;
}

//camera(s) are handled in this thread
//check if a new frame is available, and grab it, then copy to GPU
DWORD WINAPI Grab_Camera_Frame(void* cameras_responsible_p){
	Camera_Thread_Watcher cameras_responsible = *(Camera_Thread_Watcher*)cameras_responsible_p;

	while (1){
		for (unsigned char camera_num = cameras_responsible.start_cam_i; camera_num < cameras_responsible.end_cam_i; ++camera_num){
			//if there is a new frame for a camera, get it
			//http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-grab
			if (input_videos[camera_num].grab()){
				Mat next_frame;
				input_videos[camera_num].retrieve(next_frame);
				copy_new_frame(camera_num, next_frame.data);//send frame to device memory
				new_frame_grabbed = true;
			}

			//we don't want to 100% cpu, and know that a camera wont have a new frame instantly available, so yield
			Sleep(10);
		}
	}
}