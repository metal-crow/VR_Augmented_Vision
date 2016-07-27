#include "Source.h"
#include "kernal.h"

//in between step for the gpu to copy the projected frame datat to before its converted to a mat on the host
unsigned char* projected_frame_data;

//set by the camera threads so we don't unnecicarily start the gpu every main loop tick with old data
//this allows faster reaction for an actual new frame, instead of if a new frame comes in just as the gpu finishes. Then we'd have to wait
//this would normally be bad practice but a)ms visual c++ compiler may make this atomic b)it controls an if statement. worst case we drop a frame
//TODO "why not a Condition Variable?" slower (by enough to be significant?), and we don't have to worry about main spinlocking since we only create as many threads as we have cpus(do we?)
volatile bool new_frame_grabbed = false;

DWORD WINAPI Grab_Camera_Frame(void* camera_num_voidp);

int GPU_Render(HINSTANCE hinst)
{
	//setup the frame buffer and selected frame, + other gpu variables
	projected_frame_data = allocate_frames(NUMBER_OF_CAMERAS, cubeFaceWidth, cubeFaceHeight, screenWidth, screenHeight);
	if (projected_frame_data == NULL){
		return EXIT_FAILURE;
	}

	//start threads to read any new frames from the cameras
	for (unsigned char i = 0; i < NUMBER_OF_CAMERAS; ++i){
		CreateThread(NULL,0,Grab_Camera_Frame,(void*)i,0,NULL); 
	}

	while (1){
		if (new_frame_grabbed){
			new_frame_grabbed = false;

			cuda_run();//async run gpu projection

			#if DEBUG_TIME
				long start = clock();
			#endif

			//12 ms TODO
			read_projected_frame();

			printf("read projection:%ld\n", clock() - start);

			projected_frame = Mat(screenHeight, screenWidth, CV_8UC4, projected_frame_data);
			#if USE_VR
				UpdateTexture(projected_frame);//2 ms
				//printf("update texture:%ld\n", clock() - start);
				Main_VR_Render_Loop();//5 ms TODO
			#else
				imshow("", projected_frame);
				waitKey(1);
			#endif
			
			//printf("send to oculus:%ld\n", clock() - start);
		}
	}

	return EXIT_SUCCESS;
}

//Each camera is handled in this thread
//check if a new frame is available, and grab it, then copy to GPU
//2 ms per each
//camera number is embedded in the input pointer
DWORD WINAPI Grab_Camera_Frame(void* camera_num_voidp){
	unsigned char camera_num = (unsigned char)camera_num_voidp;

	while (1){
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