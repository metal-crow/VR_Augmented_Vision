#include "Source.h"
#include "kernal.h"

//in between step for the gpu to copy the projected frame datatto before its converted to a mat on the host
unsigned char* projected_frame_data;

DWORD WINAPI Grab_Camera_Frame(void* camera_num_voidp);

int GPU_Render(HINSTANCE hinst)
{
	//setup the frame buffer and selected frame, + other gpu variables
	projected_frame_data = (unsigned char*)malloc(screenWidth*screenHeight * 3 * sizeof(unsigned char));
	if (allocate_frames(NUMBER_OF_CAMERAS, cubeFaceWidth, cubeFaceHeight, screenWidth, screenHeight) != 0){
		return EXIT_FAILURE;
	}

	//start threads to read any new frames from the cameras
	for (unsigned char i = 0; i < NUMBER_OF_CAMERAS; ++i){
		CreateThread(NULL,0,Grab_Camera_Frame,(void*)i,0,NULL); 
	}

	while (1){
		#if DEBUG_TIME
			long start = clock();
		#endif

		//~<1 ms
		cuda_run();//run gpu projection

		printf("computed gpu:%ld\n", clock() - start);

		//2 ms
		read_projected_frame(projected_frame_data);

		printf("read projection:%ld\n", clock() - start);

		projected_frame = Mat(screenHeight, screenWidth, CV_8UC3, projected_frame_data);
		#if USE_VR
			UpdateTexture(projected_frame);//15 ms TODO
			printf("update texture:%ld\n", clock() - start);
			Main_VR_Render_Loop();//5 ms
		#else
			imshow("", projected_frame);
			waitKey(1);
		#endif
		#if DEBUG_TIME
			printf("send to oculus:%ld\n", clock() - start);
		#endif
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
		}
		//we don't want to 100% cpu, and know that a camera wont have a new frame instantly available, so yield
		Sleep(10);
	}
}