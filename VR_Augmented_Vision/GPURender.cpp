#include "Source.h"
#include "kernal.h"

//in between step for the gpu to copy the projected frame datatto before its converted to a mat on the host
unsigned char* projected_frame_data;

int GPU_Render(HINSTANCE hinst)
{
	//setup the frame buffer and selected frame, + other gpu variables
	projected_frame_data = (unsigned char*)malloc(screenWidth*screenHeight * 3 * sizeof(unsigned char));
	if (allocate_frames(NUMBER_OF_CAMERAS, cubeFaceWidth, cubeFaceHeight, screenWidth, screenHeight) != 0){
		return EXIT_FAILURE;
	}

	while (1){
		#if DEBUG_TIME
			long start = clock();
		#endif
		//read any new frames from the cameras
		for (unsigned char i = 0; i < NUMBER_OF_CAMERAS; ++i){
			//if there is a new frame for a camera, get it
			//http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-grab
			if (input_videos[i].grab()){
				Mat next_frame;
				input_videos[i].retrieve(next_frame);
				copy_new_frame(i, next_frame.data);//send frame to device memory
			}
		}

		cuda_run();//run gpu projection
		read_projected_frame(projected_frame_data);
		projected_frame = Mat(screenHeight, screenWidth, CV_8UC3, projected_frame_data);
		#if USE_VR
			UpdateTexture(projected_frame);
			Main_VR_Render_Loop();
		#else
			imshow("", projected_frame);
			waitKey(1);
		#endif
		#if DEBUG_TIME
			printf("time:%ld\n", clock() - start);
		#endif
		Sleep(100);
	}

	return EXIT_SUCCESS;
}