#include "Source.h"

//mutex for atomic alteration of the frame buffer pointers and index 
CRITICAL_SECTION update_frame_buffer;

//passed to each thread to give it info about what screen area it handles
typedef struct{
	unsigned int x;
	unsigned int y;
	unsigned int width;
	unsigned int height;
} Thread_Screen;

DWORD WINAPI Project_to_Screen(void* input);

int CPU_Render(HINSTANCE hinst)
{
	InitializeCriticalSection(&update_frame_buffer);//set up mutex

	projected_frame = Mat::zeros(SCREEN_HEIGHT, SCREEN_WIDTH, CV_8UC4);//CV_[The number of bits per item][Signed or Unsigned][Type Prefix]C[The channel number]

	for (unsigned char i = 0; i < NUMBER_OF_CAMERAS; ++i){
		Mat frame = Mat::zeros(CUBE_FACE_HEIGHT, CUBE_FACE_WIDTH, CV_8UC3);
		frame_array[i].frame_0 = new Mat(frame);//copy frame into heap, return pointer
		frame_array[i].frame_1 = new Mat(frame);
		frame_array[i].selected_frame = 0;
	}

	//start n threads, to cover the entire screen area
	unsigned int per_thread_width = SCREEN_WIDTH / ((NUM_THREADS-1)/3);
	unsigned int per_thread_height = SCREEN_HEIGHT / ((NUM_THREADS-1)/2);
	int x_offset = 0;
	int y_offset = 0;
	for (unsigned int i = 0; i < NUM_THREADS-1; ++i){
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
		if (x_offset % 2 == 0){
			x_offset = 0;
			y_offset++;
		}
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
				Mat* next_frame_pointer = new Mat(next_frame);
				//put new frame in framebuffer and update frame buffer current
				switch (frame_array[i].selected_frame){
					case 0:
						//this is the only syncronization needed because b4 this point, if a thread gets a pointer to the image data,
						//the mat's refcount will atomicly increment, and this section wont free the image data, but will change the pointer.
						//so the old thread will still have access to stale, unfreed data.
						EnterCriticalSection(&update_frame_buffer);
							frame_array[i].frame_1->release();
							frame_array[i].frame_1 = next_frame_pointer;//this changes the pointer to a new malloc, non-atomically.
							frame_array[i].selected_frame = 1;//must be atomic
						LeaveCriticalSection(&update_frame_buffer);
						break;
					case 1:
						EnterCriticalSection(&update_frame_buffer);
							frame_array[i].frame_0->release();
							frame_array[i].frame_0 = next_frame_pointer;
							frame_array[i].selected_frame = 0;
						LeaveCriticalSection(&update_frame_buffer);
						break;
				}
			}
		}

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
	}

	return EXIT_SUCCESS;
}

DWORD WINAPI Project_to_Screen(void* input){
	Thread_Screen* area = (Thread_Screen*)input;
	while (1){
		//http://stackoverflow.com/questions/34250742/converting-a-cubemap-into-equirectangular-panorama
		//inverse mapping

		double u, v; //Normalised texture coordinates, from 0 to 1, starting at lower left corner
		double phi, theta; //Polar coordinates

		for (int j = area->y; j < area->y+area->height; j++)
		{
			//Rows start from the bottom
			v = 1 - ((double)j / SCREEN_HEIGHT);
			theta = v * M_PI;

			for (int i = area->x; i < area->x+area->width; i++)//go along columns in inner loop for speed.
			{
				//convert x,y cartesian to u,v polar

				//Columns start from the left
				u = ((double)i / SCREEN_WIDTH);
				phi = u * 2 * M_PI;

				//convert polar to 3d vector
				double x, y, z; //Unit vector
				x = sin(phi) * sin(theta) * -1;
				y = cos(theta);
				z = cos(phi) * sin(theta) * -1;

				double xa, ya, za;
				double a;

				a = fmax(fmax(abs(x), abs(y)), abs(z));

				//Vector Parallel to the unit vector that lies on one of the cube faces
				xa = x / a;
				ya = y / a;
				za = z / a;

				Vec3b pixel;
				int xPixel, yPixel;

				if (xa == 1)
				{
					//Right
					xPixel = (int)((((za + 1.0) / 2.0) - 1.0) * CUBE_FACE_WIDTH);
					yPixel = (int)((((ya + 1.0) / 2.0)) * CUBE_FACE_HEIGHT);

					switch (frame_array[right_frame].selected_frame){
						case 0:
							pixel = frame_array[right_frame].frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel = frame_array[right_frame].frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
					//pixel = Vec3b(0, 0, 255);//red
				}
				else if (xa == -1)
				{
					//Left
					xPixel = (int)((((za + 1.0) / 2.0)) * CUBE_FACE_WIDTH);
					yPixel = (int)((((ya + 1.0) / 2.0)) * CUBE_FACE_HEIGHT);

					switch (frame_array[left_frame].selected_frame){
						case 0:
							pixel = frame_array[left_frame].frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel = frame_array[left_frame].frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
					//pixel = Vec3b(0, 255, 255);//yellow
				}
				else if (ya == -1)
				{
					//Up
					xPixel = (int)((((xa + 1.0) / 2.0)) * CUBE_FACE_WIDTH);
					yPixel = (int)((((za + 1.0) / 2.0) - 1.0) * CUBE_FACE_HEIGHT);
					//flip vertical
					yPixel = (CUBE_FACE_HEIGHT - 1) - abs(yPixel);

					switch (frame_array[top_frame].selected_frame){
						case 0:
							pixel = frame_array[top_frame].frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel = frame_array[top_frame].frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
					//pixel = Vec3b(0, 60, 255);//orange
				}
				else if (ya == 1)
				{
					//Down
					xPixel = (int)((((xa + 1.0) / 2.0)) * CUBE_FACE_WIDTH);
					yPixel = (int)((((za + 1.0) / 2.0)) * CUBE_FACE_HEIGHT);
					//flip vertical
					yPixel = (CUBE_FACE_HEIGHT - 1) - abs(yPixel);

					switch (frame_array[bottom_frame].selected_frame){
						case 0:
							pixel = frame_array[bottom_frame].frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel = frame_array[bottom_frame].frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
					//pixel = Vec3b(255, 0, 0);//blue
				}
				else if (za == 1)
				{
					//Front
					xPixel = (int)((((xa + 1.0) / 2.0)) * CUBE_FACE_WIDTH);
					yPixel = (int)((((ya + 1.0) / 2.0)) * CUBE_FACE_HEIGHT);

					switch (frame_array[front_frame].selected_frame){
						case 0:
							pixel = frame_array[front_frame].frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel = frame_array[front_frame].frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
					//pixel = Vec3b(150, 150, 150);//grey
				}
				else if (za == -1)
				{
					//Back
					xPixel = (int)((((xa + 1.0) / 2.0) - 1.0) * CUBE_FACE_WIDTH);
					yPixel = (int)((((ya + 1.0) / 2.0)) * CUBE_FACE_HEIGHT);

					switch (frame_array[back_frame].selected_frame){
						case 0:
							pixel = frame_array[back_frame].frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel = frame_array[back_frame].frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
					//pixel = Vec3b(150, 0, 0);//light blue
				}
				else
				{
					printf("Unknown face, something went wrong");
				}

				projected_frame.at<Vec4b>(j, i) = Vec4b{ pixel[2], pixel[1], pixel[0], 0xFF };
			}
		}
	}
}