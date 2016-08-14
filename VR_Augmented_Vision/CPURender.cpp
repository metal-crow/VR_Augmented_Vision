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
static inline void Update_ViewPoint_Frame(Frame_Pointer* frame, Mat* next_frame_pointer);

int CPU_Render(HINSTANCE hinst)
{
	InitializeCriticalSection(&update_frame_buffer);//set up mutex

	//initalize frame array (set default images)
	for (unsigned char i = 0; i < NUMBER_OF_VIEWPOINTS; ++i){
		Mat frame = Mat::zeros(CUBE_FACE_HEIGHT, CUBE_FACE_WIDTH, CV_8UC3);

		viewpoint_frame_array[i].left.frame_0 = new Mat(frame);//copy frame into heap, return pointer
		viewpoint_frame_array[i].left.frame_1 = new Mat(frame);
		viewpoint_frame_array[i].left.selected_frame = 0;

		viewpoint_frame_array[i].right.frame_0 = new Mat(frame);
		viewpoint_frame_array[i].right.frame_1 = new Mat(frame);
		viewpoint_frame_array[i].right.selected_frame = 0;
	}

	//start n threads, to cover the entire screen area (each thread does both output eyes)
	unsigned int per_thread_width = SCREEN_WIDTH / ((NUM_THREADS-1)/3.0);
	unsigned int per_thread_height = SCREEN_HEIGHT / ((NUM_THREADS-1)/2.0);
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

	#if DEBUG_TIME
		long start_time = clock();
		unsigned long long submit_frame_counter = 0;
	#endif

	while (1){
		//read any new frames from the cameras
		for (unsigned char i = 0; i < NUMBER_OF_VIEWPOINTS; ++i){
			//if there is a new frame for a camera, get it
			//http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-grab
			//theretically, since this doesnt have safety checks this could lead to eye misalignment(1 eye camera frame is behind the other), but trade off for quicker new frame updates
			if (input_views[i].left.grab()){
				Mat next_frame;
				input_views[i].left.retrieve(next_frame);
				Mat* next_frame_pointer = new Mat(next_frame);
				Update_ViewPoint_Frame(&viewpoint_frame_array[i].left, next_frame_pointer);
			}
			if (input_views[i].right.grab()){
				Mat next_frame;
				input_views[i].right.retrieve(next_frame);
				Mat* next_frame_pointer = new Mat(next_frame);
				Update_ViewPoint_Frame(&viewpoint_frame_array[i].right, next_frame_pointer);
			}
		}

		#if USE_VR
			UpdateTexture(projected_frame.left.data, projected_frame.right.data);
			Main_VR_Render_Loop();
		#else
			imshow("", projected_frame.left);
			waitKey(1);
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

//inline helper function for updaing the frame pointer data for an eye
static inline void Update_ViewPoint_Frame(Frame_Pointer* frame, Mat* next_frame_pointer){
	//put new frame in framebuffer and update frame buffer current
	switch (frame->selected_frame){
		case 0:
			//this is the only syncronization needed because b4 this point, if a thread gets a pointer to the image data,
			//the mat's refcount will atomicly increment, and this section wont free the image data, but will change the pointer.
			//so the old thread will still have access to stale, unfreed data.
			EnterCriticalSection(&update_frame_buffer);
				frame->frame_1->release();
				frame->frame_1 = next_frame_pointer;//this changes the pointer to a new malloc, non-atomically.
				frame->selected_frame = 1;//must be atomic
			LeaveCriticalSection(&update_frame_buffer);
			break;
		case 1:
			EnterCriticalSection(&update_frame_buffer);
				frame->frame_0->release();
				frame->frame_0 = next_frame_pointer;
				frame->selected_frame = 0;
			LeaveCriticalSection(&update_frame_buffer);
			break;
	}
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

				//handle both eyes and output images
				Vec3b pixel_left;
				Vec3b pixel_right;
				int xPixel, yPixel;

				if (xa == 1)
				{
					//Right
					xPixel = (int)((((za + 1.0) / 2.0) - 1.0) * (CUBE_FACE_WIDTH-1));
					yPixel = (int)((((ya + 1.0) / 2.0)) * (CUBE_FACE_HEIGHT-1));

					switch (viewpoint_frame_array[right_view].left.selected_frame){
						case 0:
							pixel_left = viewpoint_frame_array[right_view].left.frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel_left = viewpoint_frame_array[right_view].left.frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
					switch (viewpoint_frame_array[right_view].right.selected_frame){
						case 0:
							pixel_right = viewpoint_frame_array[right_view].right.frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel_right = viewpoint_frame_array[right_view].right.frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
				}
				else if (xa == -1)
				{
					//Left
					xPixel = (int)((((za + 1.0) / 2.0)) * (CUBE_FACE_WIDTH-1));
					yPixel = (int)((((ya + 1.0) / 2.0)) * (CUBE_FACE_HEIGHT-1));

					switch (viewpoint_frame_array[left_view].left.selected_frame){
						case 0:
							EnterCriticalSection(&update_frame_buffer);
							pixel_left = viewpoint_frame_array[left_view].left.frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel_left = viewpoint_frame_array[left_view].left.frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
					switch (viewpoint_frame_array[left_view].right.selected_frame){
						case 0:
							pixel_right = viewpoint_frame_array[left_view].right.frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel_right = viewpoint_frame_array[left_view].right.frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
				}
				else if (ya == -1)
				{
					//Up
					xPixel = (int)((((xa + 1.0) / 2.0)) * (CUBE_FACE_WIDTH-1));
					yPixel = (int)((((za + 1.0) / 2.0) - 1.0) * (CUBE_FACE_HEIGHT-1));
					//flip vertical
					yPixel = (CUBE_FACE_HEIGHT - 1) - abs(yPixel);

					switch (viewpoint_frame_array[top_view].left.selected_frame){
						case 0:
							pixel_left = viewpoint_frame_array[top_view].left.frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel_left = viewpoint_frame_array[top_view].left.frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
					switch (viewpoint_frame_array[top_view].right.selected_frame){
						case 0:
							pixel_right = viewpoint_frame_array[top_view].right.frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel_right = viewpoint_frame_array[top_view].right.frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
				}
				else if (ya == 1)
				{
					//Down
					xPixel = (int)((((xa + 1.0) / 2.0)) * (CUBE_FACE_WIDTH-1));
					yPixel = (int)((((za + 1.0) / 2.0)) * (CUBE_FACE_HEIGHT-1));
					//flip vertical
					yPixel = (CUBE_FACE_HEIGHT - 1) - abs(yPixel);

					switch (viewpoint_frame_array[bottom_view].left.selected_frame){
						case 0:
							pixel_left = viewpoint_frame_array[bottom_view].left.frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel_left = viewpoint_frame_array[bottom_view].left.frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
					switch (viewpoint_frame_array[bottom_view].right.selected_frame){
						case 0:
							pixel_right = viewpoint_frame_array[bottom_view].right.frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel_right = viewpoint_frame_array[bottom_view].right.frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
				}
				else if (za == 1)
				{
					//Front
					xPixel = (int)((((xa + 1.0) / 2.0)) * (CUBE_FACE_WIDTH-1));
					yPixel = (int)((((ya + 1.0) / 2.0)) * (CUBE_FACE_HEIGHT-1));

					switch (viewpoint_frame_array[front_view].left.selected_frame){
						case 0:
							pixel_left = viewpoint_frame_array[front_view].left.frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel_left = viewpoint_frame_array[front_view].left.frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
					switch (viewpoint_frame_array[front_view].right.selected_frame){
						case 0:
							pixel_right = viewpoint_frame_array[front_view].right.frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel_right = viewpoint_frame_array[front_view].right.frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
				}
				else if (za == -1)
				{
					//Back
					xPixel = (int)((((xa + 1.0) / 2.0) - 1.0) * (CUBE_FACE_WIDTH-1));
					yPixel = (int)((((ya + 1.0) / 2.0)) * (CUBE_FACE_HEIGHT-1));

					switch (viewpoint_frame_array[back_view].left.selected_frame){
						case 0:
							pixel_left = viewpoint_frame_array[back_view].left.frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel_left = viewpoint_frame_array[back_view].left.frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
					switch (viewpoint_frame_array[back_view].right.selected_frame){
						case 0:
							pixel_right = viewpoint_frame_array[back_view].right.frame_0->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
						case 1:
							pixel_right = viewpoint_frame_array[back_view].right.frame_1->at<Vec3b>(abs(yPixel), abs(xPixel));
							break;
					}
				}
				else
				{
					printf("Unknown face, something went wrong");
				}

				projected_frame.left.at<Vec4b>(j, i) = Vec4b{ pixel_left[2], pixel_left[1], pixel_left[0], 0xFF };
				projected_frame.right.at<Vec4b>(j, i) = Vec4b{ pixel_right[2], pixel_right[1], pixel_right[0], 0xFF };
			}
		}
	}
}