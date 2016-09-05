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

	//initalize frame array (set default images)
	for (unsigned char i = 0; i < NUMBER_OF_VIEWPOINTS; ++i){
		Mat frame = Mat::zeros(CUBE_FACE_HEIGHT, CUBE_FACE_WIDTH, CV_8UC3);
		cube_faces[i].left = Mat(frame);
		cube_faces[i].right = Mat(frame);
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
		for (unsigned char cam_i = 0; cam_i < number_of_cameras; ++cam_i){
			//if there is a new frame for a camera, get it
			//http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-grab
			//theretically, since this doesnt have safety checks this could lead to eye misalignment(1 eye camera frame is behind the other), but trade off for quicker new frame updates
			if (cameras[cam_i].cam_left.grab()){
				Mat next_frame;
				cameras[cam_i].cam_left.retrieve(next_frame);
				//copy slices in this camera into cube faces
				for (unsigned char cube_face = 0; cube_face < cameras[cam_i].number_of_cube_faces; ++cube_face){
					//update the frame with the new slice, in place(so don't need safety)
					next_frame.copyTo(Mat(cube_faces[cameras[cam_i].slices_of[cube_face].cube_face].left,
											  Rect(cameras[cam_i].slices_of[cube_face].slice_loc_in_view_x, cameras[cam_i].slices_of[cube_face].slice_loc_in_view_y, 
												   cameras[cam_i].slices_of[cube_face].slice_width,         cameras[cam_i].slices_of[cube_face].slice_height)));
				}
			}
			if (cameras[cam_i].cam_right.grab()){
				Mat next_frame;
				cameras[cam_i].cam_right.retrieve(next_frame);
				//copy slices in this camera into cube faces
				for (unsigned char cube_face = 0; cube_face < cameras[cam_i].number_of_cube_faces; ++cube_face){
					//update the frame with the new slice, in place(so don't need safety)
					next_frame.copyTo(Mat(cube_faces[cameras[cam_i].slices_of[cube_face].cube_face].right,
										  Rect(cameras[cam_i].slices_of[cube_face].slice_loc_in_view_x, cameras[cam_i].slices_of[cube_face].slice_loc_in_view_y,
											   cameras[cam_i].slices_of[cube_face].slice_width,         cameras[cam_i].slices_of[cube_face].slice_height)));
				}
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

					pixel_left = cube_faces[right_view].left.at<Vec3b>(abs(yPixel), abs(xPixel));
					pixel_right = cube_faces[right_view].right.at<Vec3b>(abs(yPixel), abs(xPixel));
				}
				else if (xa == -1)
				{
					//Left
					xPixel = (int)((((za + 1.0) / 2.0)) * (CUBE_FACE_WIDTH-1));
					yPixel = (int)((((ya + 1.0) / 2.0)) * (CUBE_FACE_HEIGHT-1));

					pixel_left = cube_faces[left_view].left.at<Vec3b>(abs(yPixel), abs(xPixel));
					pixel_right = cube_faces[left_view].right.at<Vec3b>(abs(yPixel), abs(xPixel));
				}
				else if (ya == -1)
				{
					//Up
					xPixel = (int)((((xa + 1.0) / 2.0)) * (CUBE_FACE_WIDTH-1));
					yPixel = (int)((((za + 1.0) / 2.0) - 1.0) * (CUBE_FACE_HEIGHT-1));
					//flip vertical
					yPixel = (CUBE_FACE_HEIGHT - 1) - abs(yPixel);

					pixel_left = cube_faces[top_view].left.at<Vec3b>(abs(yPixel), abs(xPixel));
					pixel_right = cube_faces[top_view].right.at<Vec3b>(abs(yPixel), abs(xPixel));
				}
				else if (ya == 1)
				{
					//Down
					xPixel = (int)((((xa + 1.0) / 2.0)) * (CUBE_FACE_WIDTH-1));
					yPixel = (int)((((za + 1.0) / 2.0)) * (CUBE_FACE_HEIGHT-1));
					//flip vertical
					yPixel = (CUBE_FACE_HEIGHT - 1) - abs(yPixel);

					pixel_left = cube_faces[bottom_view].left.at<Vec3b>(abs(yPixel), abs(xPixel));
					pixel_right = cube_faces[bottom_view].right.at<Vec3b>(abs(yPixel), abs(xPixel));
				}
				else if (za == 1)
				{
					//Front
					xPixel = (int)((((xa + 1.0) / 2.0)) * (CUBE_FACE_WIDTH-1));
					yPixel = (int)((((ya + 1.0) / 2.0)) * (CUBE_FACE_HEIGHT-1));

					pixel_left = cube_faces[front_view].left.at<Vec3b>(abs(yPixel), abs(xPixel));
					pixel_right = cube_faces[front_view].right.at<Vec3b>(abs(yPixel), abs(xPixel));
				}
				else if (za == -1)
				{
					//Back
					xPixel = (int)((((xa + 1.0) / 2.0) - 1.0) * (CUBE_FACE_WIDTH-1));
					yPixel = (int)((((ya + 1.0) / 2.0)) * (CUBE_FACE_HEIGHT-1));

					pixel_left = cube_faces[back_view].left.at<Vec3b>(abs(yPixel), abs(xPixel));
					pixel_right = cube_faces[back_view].right.at<Vec3b>(abs(yPixel), abs(xPixel));
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