#include "Kernal.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include "math.h"

#include <stdio.h>
#include <stdlib.h>

//enum for each ViewPoint
enum viewpoint_names{
	top_view = 0,
	bottom_view = 1,
	front_view = 2,
	left_view = 3,
	right_view = 4,
	back_view = 5,
};

unsigned char number_of_viewpoints;
unsigned int frame_width, frame_height;

typedef struct{
	unsigned char* left;
	unsigned char* right;
} Projected_Frame;

unsigned int projected_frame_width, projected_frame_height;
Projected_Frame projected_frame_host;//the host's copy of the projected frame. pinned memory
Projected_Frame projected_frame;//dont need a second buffer because memcpy is thread agnostic, so worst case a single pixel is corrupted

typedef struct{
	unsigned char* frame_0;//frames used for frame buffer
	unsigned char* frame_1;
	unsigned char selected_frame;//frame current in use from buffer
} Frame_Pointer;

typedef struct{
	Frame_Pointer left;
	Frame_Pointer right;
} ViewPoint_Frame_Pointer;

ViewPoint_Frame_Pointer* viewpoint_frame_array;//pointer stored on host memory, which points to device memory

#define THREADS_PER_BLOCK_MAX 1024 //note this this usually can't be used since the number of available registers is usually the lower bound
#define THREADS_PER_BLOCK_USED 512

#define BLOCKS_MAX 65535 
unsigned int BLOCKS_USED;

//allocate space for the array of images, and the buffer images for each camera, and the final projected image
//also set some global constants
//return the pointer to the host's copy of projected frame, or NULL for any error
void* allocate_frames(unsigned char arg_number_of_viewpoints,
					 unsigned int arg_frame_width, unsigned int arg_frame_height, 
					 unsigned int arg_projected_frame_width, unsigned int arg_projected_frame_height)
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		return NULL;
	}

	//save these variables
	number_of_viewpoints = arg_number_of_viewpoints;
	frame_width = arg_frame_width;
	frame_height = arg_frame_height;
	projected_frame_width = arg_projected_frame_width;
	projected_frame_height = arg_projected_frame_height;
	
	cudaStatus = cudaMalloc(&viewpoint_frame_array, number_of_viewpoints*sizeof(ViewPoint_Frame_Pointer));//allocate space for frame array
	if (cudaStatus != cudaSuccess) {
		printf("Failed to malloc frame_array!\n");
		return NULL;
	}

	for (unsigned char i = 0; i < number_of_viewpoints; ++i){
		//setup frame info locally (point to device allocations)
		ViewPoint_Frame_Pointer viewpoint_frame_info;
		cudaMalloc(&viewpoint_frame_info.left.frame_0, arg_frame_width*arg_frame_height * 3 * sizeof(unsigned char));
		cudaMalloc(&viewpoint_frame_info.left.frame_1, arg_frame_width*arg_frame_height * 3 * sizeof(unsigned char));
		viewpoint_frame_info.left.selected_frame = 0;
		cudaMalloc(&viewpoint_frame_info.right.frame_0, arg_frame_width*arg_frame_height * 3 * sizeof(unsigned char));
		cudaMalloc(&viewpoint_frame_info.right.frame_1, arg_frame_width*arg_frame_height * 3 * sizeof(unsigned char));
		viewpoint_frame_info.right.selected_frame = 0;
		//copy over local frame info to device memory
		cudaStatus = cudaMemcpy(&viewpoint_frame_array[i], &viewpoint_frame_info, sizeof(ViewPoint_Frame_Pointer), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf("Failed to copy Frame_Info!\n");
			return NULL;
		}
	}

	cudaStatus = cudaMallocHost(&projected_frame_host.left, arg_projected_frame_width*arg_projected_frame_height * 4 * sizeof(unsigned char));//allocate the host's projected frame as pinned memory
	if (cudaStatus != cudaSuccess) {
		printf("Failed to malloc projected_frame_host left!\n");
		return NULL;
	}
	cudaStatus = cudaMallocHost(&projected_frame_host.right, arg_projected_frame_width*arg_projected_frame_height * 4 * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		printf("Failed to malloc projected_frame_host right!\n");
		return NULL;
	}

	cudaStatus = cudaMalloc(&projected_frame.left, arg_projected_frame_width*arg_projected_frame_height * 4 * sizeof(unsigned char));//allocate the gpu's projected frame
	if (cudaStatus != cudaSuccess) {
		printf("Failed to malloc projected_frame left!\n");
		return NULL;
	}
	cudaStatus = cudaMalloc(&projected_frame.right, arg_projected_frame_width*arg_projected_frame_height * 4 * sizeof(unsigned char));//allocate the gpu's projected frame
	if (cudaStatus != cudaSuccess) {
		printf("Failed to malloc projected_frame right!\n");
		return NULL;
	}

	//compute the number of blocks to assign 1 pixel per thread
	BLOCKS_USED = (projected_frame_width*projected_frame_height) / THREADS_PER_BLOCK_USED;
	if (BLOCKS_USED > BLOCKS_MAX){
		printf("Require %d blocks, greater than BLOCKS_MAX.\n", BLOCKS_USED);//TODO in this case, each thread should do multiple pixels
		return NULL;
	}

	return &projected_frame_host;
}

//given a pointer to an image on the host memory, copy to the currently unused frame buffer for that frame in the device memory, and update the selected frame indicator
//don't need syncronization like in non-gpu code, because we're not changing the pointer, but writing inplace to the buffer, which is thread safe. Worst case 1 corrupted pixel.
//updating selected frame must be atomic though
void copy_new_frame(unsigned char view, bool left_eye, unsigned char* image_data){
	//since we can't dereference device memory on host code
	Frame_Pointer frame;
	if (left_eye){
		cudaMemcpy(&frame, &viewpoint_frame_array[view].left, sizeof(Frame_Pointer), cudaMemcpyDeviceToHost);//get addresses in device memory for the images
	}else{
		cudaMemcpy(&frame, &viewpoint_frame_array[view].right, sizeof(Frame_Pointer), cudaMemcpyDeviceToHost);
	}

	switch (frame.selected_frame){
		case 0:
			cudaMemcpy2DAsync(frame.frame_1, frame_width*sizeof(unsigned char) * 3, image_data, frame_width*sizeof(unsigned char) * 3, frame_width*sizeof(unsigned char) * 3, frame_height, cudaMemcpyHostToDevice);
			frame.selected_frame = 1;
			break;
		case 1:
			cudaMemcpy2DAsync(frame.frame_0, frame_width*sizeof(unsigned char) * 3, image_data, frame_width*sizeof(unsigned char) * 3, frame_width*sizeof(unsigned char) * 3, frame_height, cudaMemcpyHostToDevice);
			frame.selected_frame = 0;
			break;
	}

	if (left_eye){
		cudaMemcpy(&(viewpoint_frame_array[view].left.selected_frame), &frame.selected_frame, sizeof(unsigned char), cudaMemcpyHostToDevice);//TODO atomic
	}
	else{
		cudaMemcpy(&(viewpoint_frame_array[view].right.selected_frame), &frame.selected_frame, sizeof(unsigned char), cudaMemcpyHostToDevice);//TODO atomic
	}
}

//copy the generated projected frame stored on the gpu to the cpu memory
//use pinned host memory, non-waited async copy (because like kernal, longer to check if copied than to actually copy), and multiple streams (in form of memcpy2D)
//TODO can this be optimized so that only the necissary eye is copied?
void read_projected_frame(){
	cudaMemcpy2DAsync(projected_frame_host.left, projected_frame_width*sizeof(unsigned char) * 4, projected_frame.left, projected_frame_width*sizeof(unsigned char) * 4, projected_frame_width*sizeof(unsigned char) * 4, projected_frame_height, cudaMemcpyDeviceToHost);
	cudaMemcpy2DAsync(projected_frame_host.right, projected_frame_width*sizeof(unsigned char) * 4, projected_frame.right, projected_frame_width*sizeof(unsigned char) * 4, projected_frame_width*sizeof(unsigned char) * 4, projected_frame_height, cudaMemcpyDeviceToHost);
}

//helper function for getting pixel data from frame
__device__ __forceinline__ void Get_Pixel_From_Frame(Frame_Pointer frame, int yPixel, int xPixel, int frame_width,
													 unsigned char* pixel_out)
{
	switch (frame.selected_frame){
		case 0:
			pixel_out[0] = frame.frame_0[((abs(yPixel)*frame_width + abs(xPixel)) * 3) + 0];
			pixel_out[1] = frame.frame_0[((abs(yPixel)*frame_width + abs(xPixel)) * 3) + 1];
			pixel_out[2] = frame.frame_0[((abs(yPixel)*frame_width + abs(xPixel)) * 3) + 2];
			break;
		case 1:
			pixel_out[0] = frame.frame_1[((abs(yPixel)*frame_width + abs(xPixel)) * 3) + 0];
			pixel_out[1] = frame.frame_1[((abs(yPixel)*frame_width + abs(xPixel)) * 3) + 1];
			pixel_out[2] = frame.frame_1[((abs(yPixel)*frame_width + abs(xPixel)) * 3) + 2];
			break;
	}
}

//this is the function that is parralelized
//each handles a single pixel on the projection screen
__global__ void Project_to_Screen(unsigned int projected_frame_height, unsigned int projected_frame_width, 
								  unsigned int frame_width, unsigned int frame_height,
								  ViewPoint_Frame_Pointer* frame_array, Projected_Frame projected_frame)
{
	unsigned int thread_num = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = thread_num / projected_frame_width;//pixel row(height)
	unsigned int i = thread_num % projected_frame_width;//pixel column(width)

	//http://stackoverflow.com/questions/34250742/converting-a-cubemap-into-equirectangular-panorama
	//inverse mapping

	double u, v; //Normalised texture coordinates, from 0 to 1, starting at lower left corner
	double phi, theta; //Polar coordinates

	//convert x,y cartesian to u,v polar

	//Rows start from the bottom
	v = 1 - ((double)j / projected_frame_height);
	theta = v * CUDART_PI;

	//Columns start from the left
	u = ((double)i / projected_frame_width);
	phi = u * 2 * CUDART_PI;


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
	unsigned char pixel_left[3];
	unsigned char pixel_right[3];
	int xPixel, yPixel;

	//while (1)
	{
		if (xa == 1)
		{
			//Right
			xPixel = (int)((((za + 1.0) / 2.0) - 1.0) * frame_width);
			yPixel = (int)((((ya + 1.0) / 2.0)) * frame_height);
			Get_Pixel_From_Frame(frame_array[right_view].left, yPixel, xPixel, frame_width, pixel_left);
			Get_Pixel_From_Frame(frame_array[right_view].right, yPixel, xPixel, frame_width, pixel_right);
		}
		else if (xa == -1)
		{
			//Left
			xPixel = (int)((((za + 1.0) / 2.0)) * frame_width);
			if (xPixel >= frame_width){
				xPixel = frame_width - 1;
			}
			yPixel = (int)((((ya + 1.0) / 2.0)) * frame_height);

			Get_Pixel_From_Frame(frame_array[left_view].left, yPixel, xPixel, frame_width, pixel_left);
			Get_Pixel_From_Frame(frame_array[left_view].right, yPixel, xPixel, frame_width, pixel_right);
		}
		else if (ya == -1)
		{
			//Up
			xPixel = (int)((((xa + 1.0) / 2.0)) * frame_width);
			yPixel = (int)((((za + 1.0) / 2.0) - 1.0) * frame_height);
			//flip vertical
			yPixel = (frame_height - 1) - abs(yPixel);

			Get_Pixel_From_Frame(frame_array[top_view].left, yPixel, xPixel, frame_width, pixel_left);
			Get_Pixel_From_Frame(frame_array[top_view].right, yPixel, xPixel, frame_width, pixel_right);
		}
		else if (ya == 1)
		{
			//Down
			xPixel = (int)((((xa + 1.0) / 2.0)) * frame_width);
			yPixel = (int)((((za + 1.0) / 2.0)) * frame_height);
			//flip vertical
			yPixel = (frame_height - 1) - abs(yPixel);

			Get_Pixel_From_Frame(frame_array[bottom_view].left, yPixel, xPixel, frame_width, pixel_left);
			Get_Pixel_From_Frame(frame_array[bottom_view].right, yPixel, xPixel, frame_width, pixel_right);
		}
		else if (za == 1)
		{
			//Front
			xPixel = (int)((((xa + 1.0) / 2.0)) * frame_width);
			yPixel = (int)((((ya + 1.0) / 2.0)) * frame_height);

			Get_Pixel_From_Frame(frame_array[front_view].left, yPixel, xPixel, frame_width, pixel_left);
			Get_Pixel_From_Frame(frame_array[front_view].right, yPixel, xPixel, frame_width, pixel_right);
		}
		else if (za == -1)
		{
			//Back
			xPixel = (int)((((xa + 1.0) / 2.0) - 1.0) * frame_width);
			yPixel = (int)((((ya + 1.0) / 2.0)) * frame_height);

			Get_Pixel_From_Frame(frame_array[back_view].left, yPixel, xPixel, frame_width, pixel_left);
			Get_Pixel_From_Frame(frame_array[back_view].right, yPixel, xPixel, frame_width, pixel_right);
		}
		else
		{
			printf("Unknown face, something went wrong");
		}

		//converting to RGBA from BGR, with max A
		projected_frame.left[((j*projected_frame_width + i) * 4) + 0] = pixel_left[2];
		projected_frame.left[((j*projected_frame_width + i) * 4) + 1] = pixel_left[1];
		projected_frame.left[((j*projected_frame_width + i) * 4) + 2] = pixel_left[0];
		projected_frame.left[((j*projected_frame_width + i) * 4) + 3] = 0xFF;

		projected_frame.right[((j*projected_frame_width + i) * 4) + 0] = pixel_right[2];
		projected_frame.right[((j*projected_frame_width + i) * 4) + 1] = pixel_right[1];
		projected_frame.right[((j*projected_frame_width + i) * 4) + 2] = pixel_right[0];
		projected_frame.right[((j*projected_frame_width + i) * 4) + 3] = 0xFF;
	}
}

void cuda_run(){
	Project_to_Screen << <BLOCKS_USED, THREADS_PER_BLOCK_USED >> >(projected_frame_height, projected_frame_width, frame_width, frame_height, viewpoint_frame_array, projected_frame);

	//fuck error checking and blocking, WE'RE GOING FAST
	//we actually don't need it because it takes FAR longer to check if kernal is finished then to actually run kernal

	// Check for any errors launching the kernel
	/*cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching kernal!\n", cudaStatus);
	}*/
}