#include "Kernal.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include "math.h"

#include <stdio.h>
#include <stdlib.h>

unsigned char number_of_cameras;
unsigned int frame_width, frame_height;

unsigned int projected_frame_width, projected_frame_height;
unsigned char* projected_frame;//dont need a second buffer because memcpy is thread agnostic, so worst case a single pixel is corrupted

enum camera_namesE{
	top_frame = 0,
	bottom_frame = 1,
	front_frame = 2,
	left_frame = 3,
	right_frame = 4,
	back_frame = 5,
} camera_names;

typedef struct{
	unsigned char* frame_0;//these two serve as frame buffers
	unsigned char* frame_1;
	unsigned char selected_frame;//this is used by the thread to get pixel data from the current frame. Must be in range 0 to number_of buffer frames-1
} Frame_Info;

Frame_Info* frame_array;//pointer stored on host memory, which points to device memory

#define THREADS_PER_BLOCK_MAX 1024 //note this this usually can't be used since the number of available registers is usually the lower bound
#define THREADS_PER_BLOCK_USED 512

#define BLOCKS_MAX 65535 
unsigned int BLOCKS_USED;

//allocate space for the array of images, and the buffer images for each camera, and the final projected image
//also set some global constants
int allocate_frames(unsigned char arg_number_of_cameras, 
					 unsigned int arg_frame_width, unsigned int arg_frame_height, 
					 unsigned int arg_projected_frame_width, unsigned int arg_projected_frame_height)
{
	//save these variables
	number_of_cameras = arg_number_of_cameras;
	frame_width = arg_frame_width;
	frame_height = arg_frame_height;
	projected_frame_width = arg_projected_frame_width;
	projected_frame_height = arg_projected_frame_height;
	
	cudaError_t cudaStatus = cudaMalloc(&frame_array, arg_number_of_cameras*sizeof(Frame_Info));//allocate space for frame array
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	for (unsigned char i = 0; i < arg_number_of_cameras; ++i){
		//setup frame info locally (point to device allocations)
		Frame_Info camera_frame_info;
		cudaMalloc(&camera_frame_info.frame_0, arg_frame_width*arg_frame_height * 3 * sizeof(unsigned char));
		cudaMalloc(&camera_frame_info.frame_1, arg_frame_width*arg_frame_height * 3 * sizeof(unsigned char));
		camera_frame_info.selected_frame = 0;
		//copy over local frame info to device memory
		cudaStatus = cudaMemcpy(&frame_array[i], &camera_frame_info, sizeof(Frame_Info), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf("cudaMalloc failed!");
		}
	}

	cudaStatus = cudaMalloc(&projected_frame, arg_projected_frame_width*arg_projected_frame_height * 3 * sizeof(unsigned char));//allocate the projected frame
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	//compute the number of blocks to assign 1 pixel per thread
	BLOCKS_USED = (projected_frame_width*projected_frame_height) / THREADS_PER_BLOCK_USED;
	if (BLOCKS_USED > BLOCKS_MAX){
		printf("Require %d blocks, greater than BLOCKS_MAX.\n", BLOCKS_USED);//TODO in this case, each thread should do multiple pixels
		return 1;
	}

	return 0;
}

//given a pointer to an image on the host memory, copy to the currently unused frame buffer for that frame in the device memory, and update the selected frame indicator
//don't need syncronization like in non-gpu code, because we're not changing the pointer, but writing inplace to the buffer, which is thread safe. Worst case 1 corrupted pixel.
//updating selected frame must be atomic though
void copy_new_frame(unsigned char camera, unsigned char* image_data){
	//since we can't dereference device memory on host code
	Frame_Info frame;
	cudaMemcpy(&frame, &frame_array[camera], sizeof(Frame_Info), cudaMemcpyDeviceToHost);//get addresses in device memory for the images

	switch (frame.selected_frame){
		case 0:
			cudaMemcpy(frame.frame_1, image_data, frame_width*frame_height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
			frame.selected_frame = 1;
			break;
		case 1:
			cudaMemcpy(frame.frame_0, image_data, frame_width*frame_height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
			frame.selected_frame = 0;
			break;
	}

	cudaMemcpy(&(frame_array[camera].selected_frame), &frame.selected_frame, sizeof(unsigned char), cudaMemcpyHostToDevice);//TODO atomic
}

//copy the generated projected frame stored on the gpu to the cpu memory
void read_projected_frame(unsigned char*  host_projection_frame){
	cudaError_t cudaStatus = cudaMemcpy(host_projection_frame, projected_frame, projected_frame_width*projected_frame_height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
}

//this is the function that is parralelized
//each handles a single pixel on the projection screen
__global__ void Project_to_Screen(unsigned int projected_frame_height, unsigned int projected_frame_width, 
								  unsigned int frame_width, unsigned int frame_height,
								  Frame_Info* frame_array, unsigned char* projected_frame)
{
	unsigned int thread_num = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = thread_num / projected_frame_height;//pixel row(height)
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

	unsigned char pixel[3];
	int xPixel, yPixel;

	//while (1)
	{
		if (xa == 1)
		{
			//Right
			xPixel = (int)((((za + 1.0) / 2.0) - 1.0) * frame_width);
			yPixel = (int)((((ya + 1.0) / 2.0)) * frame_height);

			switch (frame_array[right_frame].selected_frame){
				case 0:
					pixel[0] = frame_array[right_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 0];
					pixel[1] = frame_array[right_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 1];
					pixel[2] = frame_array[right_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 2];
					break;
				case 1:
					pixel[0] = frame_array[right_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 0];
					pixel[1] = frame_array[right_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 1];
					pixel[2] = frame_array[right_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 2];
					break;
			}
			/*pixel[0] = 0;
			pixel[1] = 0;
			pixel[2] = 255;//red*/
		}
		else if (xa == -1)
		{
			//Left
			xPixel = (int)((((za + 1.0) / 2.0)) * frame_width);
			yPixel = (int)((((ya + 1.0) / 2.0)) * frame_height);

			switch (frame_array[left_frame].selected_frame){
				case 0:
					pixel[0] = frame_array[left_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 0];
					pixel[1] = frame_array[left_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 1];
					pixel[2] = frame_array[left_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 2];
					break;
				case 1:
					pixel[0] = frame_array[left_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 0];
					pixel[1] = frame_array[left_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 1];
					pixel[2] = frame_array[left_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 2];
					break;
			}
			/*pixel[0] = 0;
			pixel[1] = 255;
			pixel[2] = 255;//yellow*/
		}
		else if (ya == -1)
		{
			//Up
			xPixel = (int)((((xa + 1.0) / 2.0)) * frame_width);
			yPixel = (int)((((za + 1.0) / 2.0) - 1.0) * frame_height);
			//flip vertical
			yPixel = (frame_height - 1) - abs(yPixel);

			switch (frame_array[top_frame].selected_frame){
				case 0:
					pixel[0] = frame_array[top_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 0];
					pixel[1] = frame_array[top_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 1];
					pixel[2] = frame_array[top_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 2];
					break;
				case 1:
					pixel[0] = frame_array[top_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 0];
					pixel[1] = frame_array[top_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 1];
					pixel[2] = frame_array[top_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 2];
					break;
			}
			/*pixel[0] = 0;
			pixel[1] = 60;
			pixel[2] = 255;//orange*/
		}
		else if (ya == 1)
		{
			//Down
			xPixel = (int)((((xa + 1.0) / 2.0)) * frame_width);
			yPixel = (int)((((za + 1.0) / 2.0)) * frame_height);
			//flip vertical
			yPixel = (frame_height - 1) - abs(yPixel);

			switch (frame_array[bottom_frame].selected_frame){
				case 0:
					pixel[0] = frame_array[bottom_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 0];
					pixel[1] = frame_array[bottom_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 1];
					pixel[2] = frame_array[bottom_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 2];
					break;
				case 1:
					pixel[0] = frame_array[bottom_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 0];
					pixel[1] = frame_array[bottom_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 1];
					pixel[2] = frame_array[bottom_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 2];
					break;
			}
			/*pixel[0] = 255;
			pixel[1] = 0;
			pixel[2] = 0;//blue*/
		}
		else if (za == 1)
		{
			//Front
			xPixel = (int)((((xa + 1.0) / 2.0)) * frame_width);
			yPixel = (int)((((ya + 1.0) / 2.0)) * frame_height);

			switch (frame_array[front_frame].selected_frame){
				case 0:
					pixel[0] = frame_array[front_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 0];
					pixel[1] = frame_array[front_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 1];
					pixel[2] = frame_array[front_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 2];
					break;
				case 1:
					pixel[0] = frame_array[front_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 0];
					pixel[1] = frame_array[front_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 1];
					pixel[2] = frame_array[front_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 2];
					break;
			}
			/*pixel[0] = 150;
			pixel[1] = 150;
			pixel[2] = 150;//grey*/
		}
		else if (za == -1)
		{
			//Back
			xPixel = (int)((((xa + 1.0) / 2.0) - 1.0) * frame_width);
			yPixel = (int)((((ya + 1.0) / 2.0)) * frame_height);

			switch (frame_array[back_frame].selected_frame){
				case 0:
					pixel[0] = frame_array[back_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 0];
					pixel[1] = frame_array[back_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 1];
					pixel[2] = frame_array[back_frame].frame_0[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 2];
					break;
				case 1:
					pixel[0] = frame_array[back_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 0];
					pixel[1] = frame_array[back_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 1];
					pixel[2] = frame_array[back_frame].frame_1[(abs(yPixel)*frame_width * 3 + abs(xPixel) * 3) + 2];
					break;
			}
			/*pixel[0] = 150;
			pixel[1] = 0;
			pixel[2] = 0;//light blue*/
		}
		else
		{
			printf("Unknown face, something went wrong");
		}

		projected_frame[((j*projected_frame_width + i) * 3) + 0] = pixel[0];
		projected_frame[((j*projected_frame_width + i) * 3) + 1] = pixel[1];
		projected_frame[((j*projected_frame_width + i) * 3) + 2] = pixel[2];
	}
}

void cuda_run(){
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	Project_to_Screen << <BLOCKS_USED, THREADS_PER_BLOCK_USED >> >(projected_frame_height, projected_frame_width, frame_width, frame_height, frame_array, projected_frame);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching kernal!\n", cudaStatus);
	}
}


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int a()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
