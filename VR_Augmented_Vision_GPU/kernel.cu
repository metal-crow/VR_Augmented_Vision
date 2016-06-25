#include "Kernal.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include "math.h"

#include <stdio.h>

unsigned char number_of_cameras;
unsigned int frame_width, frame_height;

unsigned int projected_frame_width, projected_frame_height;
unsigned char* projected_frame;//dont need a second buffer because memcpy is thread irrelivent, so worst case a single pixel is corrupted

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

Frame_Info* frame_array; 

//allocate space for the array of images, and the buffer images for each camera, and the final projected image
//also set some global constants
void allocate_frames(unsigned char arg_number_of_cameras, unsigned int arg_frame_width, unsigned int arg_frame_height, unsigned int arg_projected_frame_width, unsigned int arg_projected_frame_height){
	cudaMalloc(&frame_array, arg_number_of_cameras*sizeof(Frame_Info));

	number_of_cameras = arg_number_of_cameras;
	frame_width = arg_frame_width;
	frame_height = arg_frame_height;
	projected_frame_width = arg_projected_frame_width;
	projected_frame_height = arg_projected_frame_height;

	for (unsigned int i = 0; i < arg_number_of_cameras; ++i){
		cudaMalloc(&frame_array[i].frame_0, arg_frame_width*arg_frame_height*3*sizeof(unsigned char));
		cudaMalloc(&frame_array[i].frame_1, arg_frame_width*arg_frame_height*3*sizeof(unsigned char));
	}

	cudaMalloc(&projected_frame, arg_projected_frame_width*arg_projected_frame_height * 3 * sizeof(unsigned char));
}

//given a pointer to an image on the host memory, copy to the currently unused frame buffer for that frame in the device memory
void copy_new_frame(unsigned char camera, unsigned char* image_data){
	switch (frame_array[camera].selected_frame){
		case 0:
			cudaMemcpy(frame_array[camera].frame_0, image_data, frame_width*frame_height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
			break;
		case 1:
			cudaMemcpy(frame_array[camera].frame_1, image_data, frame_width*frame_height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
			break;
	}
}

//TODO cuda atomic integer alter
void swap_current_frame_buffer(unsigned char camera){

}

//this is the function that is parralelized
//each handles a single pixel on the projection screen
//TODO might have to be more like 4 pixels? Test various numbers
__global__ void Project_to_Screen(){
	//http://stackoverflow.com/questions/34250742/converting-a-cubemap-into-equirectangular-panorama
	//inverse mapping

	double u, v; //Normalised texture coordinates, from 0 to 1, starting at lower left corner
	double phi, theta; //Polar coordinates

	//convert x,y cartesian to u,v polar

	unsigned int j = 0;//pixel row(height)
	//Rows start from the bottom
	v = 1 - ((double)j / projected_frame_height);
	theta = v * CUDART_PI_F;

	unsigned int i = 0;//pixel column(width)
	//Columns start from the left
	u = ((double)i / projected_frame_width);
	phi = u * 2 * CUDART_PI_F;


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
	int xOffset, yOffset;

	while (1)
	{
		if (xa == 1)
		{
			//Right
			xPixel = (int)((((za + 1.0) / 2.0) - 1.0) * frame_width);
			yPixel = (int)((((ya + 1.0) / 2.0)) * frame_height);

			DWORD locked = WaitForSingleObject(frame_array[right_frame].lock, 10);
			if (locked == WAIT_OBJECT_0){
				pixel = (*frame_array[right_frame].frame).at<Vec3b>(abs(yPixel), abs(xPixel));
			}
			ReleaseMutex(frame_array[right_frame].lock);
		}
		else if (xa == -1)
		{
			//Left
			xPixel = (int)((((za + 1.0) / 2.0)) * frame_width);
			yPixel = (int)((((ya + 1.0) / 2.0)) * frame_height);

			DWORD locked = WaitForSingleObject(frame_array[left_frame].lock, 10);
			if (locked == WAIT_OBJECT_0){
				pixel = (*frame_array[left_frame].frame).at<Vec3b>(abs(yPixel), abs(xPixel));
			}
			ReleaseMutex(frame_array[left_frame].lock);
		}
		else if (ya == 1)
		{
			//Up
			xPixel = (int)((((xa + 1.0) / 2.0)) * frame_width);
			yPixel = (int)((((za + 1.0) / 2.0) - 1.0) * frame_height);

			DWORD locked = WaitForSingleObject(frame_array[top_frame].lock, 10);
			if (locked == WAIT_OBJECT_0){
				pixel = (*frame_array[top_frame].frame).at<Vec3b>(abs(yPixel), abs(xPixel));
			}
			ReleaseMutex(frame_array[top_frame].lock);
		}
		else if (ya == -1)
		{
			//Down
			xPixel = (int)((((xa + 1.0) / 2.0)) * frame_width);
			yPixel = (int)((((za + 1.0) / 2.0)) * frame_height);

			DWORD locked = WaitForSingleObject(frame_array[bottom_frame].lock, 10);
			if (locked == WAIT_OBJECT_0){
				pixel = (*frame_array[bottom_frame].frame).at<Vec3b>(abs(yPixel), abs(xPixel));
			}
			ReleaseMutex(frame_array[bottom_frame].lock);
		}
		else if (za == 1)
		{
			//Front
			xPixel = (int)((((xa + 1.0) / 2.0)) * frame_width);
			yPixel = (int)((((ya + 1.0) / 2.0)) * frame_height);

			DWORD locked = WaitForSingleObject(frame_array[front_frame].lock, 10);
			if (locked == WAIT_OBJECT_0){
				pixel = (*frame_array[front_frame].frame).at<Vec3b>(abs(yPixel), abs(xPixel));
			}
			ReleaseMutex(frame_array[front_frame].lock);
		}
		else if (za == -1)
		{
			//Back
			xPixel = (int)((((xa + 1.0) / 2.0) - 1.0) * frame_width);
			yPixel = (int)((((ya + 1.0) / 2.0)) * frame_height);

			DWORD locked = WaitForSingleObject(frame_array[back_frame].lock, 10);
			if (locked == WAIT_OBJECT_0){
				pixel = (*frame_array[back_frame].frame).at<Vec3b>(abs(yPixel), abs(xPixel));
			}
			ReleaseMutex(frame_array[back_frame].lock);
		}
		else
		{
			printf("Unknown face, something went wrong");
		}

		projected_frame[(j*i*3)+0] = pixel[0];
		projected_frame[(j*i*3)+1] = pixel[1];
		projected_frame[(j*i*3)+2] = pixel[2];
	}
}




cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int cuda_run()
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
