#ifndef KERNEL_H
#define KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

	__declspec(dllexport) void* allocate_frames(unsigned char arg_number_of_viewpoints, unsigned int arg_frame_width, unsigned int arg_frame_height, unsigned int arg_projected_frame_width, unsigned int arg_projected_frame_height);
	void __declspec(dllexport) copy_new_frame(unsigned char view, bool left_eye, unsigned char* image_data);
	void __declspec(dllexport) read_projected_frame();
	void __declspec(dllexport) cuda_run();

#ifdef __cplusplus
}
#endif

#endif 