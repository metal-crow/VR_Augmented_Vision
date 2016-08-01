#ifndef VRDISPLAY_H
#define VRDISPLAY_H

#include <Windows.h>

//This requires a RGBA input mat (pass in the data pointer, not the Mat itself)
void UpdateTexture(unsigned char* left, unsigned char* right);

bool Initalize_VR(HINSTANCE hinst, unsigned int output_width, unsigned int output_height);

bool Main_VR_Render_Loop();

void Exit_VR();

#endif