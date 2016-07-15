#ifndef VRDISPLAY_H
#define VRDISPLAY_H

#include <Windows.h>
#include "opencv2\core.hpp"

void UpdateTexture(cv::Mat input);

bool Initalize_VR(HINSTANCE hinst);

bool Main_VR_Render_Loop();

void Exit_VR();

#endif