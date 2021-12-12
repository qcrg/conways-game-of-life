#pragma once
#include "ModuleM.h"
#include "constants.h"

class GameTicker
{
public:
	GameTicker(Data& ptr);

	void runOneStep();
private:
	Data& data;
	cudaDeviceProp currentDevice;
};