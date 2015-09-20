#include "stdafx.h"

#pragma once

class CnnFace
{
public:
	CnnFace(string modelPath)
		:_modelPath(modelPath){};
	~CnnFace(){};

	int loadTrainedModel();

private:
	string _modelPath;

};