#include "stdafx.h"
#include "ValStructVec.h"

#pragma once

class CnnFace
{
public:
	CnnFace(string modelPath, int netSize, int netProbLayer)
		:_modelPath(modelPath)
		,_netSize(netSize)
		,_netProbLayer(netProbLayer){};
	~CnnFace(){};

	int loadTrainedModel();
	void getFaceDetectionPerImg(Mat &img, vector<Vec4i> &boxProposal, ValStructVec<float, Vec4i> &valBoxes, float thr);

private:
	string _modelPath;
	int _netSize;
	int _netProbLayer;
	Net *_cnn;

};