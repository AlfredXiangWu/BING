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
	void nonMaxSup(ValStructVec<float, Vec4i> &input, ValStructVec<float, Vec4i> &output, float IoU);
	static inline double interUnio(const Vec4i &box1, const Vec4i &box2);

private:
	string _modelPath;
	int _netSize;
	int _netProbLayer;
	Net *_cnn;

};

double CnnFace::interUnio(const Vec4i &bb, const Vec4i &bbgt)
{
	int bi[4];
	bi[0] = max(bb[0], bbgt[0]);
	bi[1] = max(bb[1], bbgt[1]);
	bi[2] = min(bb[2], bbgt[2]);
	bi[3] = min(bb[3], bbgt[3]);	

	double iw = bi[2] - bi[0] + 1;
	double ih = bi[3] - bi[1] + 1;
	double ov = 0;
	if (iw>0 && ih>0){
		double ua = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)+(bbgt[2]-bbgt[0]+1)*(bbgt[3]-bbgt[1]+1)-iw*ih;
		ov = iw*ih/ua;
	}	
	return ov;
}