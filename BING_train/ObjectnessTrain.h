#include "ValStructVec.h"
#include "FilterBING.h"
#include "Dataset.h"


#pragma once

class ObjectnessTrain
{
public:
	ObjectnessTrain(DataSet &dataSet, string modelPath, int W = 8, int NSS = 2)
		: _dataSet(dataSet)
		, _modelPath(modelPath)
		, _W(W)
		, _NSS(NSS){};
	~ObjectnessTrain(void){};
	
	void trainObjectnessModel();


private:
	const int _W;
	const int _NSS;
	
	DataSet _dataSet;
	string _modelPath;

	void generateTrainData();
	void trainStageI();

	Mat getFeature(CMat &img3u, const Vec4i &bb);
	void gradientMag(CMat &imgBGR3u, Mat &mag1u);
	static void gradientRGB(CMat &bgr3u, Mat &mag1u);
	//static void gradientGray(CMat &bgr3u, Mat &mag1u);
	//static void gradientHSV(CMat &bgr3u, Mat &mag1u);
	static void gradientXY(CMat &x1i, CMat &y1i, Mat &mag1u);

	inline double maxIntUnion(const Vec4i &bb, const vector<Vec4i> &bbgts) {double maxV = 0; for(size_t i = 0; i < bbgts.size(); i++) maxV = max(maxV, DataSet::interUnio(bb, bbgts[i])); return maxV; }

	static inline int bgrMaxDist(const Vec3b &u, const Vec3b &v) {int b = abs(u[0]-v[0]), g = abs(u[1]-v[1]), r = abs(u[2]-v[2]); b = max(b,g);  return max(b,r);}

	bool matWrite(CStr& filename, CMat& _M);
	bool matRead(const string& filename, Mat& _M);

};
