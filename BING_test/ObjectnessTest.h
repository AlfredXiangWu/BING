#include "ValStructVec.h"
#include "FilterBING.h"
#include "Dataset.h"

#pragma once

class ObjectnessTest
{
public:
	ObjectnessTest(DataSet &dataSet, string modelPath, int W = 8, int NSS = 2)
		: _dataSet(dataSet)
		, _modelPath(modelPath)
		, _W(W)
		, _NSS(NSS){};
	~ObjectnessTest(void){};

	int loadTrainedModel(string modelPath);
	void getFaceProposalsForImgsFast(vector<vector<Vec4i>> &_frsImgs, int numDetPerSize);
	void getFaceProposaksForPerImgFast(Mat &img3u, vector<Vec4i> &frsPerImg, int numDetPerSize);

	void evaluatePerImgRecall(const vector<vector<Vec4i>> &boxesTests, const int numDet);
	void illuTestReults(string &imgPath, string &savePath, const vector<Vec4i> &gtBoxesTest, const vector<Vec4i> &boxesTests);

private:
	const int _W;
	const int _NSS;
	
	DataSet _dataSet;
	string _modelPath;
	Mat _svmFilter; //Filters learned at stage I
	FilterBING _bingF;   // BING Filter

	bool matRead(const string& filename, Mat& _M);

	void gradientMag(CMat &imgBGR3u, Mat &mag1u);
	static void gradientRGB(CMat &bgr3u, Mat &mag1u);
	//static void gradientGray(CMat &bgr3u, Mat &mag1u);
	//static void gradientHSV(CMat &bgr3u, Mat &mag1u);
	static void gradientXY(CMat &x1i, CMat &y1i, Mat &mag1u);

	static inline int bgrMaxDist(const Vec3b &u, const Vec3b &v) {int b = abs(u[0]-v[0]), g = abs(u[1]-v[1]), r = abs(u[2]-v[2]); b = max(b,g);  return max(b,r);}

	//Non-maximal suppress
	static void nonMaxSup(CMat &matchCost1f, ValStructVec<float, Point> &matchCost, int NSS = 1, int maxPoint = 50, bool fast = true);
};
