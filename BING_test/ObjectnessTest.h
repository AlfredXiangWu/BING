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
	void getFaceProposaksForPerImgFast(vector<Vec4i> &frsPerImg, int numDetPerSize);


private:
	const int _W;
	const int _NSS;
	
	DataSet _dataSet;
	string _modelPath;
	Mat _svmFilter; //Filters learned at stage I
	FilterBING _bingF;   // BING Filter

	bool matRead(const string& filename, Mat& _M);
};
