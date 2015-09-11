#include "StdAfx.h"
#include "ObjectnessTest.h"
#include "CmShow.h"

int ObjectnessTest::loadTrainedModel(string modelPath)
{
	if (modelPath.size() == 0)
		modelPath = _modelPath;
	
	string s1 = modelPath + "\\" + "ObjNessB2W8MAXBGR.wS1";
	string s2 = modelPath + "\\" + "ObjNessB2W8MAXBGR.wS2";
	Mat filters1f, reW1f;
	if (!matRead(s1, filters1f))
	{
		printf("Cannot load model\n");
		return false;
	}

	CV_Assert(filters1f.type() == CV_32F);
	_svmFilter = filters1f;

	return true;
}

// for dataset
void ObjectnessTest::getFaceProposalsForImgsFast(vector<vector<Vec4i>> &_frsImgs, int numDetPerSize)
{

}

// for single image
void ObjectnessTest::getFaceProposaksForPerImgFast(vector<Vec4i> &frsPerImg, int numDetPerSize)
{

}

// Read matrix from binary file
bool ObjectnessTest::matRead(const string& filename, Mat& _M)
{
	FILE* f = fopen(_S(filename), "rb");
	if (f == NULL)
		return false;
	char buf[8];
	int pre = fread(buf,sizeof(char), 5, f);
	if (strncmp(buf, "CmMat", 5) != 0)	{
		printf("Invalidate CvMat data file %s\n", _S(filename));
		return false;
	}
	int headData[3]; // Width, height, type
	fread(headData, sizeof(int), 3, f);
	Mat M(headData[1], headData[0], headData[2]);
	fread(M.data, sizeof(char), M.step * M.rows, f);
	fclose(f);
	M.copyTo(_M);
	return true;
}