#include "StdAfx.h"
#include "ObjectnessTest.h"
#include "CmShow.h"

int ObjectnessTest::loadTrainedModel(string modelPath)
{
	if (modelPath.size() == 0)
		modelPath = _modelPath;
	
	string s1 = modelPath + "\\" + "ObjNessB2W8MAXBGR.wS1";
	string s2 = modelPath + "\\" + "ObjNessB2W8MAXBGR.wS2";


}

// for dataset
void ObjectnessTest::getFaceProposalsForImgsFast(vector<vector<Vec4i>> &_frsImgs, int numDetPerSize)
{

}

// for single image
void ObjectnessTest::getFaceProposaksForPerImgFast(vector<Vec4i> &frsPerImg, int numDetPerSize)
{

}