
#include "stdafx.h"
#include "ValStructVec.h";
#include "ObjectnessTest.h"
#include "Dataset.h"

void RunFaceProposal(int W, int NSS, int numPerSz);

void main(int argc, char* argv[])
{
	RunFaceProposal(8, 2, 130);
}

void RunFaceProposal(int W, int NSS, int numPerSz)
{
	string imgPath  = "Z:\\User\\wuxiang\\data\\face_detection\\FDDB\\originalPics";
	string listPath = "Z:\\User\\wuxiang\\data\\face_detection\\FDDB\\test.txt";
	string frPath = "Z:\\User\\team02\\data1\\FDDB\\man";
	string modelPath = "D:\\svn\\Algorithm\\wuxiang\\Code\\C\\BING\\model";

	vector<vector<Vec4i>> frsImgs;

	DataSet dataSet(imgPath, listPath, frPath);
	dataSet.loadAnnotations();

	ObjectnessTest objectNessTest(dataSet, modelPath, 8, 2);
	
	objectNessTest.loadTrainedModel(modelPath);
	objectNessTest.getFaceProposalsForImgsFast(frsImgs, 130);
}