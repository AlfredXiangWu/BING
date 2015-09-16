
#include "stdafx.h"
#include "ValStructVec.h"
#include "Dataset.h"
#include "ObjectnessTrain.h"


void trainObjectness(int W, int NSS, int numPerSz);

void main(int argc, char* argv)
{
	trainObjectness(8, 2, 130);
}


void trainObjectness(int W, int NSS, int numPerSz)
{
	// positive samples
	string imgPath  = "Z:\\Face_DB";
	string listPath = "Z:\\User\\wuxiang\\data\\face_detection\\aflw_list.txt";
	string frPath = "Z:\\Face_DB\\fr\\man";
	string modelPath = "D:\\svn\\Algorithm\\wuxiang\\Code\\C\\BING\\model";

	// load train image
	DataSet dataSet(imgPath, listPath, frPath);
	dataSet.loadAnnotations();

	ObjectnessTrain objectNessTrain(dataSet, modelPath, 8, 2);
	objectNessTrain.trainObjectnessModel();

}