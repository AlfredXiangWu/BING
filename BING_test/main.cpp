
#include "stdafx.h"
#include "ValStructVec.h"
#include "Objectness_test.h"
#include "Dataset.h"

void RunFaceProposal(int W, int NSS, int numPerSz);

void main(int argc, char* argv[])
{
	RunFaceProposal(8, 2, 130);
}

void RunFaceProposal(int W, int NSS, int numPerSz)
{
	string imgPath  = "Z:/User/wuxiang/data/face_detection/FDDB/originalPics";
	string listPath = "Z:/User/wuxiang/data/face_detection/FDDB/FDDB_list.txt";
	string frPath = "Z:/User/team02/data1/FDDB/man";

	DataSet fddb(imgPath, listPath, frPath);


}