#include "stdafx.h"

#pragma once

class DataSet
{
public:
	DataSet(CStr  &imgPath, CStr &listPath, CStr &frPath);
	~DataSet(void){};
	
	void loadAnnotations();
	static inline double interUnio(const Vec4i &box1, const Vec4i &box2);

	string imgPath;
	string listPath;
	string frPath;
	vecS imgPathFr;
	int testNum;
	vecS imgPathName;
	vector<vector<Vec4i>> imgFr;


private:
	string replaceExtName(CStr &fName, char* ext);
	vecS loadStrList(CStr &fName);
	vecS loadFrList(vecS &imgPathName);

	int loadFrs(CStr &frName, vector<Vec4i> &boxes);
	vecS stringSplit(string &str, string &pattern);
};

double DataSet::interUnio(const Vec4i &bb, const Vec4i &bbgt)
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
