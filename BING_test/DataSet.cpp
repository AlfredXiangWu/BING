#include "Dataset.h"
#include "stdafx.h"

#pragma once

DataSet::DataSet(CStr  &_imgPath, CStr &_listPath, CStr &_frPath)
{
	imgPath = _imgPath;
	listPath = _listPath;
	frPath = _frPath;
	
	imgPathName = loadStrList(listPath);
	imgPathFr = loadFrList(imgPathName);

	testNum = imgPathName.size();
}

string DataSet::replaceExtName(CStr &fName, char* ext)
{
	int i, len, idx;
	char *id;
	const int num = fName.length();
	id = new char[num];
	strcpy(id, fName.c_str());

	len = (int) strlen(id);
	for(i=len-1; i>=0; i--)
	{
		if(id[i] == '.')
		{
			break;
		}
	}
	idx = i + 1;
	len = (int) strlen(ext);
	for(i=0; i<len; i++)
	{
		id[idx+i] = ext[i];
	}
	id[idx+len] = 0;

	return string(id);
}

vecS DataSet::loadStrList(CStr &fName)
{
	ifstream fIn(fName);
	string line;
	vecS strs;
	while(getline(fIn, line) && line.size())
		strs.push_back(line);
	fIn.close();
	return strs;
}

vecS DataSet::loadFrList(vecS &imgPathName)
{
	string line;
	vecS strs;
	for (int i = 0; i < imgPathName.size(); i++)
	{
		line = replaceExtName(imgPathName[i], "fr");
		strs.push_back(line);
	}
	return strs;
}


void DataSet::loadAnnotations()
{
	vector<Vec4i> boxes;
	for (int i = 0; i < testNum; i++)
	{
		if(!loadFrs((frPath + "\\" + imgPathFr[i]), boxes))
			printf("Load %s error\n", imgPathFr[i]);
		imgFr.push_back(boxes);
		boxes.clear();
	}
}

int DataSet::loadFrs(CStr &frName, vector<Vec4i> &boxes)
{
	ifstream frIn(frName);
	string line;
	getline(frIn, line);
	int numFace = atoi(line.c_str());
	for (int i = 0; i < numFace; i++)
	{
		if(!(getline(frIn, line) && line.size()))
			return false;
		vecS temp = stringSplit(line, string("\t"));
		if (temp.size() !=4)
			return false;
		boxes.push_back(Vec4i(atoi(temp[0].c_str()), atoi(temp[1].c_str()), atoi(temp[2].c_str()), atoi(temp[3].c_str())));
	}
	frIn.close();
	return true;
}

vecS DataSet::stringSplit(string &str, string &pattern)
{
	vecS ret;
	std::string::size_type pos;
	str += pattern;
	int length = str.length();
	int start = 0;

	for (int i = 0; i < length; i++)
	{
		pos = str.find(pattern, i);
		if(pos < length)
		{
			string s = str.substr(i, pos - i);
			ret.push_back(s);
			i = pos + pattern.size() - 1;
		}
	}
	return ret;
}