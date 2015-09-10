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

