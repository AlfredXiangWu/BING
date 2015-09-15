
#include "stdafx.h"
#include "ValStructVec.h"
#include "ObjectnessTest.h"
#include "Dataset.h"


void RunFaceProposal(int W, int NSS, int numPerSz);

static int create_directory(const char *directory);
static char ** chTwoDMalloc(int iImageX, int iImageY);
static int get_dir_from_filename(const char *file_name, char *dir);
static int create_file(const char *file_name, const char type);
static void TwoDMFree(int iImageY, void **pMem);


void main(int argc, char* argv[])
{
	RunFaceProposal(8, 2, 130);
}

void RunFaceProposal(int W, int NSS, int numPerSz)
{
	// configuration
	string imgPath  = "Z:\\User\\wuxiang\\data\\face_detection\\FDDB\\originalPics";
	string listPath = "Z:\\User\\wuxiang\\data\\face_detection\\FDDB\\test.txt";
	string frPath = "Z:\\User\\team02\\data1\\FDDB\\man";
	string modelPath = "D:\\svn\\Algorithm\\wuxiang\\Code\\C\\BING\\model";

	string savePath = "D:\\BING";

	vector<vector<Vec4i>> frsImgs;
	char fr[_MAX_PATH];

	// load image
	DataSet dataSet(imgPath, listPath, frPath);
	dataSet.loadAnnotations();

	// predict
	ObjectnessTest objectNessTest(dataSet, modelPath, 8, 2);
	objectNessTest.loadTrainedModel(modelPath);
	objectNessTest.getFaceProposalsForImgsFast(frsImgs, 130);

	// save 
	for (int i = 0; i < frsImgs.size(); i++)
	{
		sprintf(fr, "%s/%s", savePath.c_str(), dataSet.imgPathFr[i].c_str());
		create_file(fr, 'b');
		FILE *fp = fopen(fr, "wt");
		fprintf(fp, "%d\n", frsImgs[i].size());
		for (int j = 0; j < frsImgs[i].size(); j++)
		{
			Vec4i box = frsImgs[i][j];
			fprintf(fp, "%d\t%d\t%d\t%d\n", box[0], box[1], box[2], box[3]);
		}
		fclose(fp);
	}

}

static int create_directory(const char *directory)
{
  int i;
  int len;
  char dir[_MAX_PATH], temp_dir[_MAX_PATH];

  memcpy(temp_dir, directory, _MAX_PATH);

  len = (int)strlen(temp_dir);
  for(i=0; i<len; i++)
  {
    if(temp_dir[i] == '\\')
      temp_dir[i] = '/';
  }
  if(temp_dir[len-1] != '/')
  {
    temp_dir[len] = '/';
    temp_dir[len+1] = 0;
    len++;
  }
  memset(dir, 0, _MAX_PATH);
  for(i=0; i<len; i++)
  {
    dir[i] = temp_dir[i];
    if(temp_dir[i] == '/')
    {
      if(i > 0)
      {
        if(temp_dir[i-1] == ':')
          continue;
        else
        {
			if(_access(dir, 0) == 0)
            continue;
          else /* create it */
          {
            if(_mkdir(dir) != 0)
              return -1;
          }
        }
      }
    }
  }

  return 0;
}


static char ** chTwoDMalloc(int iImageX, int iImageY)
{
  int i, j;
  char **pchMem = NULL;
  if((pchMem = (char **)malloc(iImageY*sizeof(char *))) == NULL)
    return NULL;
  for(i=0; i<iImageY; i++)
  {
    pchMem[i] = NULL;
    if((pchMem[i] = (char *)malloc(iImageX)) == NULL)
    {
      for(j=0; j<i; j++)
      {
        free(pchMem[j]);
      }
      free(pchMem);
      return NULL;
    }
  }
  return pchMem;
}

static int get_dir_from_filename(const char *file_name, char *dir)
{
  int len;
  int i;

  len = (int) strlen(file_name);
  for(i=len-1; i>=0; i--)
  {
    if(file_name[i] == '\\' || file_name[i] == '/')
    {
      break;
    }
  }

  strcpy(dir, file_name);
  dir[i+1] = 0;

  return 0;
}

static int create_file(const char *file_name, const char type)
{
  FILE *fp;
  char dir[_MAX_PATH];
  char mode[5];

  if(type == 'b')
  {
    strcpy(mode, "wb");
  }
  else
  {
    strcpy(mode, "w");
  }

  fp = fopen(file_name, mode);
  if(fp == NULL)
  {
    get_dir_from_filename(file_name, dir);
    create_directory(dir);
    fp = fopen(file_name, mode);
    if(fp == NULL)
      return -1;
  }
  fclose(fp);

  return 0;
}

static void TwoDMFree(int iImageY, void **pMem)
{
  int i;
  for(i=0; i<iImageY; i++)
  {
    if(pMem[i]!=NULL)
      free(pMem[i]);
  }
  free(pMem);
}