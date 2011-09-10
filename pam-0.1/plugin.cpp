/* -*- Mode: C++; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* ***** BEGIN LICENSE BLOCK *****
 * Version: NPL 1.1/GPL 2.0/LGPL 2.1
 *
 * The contents of this file are subject to the Netscape Public License
 * Version 1.1 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 * http://www.mozilla.org/NPL/
 *
 * Software distributed under the License is distributed on an "AS IS" basis,
 * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
 * for the specific language governing rights and limitations under the
 * License.
 *
 * The Original Code is mozilla.org code.
 *
 * The Initial Developer of the Original Code is 
 * Netscape Communications Corporation.
 * Portions created by the Initial Developer are Copyright (C) 1998
 * the Initial Developer. All Rights Reserved.
 *
 * Contributor(s):
 *
 * Alternatively, the contents of this file may be used under the terms of
 * either the GNU General Public License Version 2 or later (the "GPL"), or 
 * the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
 * in which case the provisions of the GPL or the LGPL are applicable instead
 * of those above. If you wish to allow use of your version of this file only
 * under the terms of either the GPL or the LGPL, and not to allow others to
 * use your version of this file under the terms of the NPL, indicate your
 * decision by deleting the provisions above and replace them with the notice
 * and other provisions required by the GPL or the LGPL. If you do not delete
 * the provisions above, a recipient may use your version of this file under
 * the terms of any one of the NPL, the GPL or the LGPL.
 *
 * ***** END LICENSE BLOCK ***** */

//////////////////////////////////////////////////
//
// CPlugin class implementation
//
#ifdef XP_WIN
#include <windows.h>
#include <windowsx.h>
#endif

#ifdef XP_MAC
#include <TextEdit.h>
#endif

#ifdef XP_UNIX
#include <string.h>
#endif

#include "plugin.h"
#include "npfunctions.h"
#include <stdlib.h>
#include <stdio.h>
#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <unistd.h>		// For _kbhit()
#include <direct.h>		// For mkdir()
#include <vector>
#include <string>
#include <sys/stat.h>
#include<sys/types.h>
#include <glib.h>
#include "gnome-keyring.h"

#define APPLICATION_NAME "gnome-keyring-query"
 #define MAX_PASSWORD_LENGTH 100


using namespace std;

double leastDistSq;
  double distSq;
static NPIdentifier sFoo_id,sFoo_id1;
// Haar Cascade file, used for Face Detection.
const char *faceCascadeFilename = "/home/home/haarcascade_frontalface_alt.xml";
const char *name1;
int SAVE_EIGENFACE_IMAGES = 1;		// Set to 0 if you dont want images of the Eigenvectors saved to files (for debugging).
//#define USE_MAHALANOBIS_DISTANCE	// You might get better recognition accuracy if you enable this.
// Global variables
IplImage ** faceImgArr        = 0; // array of face images
CvMat    *  personNumTruthMat = 0; // array of person numbers
//#define	MAX_NAME_LENGTH 256		// Give each name a fixed size for easier code.
//char **personNames = 0;			// array of person names (indexed by the person number). Added by Shervin.
vector<string> personNames;			// array of person names (indexed by the person number). Added by Shervin.
int faceWidth = 120;	// Default dimensions for faces in the face recognition database. Added by Shervin.
int faceHeight = 90;	//	"		"		"		"		"		"		"		"
int nPersons                  = 0; // the number of people in the training set. Added by Shervin.
int nTrainFaces               = 0; // the number of training images
int nEigens                   = 0; // the number of eigenvalues
IplImage * pAvgTrainImg       = 0; // the average image
IplImage ** eigenVectArr      = 0; // eigenvectorsCan't open training database file 'facedata.xml'.
CvMat * eigenValMat           = 0; // eigenvalues
CvMat * projectedTrainFaceMat = 0; // projected training faces
CvCapture* camera = 0;	// The camera device.

int set_password(const char * name, const char * password)
 {
     GnomeKeyringAttributeList * attributes;
     GnomeKeyringResult result;
     guint item_id;
     
     attributes = g_array_new(FALSE, FALSE, sizeof (GnomeKeyringAttribute));
     gnome_keyring_attribute_list_append_string(attributes,
             "name",
             name);
     gnome_keyring_attribute_list_append_string(attributes,
             "magic",
             APPLICATION_NAME);
     
     result = gnome_keyring_item_create_sync(NULL,
             GNOME_KEYRING_ITEM_GENERIC_SECRET,
             name,
             attributes,
             password,
             TRUE,
             &item_id);
     gnome_keyring_attribute_list_free(attributes);
     
     return (result == GNOME_KEYRING_RESULT_OK);
 }

char * get_password(const char * name)
 {
     GnomeKeyringAttributeList * attributes;
     GnomeKeyringResult result;
     GList *found_list;
     GList *i;
     GnomeKeyringFound * found;
     char * password;
     
     attributes = g_array_new(FALSE, FALSE, sizeof (GnomeKeyringAttribute));
     gnome_keyring_attribute_list_append_string(attributes,
             "name",
             name);
     gnome_keyring_attribute_list_append_string(attributes,
             "magic",
             APPLICATION_NAME);
     
     result = gnome_keyring_find_items_sync(GNOME_KEYRING_ITEM_GENERIC_SECRET,
             attributes,
             &found_list);
     gnome_keyring_attribute_list_free(attributes);
     
     if (result != GNOME_KEYRING_RESULT_OK)
         return NULL;
     
     for (i = found_list; i != NULL; i = i->next)
     {
         found=static_cast<GnomeKeyringFound*>(i->data);
         password=g_strdup(found->secret);
         break;
     }
     gnome_keyring_found_list_free(found_list);
     
     return password;
 }

static char* toCString(const NPString& string)
{
      char* result = static_cast<char*>(malloc(string.UTF8Length + 1));
      memcpy(result, string.UTF8Characters, string.UTF8Length);
      result[string.UTF8Length] = '\0';
      return result;

}

void millisleep(int milliseconds)
{
      usleep(milliseconds * 1000);
}

IplImage* convertFloatImageToUcharImage(const IplImage *srcImg)
{
	IplImage *dstImg = 0;
	if ((srcImg) && (srcImg->width > 0 && srcImg->height > 0)) {

		// Spread the 32bit floating point pixels to fit within 8bit pixel range.
		double minVal, maxVal;
		cvMinMaxLoc(srcImg, &minVal, &maxVal);
  // Deal with NaN and extreme values, since the DFT seems to give some NaN results.
		if (cvIsNaN(minVal) || minVal < -1e30)
			minVal = -1e30;
		if (cvIsNaN(maxVal) || maxVal > 1e30)
			maxVal = 1e30;
		if (maxVal-minVal == 0.0f)
			maxVal = minVal + 0.001;	// remove potential divide by zero errors.
  // Convert the format
		dstImg = cvCreateImage(cvSize(srcImg->width, srcImg->height), 8, 1);
		cvConvertScale(srcImg, dstImg, 255.0 / (maxVal - minVal), - minVal * 255.0 / (maxVal-minVal));
	}
	return dstImg;
}

void storeTrainingData()
{
	CvFileStorage * fileStorage;
	int i;

	// create a file-storage interface
	fileStorage = cvOpenFileStorage( "/home/home/facedata.xml", 0, CV_STORAGE_WRITE );

	// Store the person names. Added by Shervin.
	cvWriteInt( fileStorage, "nPersons", nPersons );
	for (i=0; i<nPersons; i++) {
		char varname[200];
		sprintf( varname, "personName_%d", (i+1) );
		cvWriteString(fileStorage, varname, personNames[i].c_str(), 0);
	}

	// store all the data
	cvWriteInt( fileStorage, "nEigens", nEigens );
	cvWriteInt( fileStorage, "nTrainFaces", nTrainFaces );
	cvWrite(fileStorage, "trainPersonNumMat", personNumTruthMat, cvAttrList(0,0));
	cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0,0));
	cvWrite(fileStorage, "projectedTrainFaceMat", projectedTrainFaceMat, cvAttrList(0,0));
	cvWrite(fileStorage, "avgTrainImg", pAvgTrainImg, cvAttrList(0,0));
	for(i=0; i<nEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0,0));
	}

	// release the file-storage interface
	cvReleaseFileStorage( &fileStorage );
}

void storeEigenfaceImages()
{
	// Store the average image to a file
	cvSaveImage("/home/home/out_averageImage.bmp", pAvgTrainImg);
	// Create a large image made of many eigenface images.
	// Must also convert each eigenface image to a normal 8-bit UCHAR image instead of a 32-bit float image.
	printf("Saving the %d eigenvector images as 'out_eigenfaces.bmp'\n", nEigens);
	if (nEigens > 0) {
		// Put all the eigenfaces next to each other.
		int COLUMNS = 8;	// Put upto 8 images on a row.
		int nCols = min(nEigens, COLUMNS);
		int nRows = 1 + (nEigens / COLUMNS);	// Put the rest on new rows.
		int w = eigenVectArr[0]->width;
		int h = eigenVectArr[0]->height;
		CvSize size;
		size = cvSize(nCols * w, nRows * h);
		IplImage *bigImg = cvCreateImage(size, IPL_DEPTH_8U, 1);	// 8-bit Greyscale UCHAR image
		for (int i=0; i<nEigens; i++) {
			// Get the eigenface image.
			IplImage *byteImg = convertFloatImageToUcharImage(eigenVectArr[i]);
			// Paste it into the correct position.
			int x = w * (i % COLUMNS);
			int y = h * (i / COLUMNS);
			CvRect ROI = cvRect(x, y, w, h);
			cvSetImageROI(bigImg, ROI);
			cvCopyImage(byteImg, bigImg);
			cvResetImageROI(bigImg);
			cvReleaseImage(&byteImg);
		}
		cvSaveImage("/home/home/out_eigenfaces.bmp", bigImg);
		cvReleaseImage(&bigImg);
	}
}

void doPCA()
{
	int i;
	CvTermCriteria calcLimit;
	CvSize faceImgSize;

	// set the number of eigenvalues to use
	nEigens = nTrainFaces-1;

	// allocate the eigenvector images
	faceImgSize.width  = faceImgArr[0]->width;
	faceImgSize.height = faceImgArr[0]->height;
	eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);
	for(i=0; i<nEigens; i++)
		eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	// allocate the eigenvalue array
	eigenValMat = cvCreateMat( 1, nEigens, CV_32FC1 );

	// allocate the averaged image
	pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	// set the PCA termination criterion
	calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, nEigens, 1);

	// compute average image, eigenvalues, and eigenvectors
	cvCalcEigenObjects(
		nTrainFaces,
		(void*)faceImgArr,
		(void*)eigenVectArr,
		CV_EIGOBJ_NO_CALLBACK,
		0,
		0,
		&calcLimit,
		pAvgTrainImg,
		eigenValMat->data.fl);

	cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);
}

// Read the names & image filenames of people from a text file, and load all those images listed.
int loadFaceImgArray(char * filename)
{
	FILE * imgListFile = 0;
	char imgFilename[512];
	int iFace, nFaces=0;
	int i;

	// open the input file
	if( !(imgListFile = fopen(filename, "r")) )
	{
	  return 0;
	}

	// count the number of faces
	while( fgets(imgFilename, 512, imgListFile) ) ++nFaces;
	rewind(imgListFile);

	// allocate the face-image array and person number matrix
	faceImgArr        = (IplImage **)cvAlloc( nFaces*sizeof(IplImage *) );
	personNumTruthMat = cvCreateMat( 1, nFaces, CV_32SC1 );

	personNames.clear();	// Make sure it starts as empty.
	nPersons = 0;

	// store the face images in an array
	for(iFace=0; iFace<nFaces; iFace++)
	{
		char personName[256];
		string sPersonName;
		int personNumber;

		// read person number (beginning with 1), their name and the image filename.
		fscanf(imgListFile, "%d %s %s", &personNumber, personName, imgFilename);
		sPersonName = personName;

		// Check if a new person is being loaded.
		if (personNumber > nPersons) {
			// Allocate memory for the extra person (or possibly multiple), using this new person's name.
			for (i=nPersons; i < personNumber; i++) {
				personNames.push_back( sPersonName );
			}
			nPersons = personNumber;
		
		}

		// Keep the data
		personNumTruthMat->data.i[iFace] = personNumber;

		// load the face image
		faceImgArr[iFace] = cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);

		if( !faceImgArr[iFace] )
		{
			
			return 0;
		}
	}

	fclose(imgListFile);

	return nFaces;
}


void learn(char *szFileTrain)
{
	int i, offset;

	// load training data
	nTrainFaces = loadFaceImgArray(szFileTrain);
	if( nTrainFaces < 2 )
	{
		return;
	}

	// do PCA on the training faces
	doPCA();

	// project the training images onto the PCA subspace
	projectedTrainFaceMat = cvCreateMat( nTrainFaces, nEigens, CV_32FC1 );
	offset = projectedTrainFaceMat->step / sizeof(float);
	for(i=0; i<nTrainFaces; i++)
	{
		//int offset = i * nEigens;
		cvEigenDecomposite(
			faceImgArr[i],
			nEigens,
			eigenVectArr,
			0, 0,
			pAvgTrainImg,
			//projectedTrainFaceMat->data.fl + i*nEigens);
			projectedTrainFaceMat->data.fl + i*offset);
	}

	// store the recognition data as an xml file
	storeTrainingData();

	// Save all the eigenvectors as images, so that they can be checked.
	if (SAVE_EIGENFACE_IMAGES) {
		storeEigenfaceImages();
	}

}

int findNearestNeighbor(float * projectedTestFace, float *pConfidence)
{
leastDistSq = 1e12;
 
	int i, iTrain, iNearest = 0;

	for(iTrain=0; iTrain<nTrainFaces; iTrain++)
	{
		 distSq=0;

		for(i=0; i<nEigens; i++)
		{
			float d_i = projectedTestFace[i] - projectedTrainFaceMat->data.fl[iTrain*nEigens + i];

			distSq += d_i*d_i;  // Eucledian Distance

		}

		if(distSq < leastDistSq)
		{
			leastDistSq = distSq;
			iNearest = iTrain;
		}

}

	// Return the confidence level based on the Euclidean distance,
	// so that similar images should give a confidence between 0.5 to 1.0,
	// and very different images should give a confidence between 0.0 to 0.5.
 
	*pConfidence = 1.0f - sqrt( leastDistSq / (float)(nTrainFaces * nEigens) ) / 255.0f;


	// Return the found index.
	return iNearest;
}

IplImage* cropImage(const IplImage *img, const CvRect region)
{
	IplImage *imageTmp;
	IplImage *imageRGB;
	CvSize size;
	size.height = img->height;
	size.width = img->width;

	if (img->depth != IPL_DEPTH_8U) {
		
		return 0;
	}

	// First create a new (color or greyscale) IPL Image and copy contents of img into it.
	imageTmp = cvCreateImage(size, IPL_DEPTH_8U, img->nChannels);
	cvCopy(img, imageTmp, NULL);

	// Create a new image of the detected region
	// Set region of interest to that surrounding the face
	cvSetImageROI(imageTmp, region);
	// Copy region of interest (i.e. face) into a new iplImage (imageRGB) and return it
	size.width = region.width;
	size.height = region.height;
	imageRGB = cvCreateImage(size, IPL_DEPTH_8U, img->nChannels);
	cvCopy(imageTmp, imageRGB, NULL);	// Copy just the region.

    cvReleaseImage( &imageTmp );
	return imageRGB;		
}

IplImage* resizeImage(const IplImage *origImg, int newWidth, int newHeight)
{
	IplImage *outImg = 0;
	int origWidth;
	int origHeight;
	if (origImg) {
		origWidth = origImg->width;
		origHeight = origImg->height;
	}
	if (newWidth <= 0 || newHeight <= 0 || origImg == 0 || origWidth <= 0 || origHeight <= 0) {

		return 0;
	}

	// Scale the image to the new dimensions, even if the aspect ratio will be changed.
	outImg = cvCreateImage(cvSize(newWidth, newHeight), origImg->depth, origImg->nChannels);
	if (newWidth > origImg->width && newHeight > origImg->height) {
		// Make the image larger
		cvResetImageROI((IplImage*)origImg);
		cvResize(origImg, outImg, CV_INTER_LINEAR);	// CV_INTER_CUBIC or CV_INTER_LINEAR is good for enlarging
	}
	else {
		// Make the image smaller
		cvResetImageROI((IplImage*)origImg);
		cvResize(origImg, outImg, CV_INTER_AREA);	// CV_INTER_AREA is good for shrinking / decimation, but bad at enlarging.
	}

	return outImg;
}
IplImage* convertImageToGreyscale(const IplImage *imageSrc)
{
	IplImage *imageGrey;
	// Either convert the image to greyscale, or make a copy of the existing greyscale image.
	// This is to make sure that the user can always call cvReleaseImage() on the output, whether it was greyscale or not.
	if (imageSrc->nChannels == 3) {
		imageGrey = cvCreateImage( cvGetSize(imageSrc), IPL_DEPTH_8U, 1 );
		cvCvtColor( imageSrc, imageGrey, CV_BGR2GRAY );
	}
	else {
		imageGrey = cvCloneImage(imageSrc);
	}
	return imageGrey;
}

int loadTrainingData(CvMat ** pTrainPersonNumMat)
{
	CvFileStorage * fileStorage;
	int i;

	// create a file-storage interface
	fileStorage = cvOpenFileStorage( "/home/home/facedata.xml", 0, CV_STORAGE_READ );
	if( !fileStorage ) 
  {
		return 0;
  }

	// Load the person names.
	personNames.clear();	// Make sure it starts as empty.
	nPersons = cvReadIntByName( fileStorage, 0, "nPersons", 0 );
	if (nPersons == 0) 
  {
		return 0;
	}
	// Load each person's name.
	for (i=0; i<nPersons; i++) 
  {
		string sPersonName;
		char varname[200];
		sprintf( varname, "personName_%d", (i+1) );
		sPersonName = cvReadStringByName(fileStorage, 0, varname );
		personNames.push_back( sPersonName );
	}

	// Load the data
	nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
	nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
	*pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
	eigenValMat  = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
	projectedTrainFaceMat = (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
	pAvgTrainImg = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
	eigenVectArr = (IplImage **)cvAlloc(nTrainFaces*sizeof(IplImage *));
	for(i=0; i<nEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
	}

	// release the file-storage interface
	cvReleaseFileStorage( &fileStorage );
  return 1;
}

IplImage* getCameraFrame(void)
{
	IplImage *frame;

	// If the camera hasn't been initialized, then open it.
	if (!camera) 
  {
		camera = cvCaptureFromCAM( 0 );
		if (!camera) 
    {
			return 0;
		}
		// Try to set the camera resolution
		cvSetCaptureProperty( camera, CV_CAP_PROP_FRAME_WIDTH, 320 );
		cvSetCaptureProperty( camera, CV_CAP_PROP_FRAME_HEIGHT, 240 );
		// Wait a little, so that the camera can auto-adjust itself
		millisleep(1000);	// (in milliseconds)
		frame = cvQueryFrame( camera );	// get the first frame, to make sure the camera is initialized.
	}

	frame = cvQueryFrame( camera );
	if (!frame) 
  {
		return 0;
	}
	return frame;
}

CvRect detectFaceInImage(const IplImage *inputImg, const CvHaarClassifierCascade* cascade )
{
	const CvSize minFeatureSize = cvSize(20, 20);
	const int flags = CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH;	// Only search for 1 face.
	const float search_scale_factor = 1.1f;
	IplImage *detectImg;
	IplImage *greyImg = 0;
	CvMemStorage* storage;
	CvRect rc;
	double t;
	CvSeq* rects;
	int i;

	storage = cvCreateMemStorage(0);
	cvClearMemStorage( storage );

	// If the image is color, use a greyscale copy of the image.
	detectImg = (IplImage*)inputImg;	// Assume the input image is to be used.
	if (inputImg->nChannels > 1) 
	{
		greyImg = cvCreateImage(cvSize(inputImg->width, inputImg->height), IPL_DEPTH_8U, 1 );
		cvCvtColor( inputImg, greyImg, CV_BGR2GRAY );
		detectImg = greyImg;	// Use the greyscale version as the input.
	}

	// Detect all the faces.
	t = (double)cvGetTickCount();
	rects = cvHaarDetectObjects( detectImg, (CvHaarClassifierCascade*)cascade, storage,
				search_scale_factor, 3, flags, minFeatureSize );
	t = (double)cvGetTickCount() - t;
	// Get the first detected face (the biggest).
	if (rects->total > 0) {
        rc = *(CvRect*)cvGetSeqElem( rects, 0 );
    }
	else
		rc = cvRect(-1,-1,-1,-1);	// Couldn't find the face.

	//cvReleaseHaarClassifierCascade( &cascade );
	//cvReleaseImage( &detectImg );
	if (greyImg)
		cvReleaseImage( &greyImg );
	cvReleaseMemStorage( &storage );

	return rc;	// Return the biggest face found, or (-1,-1,-1,-1).
}
	

CvMat* retrainOnline(void)
{
	CvMat *trainPersonNumMat;
	int i;

	// Free & Re-initialize the global variables.
	if (faceImgArr) {
		for (i=0; i<nTrainFaces; i++) {
			if (faceImgArr[i])
				cvReleaseImage( &faceImgArr[i] );
		}
	}
	cvFree( &faceImgArr ); // array of face images
	cvFree( &personNumTruthMat ); // array of person numbers
	personNames.clear();			// array of person names (indexed by the person number). Added by Shervin.
	nPersons = 0; // the number of people in the training set. Added by Shervin.
	nTrainFaces = 0; // the number of training images
	nEigens = 0; // the number of eigenvalues
	cvReleaseImage( &pAvgTrainImg ); // the average image
	for (i=0; i<nTrainFaces; i++) {
		if (eigenVectArr[i])
			cvReleaseImage( &eigenVectArr[i] );
	}
	cvFree( &eigenVectArr ); // eigenvectors
	cvFree( &eigenValMat ); // eigenvalues
	cvFree( &projectedTrainFaceMat ); // projected training faces

	// Retrain from the data in the files
	learn("/home/home/train.txt");

	// Load the previously saved training data
	if( !loadTrainingData( &trainPersonNumMat ) ) 
  {
		exit(1);
	}

	return trainPersonNumMat;
}

int recognizeFromCam1()
{
	int i;
	CvMat * trainPersonNumMat;  // the person numbers during training
	float * projectedTestFace;
	double timeFaceRecognizeStart;
	double tallyFaceRecognizeTime;
	CvHaarClassifierCascade* faceCascade;
	char cstr[256];
	bool saveNextFaces = false;
	char newPersonName[256];
	int newPersonFaces;
FILE *trainFile;
int id=8;
int nearest;

	trainPersonNumMat = 0;  // the person numbers during training
	projectedTestFace = 0;
	saveNextFaces = false;
	newPersonFaces = 0;

	// Load the previously saved training data
	if( loadTrainingData( &trainPersonNumMat ) ) {
		faceWidth = pAvgTrainImg->width;
		faceHeight = pAvgTrainImg->height;
	}
	
  // Project the test images onto the PCA subspace
	projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );

	// Create a GUI window for the user to see the camera image.
	cvNamedWindow("Input", CV_WINDOW_AUTOSIZE);

	// Make sure there is a "data" folder, for storing the new person.
	mkdir("home/home/data",S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

	// Load the HaarCascade classifier for face detection.
	faceCascade = (CvHaarClassifierCascade*)cvLoad(faceCascadeFilename, 0, 0, 0 );
	if( !faceCascade ) 
  {
		exit(1);
	}

	timeFaceRecognizeStart = (double)cvGetTickCount();	// Record the timing.

	while (id!=0)
	{
		int iNearest,truth;
		IplImage *camImg;
		IplImage *greyImg;
		IplImage *faceImg;
		IplImage *sizedImg;
		IplImage *equalizedImg;
		IplImage *processedFaceImg;
		CvRect faceRect;
		IplImage *shownImg;
int keyPressed=0;
		float confidence;

				strcpy(newPersonName,name1);
				
				saveNextFaces = true;
			
			
  millisleep(1000);
		// Get the camera frame
		camImg = getCameraFrame();
		if (!camImg) 
    {
			exit(1);
		}
		// Make sure the image is greyscale, since the Eigenfaces is only done on greyscale image.
		greyImg = convertImageToGreyscale(camImg);

		// Perform face detection on the input image, using the given Haar cascade classifier.
		faceRect = detectFaceInImage(greyImg, faceCascade );
		// Make sure a valid face was detected.
		if (faceRect.width > 0) 
    {
			faceImg = cropImage(greyImg, faceRect);	// Get the detected face image.
			// Make sure the image is the same dimensions as the training images.
			sizedImg = resizeImage(faceImg, faceWidth, faceHeight);
			// Give the image a standard brightness and contrast, in case it was too dark or low contrast.
			equalizedImg = cvCreateImage(cvGetSize(sizedImg), 8, 1);	// Create an empty greyscale image
			cvEqualizeHist(sizedImg, equalizedImg);
			processedFaceImg = equalizedImg;
			if (!processedFaceImg) 
      {
				exit(1);
			}

			// If the face rec database has been loaded, then try to recognize the person currently detected.
			if (nEigens > 0) 
      {
				// project the test image onto the PCA subspace
				cvEigenDecomposite(
					processedFaceImg,
					nEigens,
					eigenVectArr,
					0, 0,
					pAvgTrainImg,
					projectedTestFace);

				// Check which person it is most likely to be.
				iNearest = findNearestNeighbor(projectedTestFace, &confidence);
				nearest  = trainPersonNumMat->data.i[iNearest];

				printf("Most likely person in camera: '%s' (confidence=%f.\n", personNames[nearest-1].c_str(), confidence);

			}//endif nEigens

			// Possibly save the processed face to the training set.
			if (saveNextFaces) 
      {
// MAYBE GET IT TO ONLY TRAIN SOME IMAGES ?
				// Use a different filename each time.
				sprintf(cstr, "data/%d_%s%d.pgm", nPersons+1, newPersonName, newPersonFaces+1);
				printf("Storing the current face of '%s' into image '%s'.\n", newPersonName, cstr);
				cvSaveImage(cstr, processedFaceImg, NULL);
				newPersonFaces++;
			}

			// Free the resources used for this frame.
			cvReleaseImage( &greyImg );
			cvReleaseImage( &faceImg );
			cvReleaseImage( &sizedImg );
			cvReleaseImage( &equalizedImg );
		}
  else
    return -1;
		// Show the data on the screen.
		shownImg = cvCloneImage(camImg);
		if (faceRect.width > 0) 
    {	// Check if a face was detected.
			// Show the detected face region.
			cvRectangle(shownImg, cvPoint(faceRect.x, faceRect.y), cvPoint(faceRect.x + faceRect.width-1, faceRect.y + faceRect.height-1), CV_RGB(0,255,0), 1, 8, 0);
			if (nEigens > 0) 
      {	// Check if the face recognition database is loaded and a person was recognized.
				// Show the name of the recognized person, overlayed on the image below their face.
				CvFont font;
				cvInitFont(&font,CV_FONT_HERSHEY_PLAIN, 1.0, 1.0, 0,1,CV_AA);
				CvScalar textColor = CV_RGB(0,255,255);	// light blue text
				char text[256];
			
				cvPutText(shownImg, text, cvPoint(faceRect.x, faceRect.y + faceRect.height + 15), &font, textColor);
				
			}
		}

		// Display the image.
		cvShowImage("Input", shownImg);

		// Give some time for OpenCV to draw the GUI and check if the user has pressed something in the GUI window.
		keyPressed = cvWaitKey(10);
		if (keyPressed == 27) 
    {	// Check if the user hit the 'Escape' key in the GUI window.
			break;	// Stop processing input.
		}
    id=id-1;
		cvReleaseImage( &shownImg );
	}

// stop saving next faces.
				// Store the saved data into the training file.
			
				// Append the new person to the end of the training data.
				trainFile = fopen("/home/home/train.txt", "a");
				for (i=0; i<newPersonFaces; i++) 
        {
					sprintf(cstr, "data/%d_%s%d.pgm", nPersons+1, newPersonName, i+1);
					fprintf(trainFile, "%d %s %s\n", nPersons+1, newPersonName, cstr);
				}
				fclose(trainFile);
	// Now there is one more person in the database, ready for retraining.
		
				//break;
			//case 'r':

				// Re-initialize the local data.
				projectedTestFace = 0;
				saveNextFaces = false;
				newPersonFaces = 0;

				// Retrain from the new database without shutting down.
				// Depending on the number of images in the training set and number of people, it might take 30 seconds or so.
				cvFree( &trainPersonNumMat );	// Free the previous data before getting new data
				trainPersonNumMat = retrainOnline();
				// Project the test images onto the PCA subspace
				cvFree(&projectedTestFace);	// Free the previous data before getting new data
				projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );
	
	// Free the camera and memory resources used.
	cvReleaseCapture( &camera );
	cvReleaseHaarClassifierCascade( &faceCascade );
	return nearest;

}

int recognizeFromCam()
{
int i;
	
	CvMat * trainPersonNumMat;  // the person numbers during training
	float * projectedTestFace;
	double timeFaceRecognizeStart;
	double tallyFaceRecognizeTime;
	CvHaarClassifierCascade* faceCascade;
	char cstr[256];
	bool saveNextFaces=false;
	char newPersonName[256];
	int newPersonFaces;

	trainPersonNumMat = 0;  // the person numbers during training
	projectedTestFace = 0;
	saveNextFaces=false;
	newPersonFaces = 0;

	printf("Recognizing person in the camera ...\n");

	// Load the previously saved training data
	if( loadTrainingData( &trainPersonNumMat ) ) {
		faceWidth = pAvgTrainImg->width;
		faceHeight = pAvgTrainImg->height;
	}
	else {
		//printf("ERROR in recognizeFromCam(): Couldn't load the training data!\n");
		//exit(1);
	}

	// Project the test images onto the PCA subspace
	projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );

	// Create a GUI window for the user to see the camera image.
	cvNamedWindow("Input", CV_WINDOW_AUTOSIZE);

	// Load the HaarCascade classifier for face detection.
	faceCascade = (CvHaarClassifierCascade*)cvLoad(faceCascadeFilename, 0, 0, 0 );
	if( !faceCascade ) {
		printf("ERROR in recognizeFromCam(): Could not load Haar cascade Face detection classifier in '%s'.\n", faceCascadeFilename);
		return 0;
	}
	
		int iNearest, nearest, truth;
		IplImage *camImg;
		IplImage *greyImg;
		IplImage *faceImg;
		IplImage *sizedImg;
		IplImage *equalizedImg;
		IplImage *processedFaceImg;
		CvRect faceRect;
		IplImage *shownImg;
		char keyPressed;
		FILE *trainFile;
		float confidence;
		

		camImg = getCameraFrame();
		if (!camImg) {
			printf("ERROR in recognizeFromCam(): Bad input image!\n");
			return 0;
		}
		// Make sure the image is greyscale, since the Eigenfaces is only done on greyscale image.
		greyImg = convertImageToGreyscale(camImg);

		// Perform face detection on the input image, using the given Haar cascade classifier.
		faceRect = detectFaceInImage(greyImg, faceCascade );
		
		// Make sure a valid face was detected.
		if (faceRect.width > 0) {
			faceImg = cropImage(greyImg, faceRect);	// Get the detected face image.
			if(faceImg==0)
			{
				return 0;
			}
			// Make sure the image is the same dimensions as the training images.
			sizedImg = resizeImage(faceImg, faceWidth, faceHeight);
			// Give the image a standard brightness and contrast, in case it was too dark or low contrast.
			equalizedImg = cvCreateImage(cvGetSize(sizedImg), 8, 1);	// Create an empty greyscale image
			cvEqualizeHist(sizedImg, equalizedImg);
			processedFaceImg = equalizedImg;
			if (!processedFaceImg) {
				printf("ERROR in recognizeFromCam(): Don't have input image!\n");
				return 0;;
			}

			// If the face rec database has been loaded, then try to recognize the person currently detected.
			if (nEigens > 0) {
				// project the test image onto the PCA subspace
				cvEigenDecomposite(
					processedFaceImg,
					nEigens,
					eigenVectArr,
					0, 0,
					pAvgTrainImg,
					projectedTestFace);

				// Check which person it is most likely to be.
				iNearest = findNearestNeighbor(projectedTestFace, &confidence);
				nearest  = trainPersonNumMat->data.i[iNearest];

				

			}//endif nEigens

			// Possibly save the processed face to the training set.
			if (saveNextFaces) {
// MAYBE GET IT TO ONLY TRAIN SOME IMAGES ?
				// Use a different filename each time.
				sprintf(cstr, "data/%d_%s%d.pgm", nPersons+1, newPersonName, newPersonFaces+1);
				cvSaveImage(cstr, processedFaceImg, NULL);
				newPersonFaces++;
			}

			// Free the resources used for this frame.
			cvReleaseImage( &greyImg );
			cvReleaseImage( &faceImg );
			cvReleaseImage( &sizedImg );
			cvReleaseImage( &equalizedImg );
		}
    

		// Show the data on the screen.
		shownImg = cvCloneImage(camImg);
		if (faceRect.width > 0) {	// Check if a face was detected.
			// Show the detected face region.
			cvRectangle(shownImg, cvPoint(faceRect.x, faceRect.y), cvPoint(faceRect.x + faceRect.width-1, faceRect.y + faceRect.height-1), CV_RGB(0,255,0), 1, 8, 0);
			if (nEigens > 0) {	// Check if the face recognition database is loaded and a person was recognized.
				// Show the name of the recognized person, overlayed on the image below their face.
				CvFont font;
				cvInitFont(&font,CV_FONT_HERSHEY_PLAIN, 1.0, 1.0, 0,1,CV_AA);
				CvScalar textColor = CV_RGB(0,255,255);	// light blue text
				char text[256];
				snprintf(text, sizeof(text)-1, "Name: '%s'", personNames[nearest-1].c_str());
				cvPutText(shownImg, text, cvPoint(faceRect.x, faceRect.y + faceRect.height + 15), &font, textColor);
			
      
			}
		}
		// Display the image.
		cvShowImage("Input", shownImg);
		keyPressed=cvWaitKey(2000);
		

		cvReleaseImage( &shownImg );

	// Free the camera and memory resources used.
	cvReleaseCapture( &camera );
	cvReleaseHaarClassifierCascade( &faceCascade );
  if(faceRect.width<=0)
    return -1;

	return nearest;
}



// Helper class that can be used to map calls to the NPObject hooks
// into virtual methods on instances of classes that derive from this
// class.

class ScriptablePluginObjectBase : public NPObject
{
public:
  ScriptablePluginObjectBase(NPP npp)
    : mNpp(npp)
  {
  }

  virtual ~ScriptablePluginObjectBase()
  {
  }

  // Virtual NPObject hooks called through this base class. Override
  // as you see fit.
  virtual void Invalidate();
  virtual bool HasMethod(NPIdentifier name);
  virtual bool Invoke(NPIdentifier name, const NPVariant *args,
                      uint32_t argCount, NPVariant *result);
  virtual bool InvokeDefault(const NPVariant *args, uint32_t argCount,
                             NPVariant *result);
  virtual bool HasProperty(NPIdentifier name);
  virtual bool GetProperty(NPIdentifier name, NPVariant *result);
  virtual bool SetProperty(NPIdentifier name, const NPVariant *value);
  virtual bool RemoveProperty(NPIdentifier name);
  virtual bool Enumerate(NPIdentifier **identifier, uint32_t *count);
  virtual bool Construct(const NPVariant *args, uint32_t argCount,
                         NPVariant *result);

public:
  static void _Deallocate(NPObject *npobj);
  static void _Invalidate(NPObject *npobj);
  static bool _HasMethod(NPObject *npobj, NPIdentifier name);
  static bool _Invoke(NPObject *npobj, NPIdentifier name,
                      const NPVariant *args, uint32_t argCount,
                      NPVariant *result);
  static bool _InvokeDefault(NPObject *npobj, const NPVariant *args,
                             uint32_t argCount, NPVariant *result);
  static bool _HasProperty(NPObject * npobj, NPIdentifier name);
  static bool _GetProperty(NPObject *npobj, NPIdentifier name,
                           NPVariant *result);
  static bool _SetProperty(NPObject *npobj, NPIdentifier name,
                           const NPVariant *value);
  static bool _RemoveProperty(NPObject *npobj, NPIdentifier name);
  static bool _Enumerate(NPObject *npobj, NPIdentifier **identifier,
                         uint32_t *count);
  static bool _Construct(NPObject *npobj, const NPVariant *args,
                         uint32_t argCount, NPVariant *result);

protected:
  NPP mNpp;
};

#define DECLARE_NPOBJECT_CLASS_WITH_BASE(_class, ctor)                        \
static NPClass s##_class##_NPClass = {                                        \
  NP_CLASS_STRUCT_VERSION_CTOR,                                               \
  ctor,                                                                       \
  ScriptablePluginObjectBase::_Deallocate,                                    \
  ScriptablePluginObjectBase::_Invalidate,                                    \
  ScriptablePluginObjectBase::_HasMethod,                                     \
  ScriptablePluginObjectBase::_Invoke,                                        \
  ScriptablePluginObjectBase::_InvokeDefault,                                 \
  ScriptablePluginObjectBase::_HasProperty,                                   \
  ScriptablePluginObjectBase::_GetProperty,                                   \
  ScriptablePluginObjectBase::_SetProperty,                                   \
  ScriptablePluginObjectBase::_RemoveProperty,                                \
  ScriptablePluginObjectBase::_Enumerate,                                     \
  ScriptablePluginObjectBase::_Construct                                      \
}

#define GET_NPOBJECT_CLASS(_class) &s##_class##_NPClass


void
ScriptablePluginObjectBase::Invalidate()
{
}

bool
ScriptablePluginObjectBase::HasMethod(NPIdentifier name)
{
  return false;
}

bool
ScriptablePluginObjectBase::Invoke(NPIdentifier name, const NPVariant *args,
                                   uint32_t argCount, NPVariant *result)
{
  return false;
}

bool
ScriptablePluginObjectBase::InvokeDefault(const NPVariant *args,
                                          uint32_t argCount, NPVariant *result)
{
  return false;
}

bool
ScriptablePluginObjectBase::HasProperty(NPIdentifier name)
{
  return false;
}

bool
ScriptablePluginObjectBase::GetProperty(NPIdentifier name, NPVariant *result)
{
  return false;
}

bool
ScriptablePluginObjectBase::SetProperty(NPIdentifier name,
                                        const NPVariant *value)
{
  
  return false;
}

bool
ScriptablePluginObjectBase::RemoveProperty(NPIdentifier name)
{
  return false;
}

bool
ScriptablePluginObjectBase::Enumerate(NPIdentifier **identifier,
                                      uint32_t *count)
{
  return false;
}

bool
ScriptablePluginObjectBase::Construct(const NPVariant *args, uint32_t argCount,
                                      NPVariant *result)
{
  return false;
}

// static
void
ScriptablePluginObjectBase::_Deallocate(NPObject *npobj)
{
  // Call the virtual destructor.
  delete (ScriptablePluginObjectBase *)npobj;
}

// static
void
ScriptablePluginObjectBase::_Invalidate(NPObject *npobj)
{
  ((ScriptablePluginObjectBase *)npobj)->Invalidate();
}

// static
bool
ScriptablePluginObjectBase::_HasMethod(NPObject *npobj, NPIdentifier name)
{
  return ((ScriptablePluginObjectBase *)npobj)->HasMethod(name);
}

// static
bool
ScriptablePluginObjectBase::_Invoke(NPObject *npobj, NPIdentifier name,
                                    const NPVariant *args, uint32_t argCount,
                                    NPVariant *result)
{
  return ((ScriptablePluginObjectBase *)npobj)->Invoke(name, args, argCount,
                                                       result);
}

// static
bool
ScriptablePluginObjectBase::_InvokeDefault(NPObject *npobj,
                                           const NPVariant *args,
                                           uint32_t argCount,
                                           NPVariant *result)
{
  return ((ScriptablePluginObjectBase *)npobj)->InvokeDefault(args, argCount,
                                                              result);
}

// static
bool
ScriptablePluginObjectBase::_HasProperty(NPObject * npobj, NPIdentifier name)
{
  return ((ScriptablePluginObjectBase *)npobj)->HasProperty(name);
}

// static
bool
ScriptablePluginObjectBase::_GetProperty(NPObject *npobj, NPIdentifier name,
                                         NPVariant *result)
{
  return ((ScriptablePluginObjectBase *)npobj)->GetProperty(name, result);
}

// static
bool
ScriptablePluginObjectBase::_SetProperty(NPObject *npobj, NPIdentifier name,
                                         const NPVariant *value)
{
  return ((ScriptablePluginObjectBase *)npobj)->SetProperty(name, value);
}

// static
bool
ScriptablePluginObjectBase::_RemoveProperty(NPObject *npobj, NPIdentifier name)
{
  return ((ScriptablePluginObjectBase *)npobj)->RemoveProperty(name);
}

// static
bool
ScriptablePluginObjectBase::_Enumerate(NPObject *npobj,
                                       NPIdentifier **identifier,
                                       uint32_t *count)
{
  return ((ScriptablePluginObjectBase *)npobj)->Enumerate(identifier, count);
}

// static
bool
ScriptablePluginObjectBase::_Construct(NPObject *npobj, const NPVariant *args,
                                       uint32_t argCount, NPVariant *result)
{
  return ((ScriptablePluginObjectBase *)npobj)->Construct(args, argCount,
                                                          result);
}


class ConstructablePluginObject : public ScriptablePluginObjectBase
{
public:
  ConstructablePluginObject(NPP npp)
    : ScriptablePluginObjectBase(npp)
  {
  }

  virtual bool Construct(const NPVariant *args, uint32_t argCount,
                         NPVariant *result);
};

static NPObject *
AllocateConstructablePluginObject(NPP npp, NPClass *aClass)
{
  return new ConstructablePluginObject(npp);
}

DECLARE_NPOBJECT_CLASS_WITH_BASE(ConstructablePluginObject,
                                 AllocateConstructablePluginObject);

bool
ConstructablePluginObject::Construct(const NPVariant *args, uint32_t argCount,
                                     NPVariant *result)
{
  printf("Creating new ConstructablePluginObject!\n");

  NPObject *myobj =
    NPN_CreateObject(mNpp, GET_NPOBJECT_CLASS(ConstructablePluginObject));
  if (!myobj)
    return false;

  OBJECT_TO_NPVARIANT(myobj, *result);

  return true;
}

class ScriptablePluginObject : public ScriptablePluginObjectBase
{
public:
  ScriptablePluginObject(NPP npp)
    : ScriptablePluginObjectBase(npp)
  {
  }

  virtual bool HasMethod(NPIdentifier name);
  virtual bool HasProperty(NPIdentifier name);
  virtual bool GetProperty(NPIdentifier name, NPVariant *result);
  virtual bool Invoke(NPIdentifier name, const NPVariant *args,
                      uint32_t argCount, NPVariant *result);
  virtual bool InvokeDefault(const NPVariant *args, uint32_t argCount,
                             NPVariant *result);
 
};

static NPObject *
AllocateScriptablePluginObject(NPP npp, NPClass *aClass)
{
  return new ScriptablePluginObject(npp);
}

DECLARE_NPOBJECT_CLASS_WITH_BASE(ScriptablePluginObject,
                                 AllocateScriptablePluginObject);

bool
ScriptablePluginObject::HasMethod(NPIdentifier name)
{
  
  return name == sFoo_id;
	
}

bool
ScriptablePluginObject::HasProperty(NPIdentifier name)
{
  return false;
}

bool
ScriptablePluginObject::GetProperty(NPIdentifier name, NPVariant *result)
{
 
return true;
}

bool
ScriptablePluginObject::Invoke(NPIdentifier name, const NPVariant *args,
                               uint32_t argCount, NPVariant *result)
{

if(name==sFoo_id) 
{

int i,j,flag2,k;
CvMat *trainPersonNumMat;
flag2=0;
trainPersonNumMat=0;
 
char *password;
NPString target=NPVARIANT_TO_STRING(args[0]);
name1=toCString(target);
NPString check=NPVARIANT_TO_STRING(args[1]);
const char *check1=toCString(check);
if(strcmp(check1,"search")==0)
{
  k=loadTrainingData( &trainPersonNumMat );
  for(j=0;j<nPersons;j++)
  {
    if(strcmp(personNames[j].c_str(),name1)==0)
    {
    flag2=1;
break;
    }
  }

  if(flag2==1)
  {
    flag2=0;
    char *npOutString = (char *)NPN_MemAlloc(strlen("yes") + 1);
    if (!npOutString)
      return false;
    strcpy(npOutString,"yes");
    STRINGZ_TO_NPVARIANT(npOutString, *result);
return true;
  }
  else
  {
    char *npOutString = (char *)NPN_MemAlloc(strlen("no") + 1);
    if (!npOutString)
      return false;
    strcpy(npOutString,"no");
    STRINGZ_TO_NPVARIANT(npOutString, *result);
return true;
}
}
else if(strcmp(check1,"pri")==0)
{
  i=recognizeFromCam();
  if(i==-1)
    {
    char *npOutString = (char *)NPN_MemAlloc(strlen("noface") + 1);
    if (!npOutString)
      return false;
    strcpy(npOutString,password);
    STRINGZ_TO_NPVARIANT(npOutString, *result);
    return true;
    }
if(strcmp(personNames[i-1].c_str(),name1)==0)
    {
g_set_application_name(APPLICATION_NAME);
     
    
     

    password = get_password(name1);
             if (!password)
             {
                 return false;
             }
              char *npOutString = (char *)NPN_MemAlloc(strlen(password) + 1);
    if (!npOutString)
      return false;
    strcpy(npOutString,password);
    STRINGZ_TO_NPVARIANT(npOutString, *result);
             
             g_free(password);
return true;

}
else{ 

char *npOutString = (char *)NPN_MemAlloc(strlen("default") + 1);
    if (!npOutString)
      return false;
    strcpy(npOutString,"default");
    STRINGZ_TO_NPVARIANT(npOutString, *result);
return true;
}


}
else if(strcmp(check1,"train")==0)
{

  i=recognizeFromCam1();
NPString check2=NPVARIANT_TO_STRING(args[2]);
const char *check3=toCString(check2);
i=set_password(name1,check3);
char *npOutString = (char *)NPN_MemAlloc(strlen("yes") + 1);
    if (!npOutString)
      return false;
    strcpy(npOutString,"yes");
    STRINGZ_TO_NPVARIANT(npOutString, *result);
return true;
return true;
}
}
  

	

	
	
    
    
return false;    
   

 
}

bool
ScriptablePluginObject::InvokeDefault(const NPVariant *args, uint32_t argCount,
                                      NPVariant *result)
{
  return false;
}

CPlugin::CPlugin(NPP pNPInstance) :
  m_pNPInstance(pNPInstance),
  m_pNPStream(NULL),
  m_bInitialized(false),
  m_pScriptableObject(NULL)
{

  
 
  sFoo_id = NPN_GetStringIdentifier("Auth");
  
  
  
  

}

CPlugin::~CPlugin()
{
  
}


NPBool CPlugin::init(NPWindow* pNPWindow)
{
  return false;
}



NPBool CPlugin::isInitialized()
{
  return m_bInitialized;
}







NPObject *
CPlugin::GetScriptableObject()
{
  if (!m_pScriptableObject) {
    m_pScriptableObject =
      NPN_CreateObject(m_pNPInstance,
                       GET_NPOBJECT_CLASS(ScriptablePluginObject));
  }

  if (m_pScriptableObject) {
    NPN_RetainObject(m_pScriptableObject);
  }

  return m_pScriptableObject;
}


