#include <iostream>
#include <string>
#include<stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include "ueye.h"
 
using namespace std;
using namespace cv;

void initializeCameraInterface(HIDS* hCam_internal){
    // Inicijaliziranje kamere
    INT nRet = is_InitCamera (hCam_internal, NULL);
    if (nRet == IS_SUCCESS){
        cout << "Camera initialized!" << endl;
    }
    //SENSORINFO m_sInfo;			// sensor information struct
        //is_GetSensorInfo(*hCam_internal, &m_sInfo);
 
    // Postavljanje ColorMode-a kamere
    INT colorMode = IS_CM_BGR8_PACKED;
    nRet = is_SetColorMode(*hCam_internal,colorMode);
 
    if (nRet == IS_SUCCESS){
        cout << "Camera color mode succesfully set!" << endl;
    }
 
    //Postavljanje DisplayModea - alociranje memorije za dohvacanje slike
    INT displayMode = IS_SET_DM_DIB;
    nRet = is_SetDisplayMode (*hCam_internal, displayMode);
 
}
 
Mat getFrame(HIDS* hCam, int width, int height, Mat& mat) {
    // alociranje memorije
    char* pMem = NULL;
    int memID = 0;
    is_AllocImageMem(*hCam, width, height, 24, &pMem, &memID);
 

   
 // Aktiviranje memorije za sliku i dohvacanje slike
    is_SetImageMem(*hCam, pMem, memID);
    is_FreezeVideo(*hCam, IS_WAIT);
 
    // Prebacivanje slike u mat format
    VOID* pMem_b;
    int retInt = is_GetImageMem(*hCam, &pMem_b);
    if (retInt != IS_SUCCESS) {
        cout << "Image data could not be read from memory!" << endl;
    }
    memcpy(mat.ptr(), pMem_b, mat.cols*mat.rows*3); // dodano *3 za BGR, inace grayscale
    is_FreeImageMem(*hCam, pMem, memID);
 
        return mat;
}
vector<Point> getContours(Mat image) {
 
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
 
    findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> conPoly(contours.size());
    vector<Rect> boundRect(contours.size());
 
    vector<Point> biggest;
    int maxArea=0;
    for (int i = 0; i < contours.size(); i++)
    {
        int area = contourArea(contours[i]);  //podesit area > xyz za primjenu (na drugoj traci barem 8000)
 
        if (area > 5000) {
            float peri = arcLength(contours[i], true);
            approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
 
            if (area > maxArea && conPoly[i].size()==4 ) {
                biggest = { conPoly[i][0],conPoly[i][1] ,conPoly[i][2] ,conPoly[i][3] };
                maxArea = area;
            }
        }
    }
    return biggest;
}
 
vector<Point> rearrange(vector<Point> points)
{
    vector<Point> newPoints;
    vector<int>  sumPoints, subPoints;
 
    for (int i = 0; i < 4; i++)
    {
        sumPoints.push_back(points[i].x + points[i].y);
        subPoints.push_back(points[i].x - points[i].y);
    }
 
    newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // 0
    newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); //1
    newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); //2
    newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); //3
 
    return newPoints;
}
 
Mat imgWarp;
Mat getWarp(Mat image, vector<Point> points, float w, float h )
{
    Point2f src[4] = { points[0],points[1],points[2],points[3] };
    Point2f dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };
 
    Mat matrix = getPerspectiveTransform(src, dst);
    warpPerspective(image, imgWarp, matrix, Point(w, h));
 
    return imgWarp;
}

int main()
{
    HIDS hCam = 0;
    initializeCameraInterface(&hCam);
    cuda::printCudaDeviceInfo(0);
    cout << "\n Press 'q' to decode saved images.\n"<< endl;
 
    //inicijalizacija
    int height = 1080; //sirina i visina dohvacene slike
    int width = 1920;
    int w = 600; //sirina i visina naljepnice
    
    int h = 600;
    int b = 1;   //inicijalizacija za spremanje slika (ime=b.tif)
    int noimg = 50; //broj spremljenih slika naljepnice
    Mat img (height, width, CV_8UC3); // image(height, width, type)
    Mat imgprocessedcpu,test;
    Mat imgWarped(w, h, CV_8UC3, Scalar(0, 0, 0));
    cuda::GpuMat warmupgpu, imggpu,imggray, imgblur,imgthr, imgcanny,imgprocessed;
 
    //za procesiranje slike
    cv::Ptr<cv::cuda::CannyEdgeDetector> C = cv::cuda::createCannyEdgeDetector(120,170,3,false); //podesiti vrijednosti s obzirom na osvijetljenje, gledati imgprocessedcpu
    cv::Mat element = cv::getStructuringElement (cv::MORPH_RECT,cv::Size(3,3));
    cv::Ptr<cv::cuda::Filter> dilate;
    

	dilate = cv::cuda::createMorphologyFilter (cv::MORPH_DILATE,CV_8UC1,element);
    vector<Point> initialPoints, arrangedPoints;
 
 
    while(true){
        //dohvacanje slike
        auto start = getTickCount();
        getFrame(&hCam, img.cols, img.rows, img);
 
        //upload na gpu
        imggpu.upload(img);
 
        //procesiranje slike
        cuda::cvtColor(imggpu, imggray, COLOR_BGR2GRAY);
        C->detect(imggray, imgcanny);
        dilate->apply(imgcanny,imgprocessed);
 
        //download na cpu
        imgprocessed.download(imgprocessedcpu);
 
        //findcontours
        initialPoints = getContours(imgprocessedcpu);
        if (!initialPoints.empty()){
            arrangedPoints = rearrange(initialPoints);
 
            //warping
            if (!arrangedPoints.empty()){
                imgWarped = getWarp(img, arrangedPoints, w, h);
                //save images
               
		 if (b <= noimg){
 
                    string ime = to_string(b)+ ".tif";
                    imwrite(ime, imgWarped);
                    b= b+1;
 
                }
            }
        }
 
        //FPS
        auto end = getTickCount();
        auto totalTime = (end - start) / getTickFrequency();
        auto fps = 1 / totalTime;
 
        //prikaz rezultata
        putText(img, "FPS: " + to_string(int(fps)), Point(20, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 2, 237), 3, false);
        imshow("Image", img);
        imshow ("Procesirana slika", imgprocessedcpu);
        imshow ("Naljepnica", imgWarped);
 
        //exit
        if(waitKey(2) == 'q'){
            break;
        }
    }
        is_ExitCamera(hCam);
 
        //qr code detection
        QRCodeDetector decoder = QRCodeDetector();
        std::vector<Point> points;
        int count = 0;
        int failcount = 0;
        int goodcount = 0;
        auto startqr = getTickCount();
 
        for (int a=1; a <= b-1; a++){
            string imea = to_string(a)+ ".tif";
            Mat qr = imread(imea,IMREAD_UNCHANGED);
            std::string predmet = decoder.detectAndDecode(qr, points);
                        if (!points.empty()) {
 
                            if (predmet.rfind("P",0)==0) {
                                count = count +1;
                                cout <<"Decoded data: " <<predmet<<" : " << count << endl;
                                goodcount=goodcount +1;
 
                            }
                        else {cout <<"Fail :" << count << endl;
                                    count = count +1;
                                    failcount = failcount +1;}
 
                        }
        }
 
        auto endqr = getTickCount();
        auto totalTimeqr = (endqr - startqr) / getTickFrequency();
        cout << "\nSuccessfuly decoded " << goodcount << " out of "<< b-1 <<" images.\n" << "Failed to decode " << failcount << " images. \n" << 100*goodcount/(b-1) << "% success rate" << endl;
 
        cout << "Total time for QR code decoding elapsed: " << totalTimeqr << "seconds \n" << endl;
 
  
        return 0;
}


