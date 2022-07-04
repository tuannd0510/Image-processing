#include <opencv2/highgui.hpp>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "NumCpp/Functions/where.hpp"
#include <NumCpp.hpp>
#include <fstream>

using namespace std;

bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i < j );
}

//Compare the contour area (USB_Port_Lean is used for contour sorting)
bool Contour_Area(vector<cv::Point> contour1, vector<cv::Point> contour2)
{
  return cv::contourArea(contour1)< cv::contourArea(contour2);
}

int main(){
    cv::Mat input, intermediate, iblur, edge, output;
    string pathsrc = "test.jpg";
    input = cv::imread(pathsrc, cv::IMREAD_COLOR);
    if(! input.data){
        cout<<"Could not open file" << endl;
        return -1;
    } 
    
    cv::namedWindow("input image", cv::WINDOW_AUTOSIZE);
    cv::imshow("input image", input);
    cv::waitKey(0);

    resize(input, intermediate, cv::Size(620,480)); //resize 620,480
    cvtColor(intermediate, intermediate, cv::COLOR_BGR2GRAY); // gray
    bilateralFilter(intermediate, iblur, 15, 80, 80, cv::BORDER_DEFAULT); // blur
    Canny(iblur, edge, 30,200); // perform edge detection

    /* find contour */
    threshold(edge, edge, 128, 255, cv::THRESH_BINARY);

    vector<vector<cv::Point> > contours;
    cv::Mat contourOutput = edge.clone();
    cv::findContours(contourOutput,contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0)); // tim cac contour trong anh
    
    /* sort contours */
    cv::sort(contours.begin(), contours.end(), compareContourAreas);

    /* grab contours */
    vector<cv::Point> smallestContour = contours[contours.size()-1]; 
    vector<cv::Point> biggestContour = contours[0]; 
    vector<cv::Point> screenCnt;
    cv::Mat approx, peri; 
    for (int i = 0; i < contours.size(); i++)
    {
        double peri = arcLength(contours[i],true);
        approxPolyDP(contours[i],approx, 0.018*peri,true);
        // cout << approx << endl;
        // cout << "ok" << endl;
        if (sizeof(approx) == 4){
            screenCnt = approx;
        }
        break;
    }

    // int detection =0;
    // if (screenCnt == )
    // {
    //     detection = 0;
    // }else
    // {
    //     detection = 1;
    // }
    
    // if (detection ==1 ){
    //     drawContours(input, screenCnt[],-1,(0,255,0),3);
    // }
    
    drawContours(input, screenCnt, -1, (0,255,0), 3);
    

    /* Masking the part other than the number plate */
    cv::Mat mask(480,620,CV_8UC3, cv::Scalar(0,0,0));
    drawContours(mask, screenCnt, 0,255,-1);
    bitwise_and(input, input, mask = mask);

    ofstream myfile;
    myfile.open ("example.txt");
    myfile << mask;
    myfile.close();

    /* Now crop */
    cv::Mat xy(480,620,CV_8UC3, cv::Scalar(0,0,0));
    // Mat id = {255,255,255};
    // nc::where(mask == id)[0];
    cv::Mat a(480,620, CV_8UC3);
    
    // _OutputArray();
    // findNonZero(mask==255,a);

    // for (int i = 0; i < 2; i++)
    // {
    //     for (int j = 0; j < 3; j++)
    //     {
    //         cout << ad[i][j];
    //     }
    // }
        
    /* Read the number plate */

    // imshow("intermediate",contours);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}