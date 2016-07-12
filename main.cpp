#include <iostream>
#include <algorithm>
#include <vector>
#include "Blob.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/*
These source codes are part of NCTUCS project done by NCTU 0016014, 0016007.
Original object detection source codes:
https://github.com/JaimeIvanCervantes/Tracking
*/

bool sortByFrameCount(const Blob &lhs, const Blob &rhs) { return lhs.frameCount > rhs.frameCount; }

int main(int argc, const char * argv[]){
    int frameNumber = 0, ID = 0;
    cv::VideoCapture cap;
    cap.open("http://210.241.67.167:80/abs2mjpg/mjpg?camera=82&dummy=.mjpg");//
    cv::Mat rawFrame,rawCopyFrame,foregroundFrame, foregroundFrameBuffer, roiFrame, roiFrameBuffer, hsvRoiFrame, roiFrameMask;
    cv::BackgroundSubtractorMOG2 mog;
    cv::vector<Blob> blobContainer;

    cv::vector<cv::vector<cv::Point> > contours;
    cv::vector<cv::Vec4i> hierarchy;

    cv::Point dir1, dir2, dir1end, dir2end, car1text, car2text;
    dir1.x=180; dir1.y=180;
    dir2.x=120; dir2.y=80;
    dir1end.x=1; dir1end.y=dir1.y;
    dir2end.x=319; dir2end.y=dir2.y;
    car1text.x=0; car1text.y=230;
    car2text.x=0; car2text.y=20;
    char s[64];
    //dir1end.x=dir1.x; dir1end.y=1;
    //dir2end.x=dir2.x; dir2end.y=239;

    if(cap.isOpened())
            std::cout << cap.get(CV_CAP_PROP_FRAME_WIDTH) << " " << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << std::endl;
    int car1=0, car2=0;
    while(1) {
        frameNumber++;
        //std::cout << frameNumber << std::endl;

        if(!cap.isOpened())
            return -1;
        else
            cap >> rawFrame;
        //Backgroud Substraction
        cv::Rect boundingRectangle;
        rawCopyFrame=rawFrame.clone();

        mog(rawFrame,foregroundFrame,-1);
        mog.set("nmixtures", 2);
        mog.set("detectShadows",0);

        cv::threshold(foregroundFrame,foregroundFrame,130,255,cv::THRESH_BINARY);
        cv::medianBlur(foregroundFrame,foregroundFrame,3);
        cv::erode(foregroundFrame,foregroundFrame,cv::Mat());
        cv::dilate(foregroundFrame,foregroundFrame,cv::Mat());

        foregroundFrameBuffer = foregroundFrame.clone();

        cv::findContours( foregroundFrame, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

        cv::Mat contoursFrame = cv::Mat::zeros( foregroundFrame.size(), CV_8UC3 );

        std::vector<int> contourTaken(contours.size(),0);
        //Remove old container
        for (std::vector<Blob>::iterator it=blobContainer.begin(); it!=blobContainer.end();) {
            if (frameNumber - it->lastFrameNumber > 100 ) {  //&& it->frames.size() < 15)
                it = blobContainer.erase(it);
            } else {
                ++it;
            }
        }

        for (unsigned int bli = 0; bli < blobContainer.size(); bli++) {
            // Clean contact contours
            blobContainer[bli].contactContours.clear();
            blobContainer[bli].collision = 0;

            // Loop contours
            for( unsigned int coi = 0; coi<contours.size(); coi++ ) {
                // Obtain ROI from bounding rectangle of contours
                boundingRectangle = cv::boundingRect(contours[coi]);

                // Get distance
                float distance = sqrt(pow((blobContainer[bli].lastRectangle.x + blobContainer[bli].lastRectangle.width/2.0)-(boundingRectangle.x + boundingRectangle.width/2.0),2.0)+
                pow((blobContainer[bli].lastRectangle.y + blobContainer[bli].lastRectangle.height/2.0)-(boundingRectangle.y + boundingRectangle.height/2.0),2.0));

                // Detect collisions
                if (distance < std::min(blobContainer[bli].lastRectangle.width,blobContainer[bli].lastRectangle.height) &&//(distance < fmaxf(boundingRectangle.width,boundingRectangle.height)*2.0 &&
                frameNumber - blobContainer[bli].lastFrameNumber == 1 &&
                std::max(boundingRectangle.width,boundingRectangle.height) > std::max(cap.get(CV_CAP_PROP_FRAME_WIDTH),cap.get(CV_CAP_PROP_FRAME_HEIGHT))/40)
                {
                    blobContainer[bli].contactContours.push_back(coi);
                }

            }

        }

        if (blobContainer.size() > 1)
            std::sort(blobContainer.begin(),blobContainer.end(), sortByFrameCount);

        for (unsigned int bli = 0; bli<blobContainer.size(); bli++) {
            unsigned int maxArea = 0;
            int selectedContourIndex = -1;
            for (unsigned int cni = 0; cni<blobContainer[bli].contactContours.size(); cni++) {
                int coi = blobContainer[bli].contactContours[cni];
                int contourArea = cv::boundingRect(contours[coi]).width*cv::boundingRect(contours[coi]).height;
                if (contourArea > maxArea) {
                    maxArea = contourArea;
                    selectedContourIndex = coi;
                }
            }

            if (selectedContourIndex != -1 && contourTaken[selectedContourIndex] == 0) {
                contourTaken[selectedContourIndex] = 1;
                blobContainer[bli].contactContours.push_back(selectedContourIndex);

                // Get contour properties
                boundingRectangle = cv::boundingRect(contours[selectedContourIndex]);
                roiFrameMask =  foregroundFrameBuffer(boundingRectangle).clone();
                roiFrame = foregroundFrameBuffer(boundingRectangle).clone();
                roiFrameBuffer = rawFrame(boundingRectangle).clone();
                roiFrameBuffer.copyTo(roiFrame, roiFrameMask);

                // Append objects
                blobContainer[bli].frameCount++;
                blobContainer[bli].lastFrameNumber = frameNumber;
                blobContainer[bli].lastRectangle = boundingRectangle;
                blobContainer[bli].frames.push_back(roiFrameBuffer);
                blobContainer[bli].avgWidth = .8*blobContainer[bli].avgWidth + .2*roiFrame.size().width;
                blobContainer[bli].avgHeight = .8*blobContainer[bli].avgHeight + .2*roiFrame.size().height;
                blobContainer[bli].maxWidth = std::max(blobContainer[bli].maxWidth,roiFrame.size().width);
                blobContainer[bli].maxHeight = std::max(blobContainer[bli].maxHeight,roiFrame.size().height);
            }

            if (blobContainer[bli].contactContours.size() > 1) {
                blobContainer[bli].collision = 1;
            }

        }

        for (unsigned int coi = 0; coi < contours.size(); coi++) {
            boundingRectangle = cv::boundingRect(contours[coi]);
            if (contourTaken[coi] == 0 &&
            std::max(boundingRectangle.width,boundingRectangle.height) > std::max(cap.get(CV_CAP_PROP_FRAME_WIDTH),cap.get(CV_CAP_PROP_FRAME_HEIGHT))/20) {
                // Get contour properties
                boundingRectangle = cv::boundingRect(contours[coi]);
                roiFrameMask =  foregroundFrameBuffer(boundingRectangle).clone();
                roiFrame = foregroundFrameBuffer(boundingRectangle).clone();
                roiFrameBuffer = rawFrame(boundingRectangle).clone();
                roiFrameBuffer.copyTo(roiFrame, roiFrameMask);

                // Create objects
                ID++;
                Blob newObject;
                newObject.ID = ID;
                newObject.frameCount = 1;
                newObject.lineFrameCount = 30;
                newObject.firstFrameNumber = frameNumber;
                newObject.lastFrameNumber = frameNumber;
                newObject.currentPosition.x = boundingRectangle.x + boundingRectangle.width/2.0;
                newObject.currentPosition.y = boundingRectangle.y + boundingRectangle.height/2.0;
                newObject.line.x = 0.0;
                newObject.line.y = 0.0;
                newObject.firstRectangle = boundingRectangle;
                newObject.lastRectangle = boundingRectangle;
                newObject.detected = 0;
                newObject.frames.push_back(roiFrameBuffer);
                newObject.avgWidth = roiFrame.size().width;
                newObject.avgHeight = roiFrame.size().height;
                newObject.maxWidth = roiFrame.size().width;
                newObject.maxHeight = roiFrame.size().height;
                blobContainer.push_back(newObject);
            }
        }
        ////////////////////////////////
        //Below is Main Detection Part// Also our main modification
        ////////////////////////////////
        for (unsigned int bli = 0; bli<blobContainer.size(); bli++) {
            if (blobContainer[bli].lastFrameNumber == frameNumber && blobContainer[bli].frameCount > 12) {//blobContainer[bli].frameCount > 10) {

                int detect=0;
                cv::Point temp, dir, lineend, rectpoint, arrow;
                float nm;
                double pi=3.14159;
                double t, a=30.0;
                char s[64];
                if (blobContainer[bli].collision == 1) {
                    rectpoint.x = blobContainer[bli].lastRectangle.x;
                    rectpoint.y = blobContainer[bli].lastRectangle.y;
                    temp.x = rectpoint.x + blobContainer[bli].lastRectangle.width/2.0;
                    temp.y = rectpoint.y + blobContainer[bli].lastRectangle.height/2.0;
                    //move over the detection line, setting detected flag
                    if((((temp.x<dir1.x && temp.y>dir1.y) && (blobContainer[bli].currentPosition.x<dir1.x && blobContainer[bli].currentPosition.y<dir1.y)) ||
                        ((temp.x>dir2.x && temp.y<dir2.y) && (blobContainer[bli].currentPosition.x>dir2.x && blobContainer[bli].currentPosition.y>dir2.y))) &&
                        !blobContainer[bli].detected) {
                        detect=1;
                        blobContainer[bli].detected=1;
                    }

                    dir.x = temp.x-blobContainer[bli].currentPosition.x;
                    dir.y = temp.y-blobContainer[bli].currentPosition.y;
                    nm = sqrt(pow(dir.x,2.0)+pow(dir.y,2.0));
                    nm = nm==0.0?1.0:nm;
                    //reset direction every 30 frames
                    if(blobContainer[bli].lineFrameCount-- < 0) {
                        blobContainer[bli].line.x = 0.0;
                        blobContainer[bli].line.y = 0.0;
                        blobContainer[bli].lineFrameCount = 30;
                    }
                    //draw direction
                    if((dir.x >= blobContainer[bli].lastRectangle.width/2.0 || dir.x <= blobContainer[bli].lastRectangle.width/-2.0) &&
                    (dir.y >= blobContainer[bli].lastRectangle.height/2.0 || dir.y <= blobContainer[bli].lastRectangle.height/-2.0)) {
                        //std::cout << bli << " " << dir.x/nm << " " << dir.y/nm << std::endl;
                        blobContainer[bli].currentPosition.x = temp.x;
                        blobContainer[bli].currentPosition.y = temp.y;
                        blobContainer[bli].line.x = dir.x/nm*16.0;
                        blobContainer[bli].line.y = dir.y/nm*16.0;
                    }
                    lineend.x = temp.x+blobContainer[bli].line.x;
                    lineend.y = temp.y+blobContainer[bli].line.y;
                    rectangle(rawFrame, blobContainer[bli].lastRectangle, cv::Scalar(0,0,255),2);
                    //moved
                    if(dir.x*dir.y != 0) {
                        cv::line(rawFrame, temp, lineend, cv::Scalar(0,255,0), 1, 4, 0);
                        //threshold
                        if((lineend.x-temp.x)*(lineend.y-temp.y)>=1.0 || (lineend.x-temp.x)*(lineend.y-temp.y)<=-1.0) {
                            if(detect) {
                                dir.y>0 ? car1++ : car2++;
                                //std::cout << "car1: " << car1 << " === car2: " << car2 << std::endl;
                            }
                            //draw arrow
                            t = atan2(double(-1.0*blobContainer[bli].line.y), double(-1.0*blobContainer[bli].line.x));
                            arrow.x = lineend.x + 8*cos(t+pi*a/180.0);
                            arrow.y = lineend.y + 8*sin(t+pi*a/180.0);
                            cv::line(rawFrame, lineend, arrow, cv::Scalar(0,255,0), 1, 4, 0);
                            arrow.x = lineend.x + 8*cos(t-pi*a/180.0);
                            arrow.y = lineend.y + 8*sin(t-pi*a/180.0);
                            cv::line(rawFrame, lineend, arrow, cv::Scalar(0,255,0), 1, 4, 0);
                        }
                    }
                } //unused
                /*else {
                    rectpoint.x = blobContainer[bli].lastRectangle.x;
                    rectpoint.y = blobContainer[bli].lastRectangle.y;
                    temp.x = rectpoint.x + blobContainer[bli].lastRectangle.width/2.0;
                    temp.y = rectpoint.y + blobContainer[bli].lastRectangle.height/2.0;
                    dir.x = temp.x-blobContainer[bli].currentPosition.x;
                    dir.y = temp.y-blobContainer[bli].currentPosition.y;
                    nm = sqrt(pow(dir.x,2.0)+pow(dir.y,2.0));
                    nm = nm==0.0?1.0:nm;
                    if(blobContainer[bli].lineFrameCount-- < 0) {
                        blobContainer[bli].line.x = 0.0;
                        blobContainer[bli].line.y = 0.0;
                        blobContainer[bli].lineFrameCount = 30;
                    }
                    if((dir.x >= blobContainer[bli].lastRectangle.width/2.0 || dir.x <= blobContainer[bli].lastRectangle.width/-2.0) &&
                    (dir.y >= blobContainer[bli].lastRectangle.height/2.0 || dir.y <= blobContainer[bli].lastRectangle.height/-2.0)) {
                        std::cout << bli << " " << dir.x/nm << " " << dir.y/nm << std::endl;
                        blobContainer[bli].currentPosition.x = temp.x;
                        blobContainer[bli].currentPosition.y = temp.y;
                        blobContainer[bli].line.x = dir.x/nm*16.0;
                        blobContainer[bli].line.y = dir.y/nm*16.0;
                    }
                    lineend.x = temp.x+blobContainer[bli].line.x;
                    lineend.y = temp.y+blobContainer[bli].line.y;
                    rectangle(rawFrame, blobContainer[bli].lastRectangle, cv::Scalar(255,0,0),2);
                    sprintf(s, "%d", bli);
                    //cv::putText(rawFrame, s, rectpoint, 0, 0.5, cv::Scalar(0,255,0), 1, 8, false);
                    if(dir.x*dir.y != 0) {
                        cv::line(rawFrame, temp, lineend, cv::Scalar(0,255,0), 1, 4, 0);
                        if((lineend.x-temp.x)*(lineend.y-temp.y)>=1.0 || (lineend.x-temp.x)*(lineend.y-temp.y)<=-1.0) {
                            t = atan2(double(-1.0*blobContainer[bli].line.y), double(-1.0*blobContainer[bli].line.x));
                            arrow.x = lineend.x + 8*cos(t+pi*a/180.0);
                            arrow.y = lineend.y + 8*sin(t+pi*a/180.0);
                            cv::line(rawFrame, lineend, arrow, cv::Scalar(0,255,0), 1, 4, 0);
                            arrow.x = lineend.x + 8*cos(t-pi*a/180.0);
                            arrow.y = lineend.y + 8*sin(t-pi*a/180.0);
                            cv::line(rawFrame, lineend, arrow, cv::Scalar(0,255,0), 1, 4, 0);
                        }
                    }
                }*/
            }
        }
        //draw detectionline
        cv::line(rawFrame, dir1, dir1end, cv::Scalar(255,128,255), 1, 4, 0);
        cv::line(rawFrame, dir2, dir2end, cv::Scalar(0,255,255), 1, 4, 0);
        //draw car number counter
        sprintf(s, "%d", car1);
        cv::putText(rawFrame, s, car1text, 0, 0.5, cv::Scalar(255,128,255), 1, 8, false);
        sprintf(s, "%d", car2);
        cv::putText(rawFrame, s, car2text, 0, 0.5, cv::Scalar(0,255,255), 1, 8, false);
        // Convert from cv::Mat to Qimage. Source: StackOverflow (http://stackoverflow.com/questions/5026965/how-to-convert-an-opencv-cvmat-to-qimage)
        //frameImageDebug = QImage((uchar*) foregroundFrame.data, foregroundFrame.cols, foregroundFrame.rows, foregroundFrame.step, QImage::Format_Indexed8);
        //frameImageDebug = QImage((uchar*) roiFrame.data, roiFrame.cols, roiFrame.rows, roiFrame.step, QImage::Format_RGB888);
        //frameImageDebug = QImage((uchar*) contoursFrame.data, contoursFrame.cols, contoursFrame.rows, contoursFrame.step, QImage::Format_RGB888);

        //frameImage = QImage((uchar*) rawFrame.data, rawFrame.cols, rawFrame.rows, rawFrame.step, QImage::Format_RGB888);
        cv::resize(rawFrame, rawFrame, cv::Size(rawFrame.size().width*2, rawFrame.size().height*2), 0, 0, CV_INTER_LINEAR);//double frame size
        //frameImage = ImageFormat::Mat2QImage(rawFrame);

        imwrite("/var/www/html/output.jpg", rawFrame);//write frame
        //videoFrameLabel->setPixmap(QPixmap::fromImage(frameImage));
        //videoFrameLabelDebug->setPixmap(QPixmap::fromImage(frameImageDebug));

        //fps config
        int key = cv::waitKey(100);
        if(key == 's')
            cv::waitKey(0);
        else if(key >= 0)
            break;
    }

    return 0;
}
