#ifndef VIDEOSTREAMTHREAD_H
#define VIDEOSTREAMTHREAD_H

#include <QtCore>
#include <QDebug>

#include <stdio.h>

#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <opencv\cv.h>
#include <opencv\highgui.h>

using namespace cv;
using namespace std;

class VideoStreamThread : public QThread
{
    Q_OBJECT

public:
    VideoStreamThread(QObject *parent = 0);

    void beginVideoStream();
    Mat VideoStreamThread::getCameraFeed();
    void VideoStreamThread::setRun(bool _continueRunning);

    Size boardSize;

    bool continueRunning = true;

    void unlockMutex();


protected:
    void run() Q_DECL_OVERRIDE;

signals:
   void frameReady(cv::Mat frame);//, string frameSide);

public slots:

private:
    int modeOfOperation;
    string videoStreamAddress;
    string windowName;
    VideoCapture capture;
    Mat cameraFeed, exportedFeed;

    vector<vector<Point2f> > image_points;          //2D image points
    vector<vector<Point3f> > object_points;         //3D object points

    string IMAGE_FOLDER = "C:\\Users\\vegarsl\\Pictures\\StereoSamples\\Calibration\\Data\\left\\";
    string DATA_FOLDER = "C:\\Users\\vegarsl\\Pictures\\StereoSamples\\Calibration\\Data\\";

    vector<vector<Point2f> > imagePoints;

    QMutex mutex;

};

#endif // VIDEOSTREAMTHREAD_H
