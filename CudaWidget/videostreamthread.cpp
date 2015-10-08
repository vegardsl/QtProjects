

#include "videostreamthread.h"

VideoStreamThread::VideoStreamThread(QObject *parent) : QThread(parent)
{

}

void VideoStreamThread::unlockMutex()
{
    mutex.unlock();
    //qDebug() << "Mutex released";
}

void VideoStreamThread::beginVideoStream(){
   //videoStreamAddress = address;
   //windowName = newWindowName;
    start();
}

Mat VideoStreamThread::getCameraFeed(){
    //if(!exportedFeed.empty())
        return cameraFeed;
    //else return cameraFeed;
}

void VideoStreamThread::setRun(bool _continueRunning)
{
    continueRunning = _continueRunning;
}

void VideoStreamThread::run()
{

    qDebug() << "Running";

    VideoCapture capture(0);
    cout << "Opening video stream." << endl;

    int iterationCounter = 0;

    while (continueRunning){


        //store image to matrix
        //capture.grab();
        //capture.retrieve(cameraFeed);
        capture.read(cameraFeed);

        iterationCounter++;
        //if(iterationCounter%30 == 0){
            //std::cout << "30 iterations performed on "+windowName << std::endl;
            //fastNlMeansDenoisingColored(cameraFeed, exportedFeed, 3,3,7,21);
        if(mutex.tryLock(0)){
            //qDebug() << "Mutex locked";
            emit frameReady(cameraFeed);//, windowName);
        }

        //}

        //cout << "Time elapsed: " << t * 1000 / getTickFrequency() << std::endl;

        //imshow(windowName, cameraFeed);
        waitKey(30);
    }
    capture.release();
    /* VideoStreamThread is finished. */
}
