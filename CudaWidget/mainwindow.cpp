#include "mainwindow.h"
#include "ui_mainwindow.h"




/* ---- CONSTRUCTOR -------------------------------------------------------- */
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

}

/* ---- DESTRUCTOR -------------------------------------------------------- */
MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    //cuda::CannyEdgeDetector
    qDebug() << "pushbtn clicked";

    mVideoStreamThread.beginVideoStream();

    qRegisterMetaType< cv::Mat >("cv::Mat");
    connect(&mVideoStreamThread,
            SIGNAL(frameReady(cv::Mat)),
            this,
            SLOT(prepAndDisplayNewFrame(cv::Mat)));

    vanishingPoint[0] = 320;
    vanishingPoint[1] = 240;


    //a = imageHeight*sigma;
    //b = imageWidth*sigma;
}

QImage MainWindow::convertOpenCVMatToQtQImage(Mat mat) {
    if(mat.channels() == 1) {                   // if grayscale image
        return QImage((uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);     // declare and return a QImage
    } else if(mat.channels() == 3) {            // if 3 channel color image
        cv::cvtColor(mat, mat, CV_BGR2RGB);     // invert BGR to RGB
        return QImage((uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);       // declare and return a QImage
    } else {
        qDebug() << "in convertOpenCVMatToQtQImage, image was not 1 channel or 3 channel, should never get here";
    }
    return QImage();        // return a blank QImage if the above did not work
}

void MainWindow::prepAndDisplayNewFrame(Mat imgOriginal)
{
    //Mat imgOriginal;            // input image
    Mat imgGrayscale;           // grayscale of input image
    Mat imgBlurred;             // intermediate blured image
    Mat imgCanny;               // Canny edge image
    Mat imgCanny_color;         // HoughLinesP image
    //int MAX_NUM_LINES = 50;
    //int maxNumLines = MAX_NUM_LINES;

    /* ---- Allocate memory for line information storage ---- */
    if((lines = (double **)calloc(MAX_NUM_LINES,sizeof(*lines))) == NULL)
    {
        cout << "** calloc failed!" << endl;
        return;
    }
    for(int i = 0; i<MAX_NUM_LINES; i++)
    {
        if((lines[i] = (double*)calloc(3,sizeof(*lines[i]))) == NULL)
        {
            cout << "* calloc failed!" << endl;
            return;
        }
    }

    //QString strFileName = QFileDialog::getOpenFileName();       // bring up open file dialog

    //imgOriginal = cv::imread(strFileName.toStdString());        // open image

    //VideoCapture vcap(0);

    //while(1)
    //{
        //vcap.read(imgOriginal);
        /*
        if(imgOriginal.empty())
        {
            continue;
        }
        */

        //while(imgOriginal.empty())
        //{
        //    waitKey(100);
        //}

        Ptr<CannyEdgeDetector> cannyEdge = cuda::createCannyEdgeDetector(50.0,75.0,3,false);
        //Ptr<CannyEdgeDetector> cannyEdge = cuda::createCannyEdgeDetector(1200.0,1300.0,5,false);

        cv::cvtColor(imgOriginal, imgGrayscale, CV_BGR2GRAY);               // convert to grayscale
        GaussianBlur(imgGrayscale, imgBlurred, cv::Size(3, 3), 1.5);    // blur
        //Canny(imgBlurred, imgCanny, 50, 200,3);                          // get Canny edges

        GpuMat d_imgCanny;//(imgCanny);
        GpuMat d_imgCanny32;//(imgCanny)
        GpuMat d_imgBlurred(imgBlurred);
        GpuMat d_imgBlurred32(imgBlurred);
        GpuMat d_lines;

        int numLines = 0;
        bool lineStored = false;

        const int64 start = getTickCount();

        //d_imgBlurred.convertTo(d_imgBlurred32, CV_32SC1);
        //d_imgCanny32.convertTo(d_imgCanny32, CV_32SC1);
        cannyEdge->detect(d_imgBlurred, d_imgCanny);

        if(d_imgCanny.empty())
        {
            cout << "Cuda canny failed!" << endl;
        }

        d_imgCanny.download(imgCanny);
/*
        Ptr<HoughSegmentDetector> hough = createHoughSegmentDetector(1.0f
                                                                     , (float) (CV_PI / 180.0f)
                                                                     , 150    // minLineLength.
                                                                     , 50     // maxLineGap.
                                                                     , MAX_NUM_LINES // Maximum number of lines to detect.
                                                                     );
*/

        Ptr<HoughSegmentDetector> hough = createHoughSegmentDetector(1.0f
                                                                     , (float) (CV_PI / 180.0f)
                                                                     , 50    // minLineLength.
                                                                     , 35     // maxLineGap.
                                                                     , MAX_NUM_LINES // Maximum number of lines to detect.
                                                                     );

        hough->detect(d_imgCanny, d_lines);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        //cout << "GPU Time : " << timeSec * 1000 << " ms" << endl;
        //cout << "GPU Found : " << d_lines.cols << endl;

        vector<Vec4i> lines_gpu;
        if (!d_lines.empty())
        {
            lines_gpu.resize(d_lines.cols);
            Mat h_lines(1, d_lines.cols, CV_32SC4, &lines_gpu[0]);
            d_lines.download(h_lines);
        }

        double x_sum = 0;
        double y_sum = 0;
        double x_mean, y_mean;
        for (size_t i = 0; i < lines_gpu.size(); ++i)
        {
            Vec4i l = lines_gpu[i];
            //line(imgOriginal, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
            //cout << "First point: ";
            //cout << Point(l[0], l[1]) << endl;
            //cout << "Second point: ";
            //cout << Point(l[2], l[3]) << endl;
            //numLines = (int)i;

            lineStored = storeLine(Point(l[0], l[1]),Point(l[2], l[3]),numLines);
            if(lineStored)
            {
                x_sum += l[0] + l[2];
                y_sum += l[1] + l[3];
                numLines++;
            }

            //waitKey();
        }

        x_mean = x_sum/(2*numLines);
        y_mean = y_sum/(2*numLines);
        cv::Point2f mean(x_mean,y_mean);
        double X[2] = {x_mean, y_mean};
        double covarianceArray[2][2] = {0};
        for(int j = 0; j<2; j++)
        {
            for(int k = 0; k<2; k++)
            {
                for (size_t i = 0; i < lines_gpu.size(); ++i)
                {
                   Vec4i l = lines_gpu[i];
                   covarianceArray[j][k] += (l[0+j] - X[j])*(l[0+k] - X[k]);
                   covarianceArray[j][k] += (l[2+j] - X[j])*(l[2+k] - X[k]);
                }
                covarianceArray[j][k] = (1.0/ (2*numLines - 1) )*covarianceArray[j][k];
            }
        }

        Mat covmat = (Mat_<double>(2,2) <<
                      covarianceArray[0][0], covarianceArray[0][1],
                      covarianceArray[1][0], covarianceArray[1][1]
                      );

        Mat eigenvalues, eigenvectors;
        eigen(covmat, eigenvalues, eigenvectors);
        double angle = qAtan2(eigenvectors.at<double>(0,1), eigenvectors.at<double>(0,0));

        //Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
        if(angle < 0)
            angle += 6.28318530718;

        //Conver to degrees instead of radians
        angle = 180*angle/3.14159265359;

        //Calculate the size of the minor and major axes
        double halfmajoraxissize=2.4477*sqrt(eigenvalues.at<double>(0));
        double halfminoraxissize=2.4477*sqrt(eigenvalues.at<double>(1));

        //Return the oriented ellipse
        //The -angle is used because OpenCV defines the angle clockwise instead of anti-clockwise
        RotatedRect ellipse = RotatedRect(mean, cv::Size2f(halfmajoraxissize, halfminoraxissize), -angle);
        cv::ellipse(imgOriginal, ellipse, Scalar::all(255), 2);

        /* Could be used for later ;)*/
        //int rows = imgOriginal.rows;
        //int cols = imgOriginal.cols;
        //qDebug() << "imgOriginal dims: " + QString::number(rows) + " " + QString::number(cols);

        /*
        vector<Vec4i> lines;
        HoughLinesP( imgCanny, lines, 1, CV_PI/180, 80, 30, 10 );
        cuda::HoughLinesDetector
        for( size_t i = 0; i < lines.size(); i++ )
        {
            float x1 = lines[i][0], y1 = lines[i][1];
            float x2 = lines[i][2], y2 = lines[i][3];


            line( imgOriginal, Point(lines[i][0], lines[i][1]),
            Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 4 );
        }

*/
/*
        vector<Vec2f> lines;
        HoughLines(imgCanny, lines, 1, CV_PI/180, 100, 0, 0 );

        for( size_t i = 0; i < lines.size(); i++ )
        {
           float rho = lines[i][0], theta = lines[i][1];
           Point pt1, pt2;
           double a = cos(theta), b = sin(theta);
           double x0 = a*rho, y0 = b*rho;
           pt1.x = cvRound(x0 + 1000*(-b));
           pt1.y = cvRound(y0 + 1000*(a));
           pt2.x = cvRound(x0 - 1000*(-b));
           pt2.y = cvRound(y0 - 1000*(a));
           line( imgOriginal, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
        }
*/

        /*
        if(!imgOriginal.empty()||!imgLines.empty())
        {
            //mVideoStreamThread.unlockMutex();
            qDebug() << "No lines were found.";
        }
        else
        {
            imshow("detected lines", imgLines);
            qDebug() << "Lines were found.";
        }
        */

        //printStoredLines(numLines);
        calculateLineIntersect(numLines);

        /* ---- Draw intersection points---- */
        /*
        for(int i = 0; i < numStoredPoints; i++)
        {
            Point p;
            p.x = lineIntersections[i][0];
            p.y = lineIntersections[i][1];
            if( !( (p.x < 0 || 640 < p.x) || (p.y < 0 || 480 < p.y) ) )
            {
                circle(imgOriginal,p,7,Scalar(255, 0, 0),3);

            }
        }
*/

        ptr_vanishingPoint = (int *)malloc(2*sizeof(int));
        findVanishingPoint(numStoredPoints);

        if((0 <= ptr_vanishingPoint[0] && ptr_vanishingPoint[0]< 640) && (0 <= ptr_vanishingPoint[1] && ptr_vanishingPoint[1] < 480))
        {
            vanishingPoint[0] = ptr_vanishingPoint[0];
            vanishingPoint[1] = ptr_vanishingPoint[1];
        }

        //circle(imgOriginal,Point((int)vanishingPoint[0],(int)vanishingPoint[1]),10,Scalar(0, 255, 0),3);
        circle(imgOriginal,Point(vanishingPoint[0],vanishingPoint[1]),10,Scalar(0, 255, 0),3);

        free(ptr_vanishingPoint);

        QImage qimgOriginal = convertOpenCVMatToQtQImage(imgOriginal);  // convert original and Canny images to QImage
        QImage qimgCanny = convertOpenCVMatToQtQImage(imgCanny);        //

        ui->imageOriginal->setPixmap(QPixmap::fromImage(qimgOriginal));   // show original and Canny images on labels
        ui->imageCanny->setPixmap(QPixmap::fromImage(qimgCanny));         //

        /* ---- Free memory before next iteration ---- */
        for (int i = 0; i < sizeAllocated; i++)
        {
           free(lineIntersections[i]);
        }
        free(lineIntersections);

        for(int i = 0; i < numLines; i++)
        {
            free(lines[i]);
        }
        free(lines);

        mVideoStreamThread.unlockMutex(); // Unable calls to this function before it is finished.
    //}
}

/**
 * @brief MainWindow::storeLine
 * @param p1
 * @param p2
 *
 * Stores lines in the vector "lines". The format is: [slope M, base].
 * Functionality may, and has been confirmed by the function
 * "printStoredLines".
 */
bool MainWindow::storeLine(Point p1, Point p2, int lineNumber)
{
    /* ---- Calculate line parameters. ---- */
    double x1 = (double)p1.x, x2 = (double)p2.x;
    double y1 = (double)p1.y, y2 = (double)p2.y;
    double delta_x = x2 - x1, delta_y = y2 - y1;
    double M = (delta_y)/(delta_x);
    double base = x1*M + y1; // Changed sign. TODO: Verify!
    double length = qSqrt(qPow(delta_y,2) + qPow(delta_x,2));

    if(M < -60000.0 || M > 60000 || M == 0)
    {
        // Line is considered unsuitable for further calculations.
        //cout << "Line is considered unsuitable for further calculations. "<<endl;
        return false;
    }
    if(base < -60000.0 || base > 60000)
    {
        // Line is considered unsuitable for further calculations.
        //cout << "Line is considered unsuitable for further calculations. "<<endl;
        return false;
    }

    lines[lineNumber][0] = M;
    lines[lineNumber][1] = base;
    lines[lineNumber][2] = length;
/*
    cout << "Line to be stored. "<<endl;
    cout << "P1" << endl;
    cout << x1 <<endl;
    cout << y1 <<endl;
    cout << "P2" <<endl;
    cout << x2 <<endl;
    cout << y2 <<endl;

    cout << M <<endl;
    cout << base <<endl;
*/
    return true;
}

/**
 * @brief MainWindow::printStoredLines
 * A test function. Used to check if data contained in the vector "lines" is
 * stored correctly.
 */
void MainWindow::printStoredLines(int numLines)
{
    ///cout << "Called print stored lines." + to_string(numLines) <<endl;

    for(int i = 0; i < numLines; i++)
    {
        double M = lines[i][0];
        double base = lines[i][1];
        //cout << "Stored line: "<<endl;
        //cout << M <<endl;
        //cout << base <<endl;
    }
}

int MainWindow::calculateLineIntersect(int numLines)
{
    sizeAllocated = MAX_NUM_LINES;
    /* ---- Initial memory allocation for line intercect information storage ---- */
    if((lineIntersections = (double **)calloc(sizeAllocated,sizeof(*lineIntersections))) == NULL)
    {
        cout << "** calloc failed!" << endl;
        return -1;
    }
    for(int i = 0; i<sizeAllocated; i++)
    {
        if((lineIntersections[i] = (double*)calloc(3,sizeof(*lineIntersections[i]))) == NULL)
        {
            cout << "* calloc failed!" << endl;
            return -1;
        }
    }

    numStoredPoints = 0;

    for(int i = 0; i < numLines; i++)
    {
        double M1 = lines[i][0];
        double base1 = lines[i][1];
        double length1 = lines[i][2];
        for(int j = i+1; j < numLines; j++)
        {
            double M2 = lines[j][0];
            double base2 = lines[j][1];
            double length2 = lines[j][2];

            Point p;
            //cout << to_string(M1) + " " + to_string(base1) + " " + to_string(M2) + " " + to_string(base2) << endl;
            p.x = (int) ((base1-base2) / (M2 - M1));
            p.y = (int) ((M1*p.x) + base1);
            //cout << "Intersection points: ";
            //cout << to_string(p.x) + " " + to_string(p.y) << endl;
            if((0 <= p.x && p.x< 640) && (0 <= p.y && p.y < 480))
            {
                double weight = length1*length2;   // w(p) = |length of line 1|*|length of line 2|
                int ret = storeLineIntersection(p.x, p.y, weight, numStoredPoints, sizeAllocated);
                if(ret > 0)
                {
                    sizeAllocated = ret;
                    numStoredPoints++;
                    //qDebug() << "Stored an intersection point: "
                      //          + QString::number(p.x)
                        //        + " "
                          //      + QString::number(p.y)
                            //    + " "
                              //  + QString::number(weight);
                }
                else
                {
                    return -1;
                }
            }
        }
    }
    return 1;
}

int MainWindow::storeLineIntersection(int x, int y, double weight, int numStoredPoints, int sizeAllocated)
{
    if(numStoredPoints < sizeAllocated)
    {
        lineIntersections[numStoredPoints][0] = (double) x;
        lineIntersections[numStoredPoints][1] = (double) y;
        lineIntersections[numStoredPoints][2] = weight; // Wheight asociated with the point p.
    }
    else // More memory must be allocated.
    {
        int ret = expandLineIntersectionStorage(sizeAllocated);
        if(ret > 0)
        {
            sizeAllocated = ret;
            qDebug() << "Allocated more memory.";
        }
        else
        {
            qDebug() << "Failed to allocate memory.";
            return -1;
        }

        lineIntersections[numStoredPoints][0] = (double) x;
        lineIntersections[numStoredPoints][1] = (double) y;
        lineIntersections[numStoredPoints][2] = weight; // Wheight asociated with the point p.

    }
    return sizeAllocated;
}

int MainWindow::expandLineIntersectionStorage(int sizeAllocated)
{
    double **tmp;
    if ((tmp = (double**)realloc(lineIntersections, sizeof(double *) * (sizeAllocated + MAX_NUM_LINES))) == NULL)
    {
        /* Possible free on ip? Depends on what you want */
        return -1;
        fprintf(stderr, "ERROR: realloc failed");
    }
    sizeAllocated = sizeAllocated + MAX_NUM_LINES;
    lineIntersections = tmp;
    for(int i = 0; i<sizeAllocated; i++)
    {
        if((lineIntersections[i] = (double*)calloc(3,sizeof(*lineIntersections[i]))) == NULL)
        {
            cout << "* calloc failed!" << endl;
            return -1;
        }
    }

    return sizeAllocated;
}

void MainWindow::findVanishingPoint(int numStoredPoints)
{
    // Initialize image grid and set initial cell weight to zero.
    double B[10][10]; // Set of grid cells B_ij, i = 1,...,10, j = 1,...,10
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            B[i][j] = 0.0;
        }
    }

    bool breakFlag;
    for(int k = 0; k < numStoredPoints; k++)
    {
        Point p;
        p.x = lineIntersections[k][0];
        p.y = lineIntersections[k][1];
        int weight = lineIntersections[k][2];
        breakFlag = false;
        double ret;
        int i, j;
        for(i = 0; i<10; i++)
        {
            for(j = 0; j < 10; j++)
            {
                if(pointFitsInCell(p.x,p.y,i,j))
                {
                    ret = calcPointWeightContribution(p.x,
                                                      p.y,
                                                      weight,
                                                      i,
                                                      j);

                    breakFlag = true;
                    break;
                }
            }
            if(breakFlag){break;}
        }
        if(ret < 0)
        {
            //TODO: Handle the error.
        }
        else
        {
            //qDebug() << "Point added to cell: "
              //          + QString::number(i)
                //        + " "
                  //      + QString::number(j)
                    //    + " "
                      //  + QString::number(ret);
            B[i][j] += ret;
        }
    }
    int i, j;
    int heaviestCell[2]; // {i,j}
    heaviestCell[0] = -1.0;
    heaviestCell[1] = -1.0;
    double largestWeight = 0.0;
    for(i = 0; i<10; i++)
    {
        for(j = 0; j < 10; j++)
        {
            if(B[i][j] > largestWeight)
            {
                heaviestCell[0] = i;
                heaviestCell[1] = j; // For zero indexing
            }
        }
    }
    if( !(heaviestCell[0] > 0 && heaviestCell[1] > 0) ){return;} // No vanishing point was found.
    //qDebug() << "Heaviest cell: " + QString::number(heaviestCell[0]) + " " + QString::number(heaviestCell[1] );
    double sum_Xpos = 0.0, sum_yPos = 0.0;
    double average_Xpos, average_yPos;
    double pointsInCell = 0.0;
    int cellLeftBound = 64*heaviestCell[0], cellRightBound = 64*(heaviestCell[0]+1);
    int cellLowerBound = 48*heaviestCell[1], cellUpperBound = 48*(heaviestCell[1]+1);
    for(int k = 0; k < numStoredPoints; k++)
    {
        Point p;
        p.x = lineIntersections[k][0];
        p.y = lineIntersections[k][1];

        //qDebug() << "Point: " + QString::number(p.x) + " " + QString::number(p.y);
        //qDebug() << "Bounds: "
          //          + QString::number(cellLeftBound) + " " + QString::number(cellRightBound) + " "
            //        + QString::number(cellLowerBound) + " " + QString::number(cellUpperBound);
        if((cellLeftBound <= p.x && p.x< cellRightBound) && (cellLowerBound <= p.y && p.y < cellUpperBound))
        {
            // The point is in the cell.
            pointsInCell += 1.0;
            sum_Xpos += (double)p.x;
            sum_yPos += (double)p.y;
        }
    }
    average_Xpos = sum_Xpos/pointsInCell;
    average_yPos = sum_yPos/pointsInCell;
    //qDebug() << "Points in cell: " + QString::number(pointsInCell);
    //qDebug() << "sum position: " + QString::number(sum_yPos);
    //qDebug() << "Vanishingpoint position: " + QString::number(average_Xpos) + " " + QString::number(average_yPos);

    // Store vanishing point coordinates.
    ptr_vanishingPoint[0] = (int) average_Xpos;
    ptr_vanishingPoint[1] = (int) average_yPos;

    return;
}

bool MainWindow::pointFitsInCell(int x, int y, int i, int j)
{
    int i_upper = i+1;    // Correct for zero indexation.
    int j_upper = j+1;    // Correct for zero indexation.
    if(!(0 <= x && x< 640) && !(0 <= y && y < 480))
    {
        return false; // The point is off the screen.
    }
    if((x < i_upper*64) && (y < j_upper*48)) // Upper bound.
    {
        if((x > i*64) && (y > j*48)) // Lower bound.
        {
            return true; // The point fits into the inquired cell.
        }
    }
    return false;
}

double MainWindow::calcPointWeightContribution(int x,
                                               int y,
                                               int weight,
                                               int rowNum,
                                               int colNum)
{
    double xKernel, yKernel, pointWeight;
    // y-axis kernel
    xKernel = sigmoidKernel_1D(x,rowNum, 'x');
    yKernel = sigmoidKernel_1D(x,colNum, 'y');

    pointWeight = weight*xKernel*yKernel;
    if(pointWeight < 0)
    {
        return -1.0;
    }
    return  pointWeight;
}

double MainWindow::sigmoidKernel_1D(int x,
                                   int rowNum,
                                   char selectedAxis)
{
    double kernelWeight = 0.0;
    int a, b; // a: lower bound. b: upper bound. a <= b;
    if(selectedAxis == 'x')
    {
        a = rowNum*(imageWidth*sigma)+x_padding;
        b = (rowNum+1)*(imageWidth*sigma)-x_padding;

    }
    else if(selectedAxis = 'y')
    {
        a = rowNum*(imageHeight*sigma)+x_padding;
        b = (rowNum+1)*(imageHeight*sigma)-x_padding;

    }
    else
    {
        //No valid choice of axis.
        return -1.0;
    }
    double u = b - a;   // Helper variable.
    double c = 1 / (1+qExp( -lambda*(x-a) )); // Helper variable.
    double d = 1 / (1+qExp( -lambda*(x-b) )); // Helper variable.
    kernelWeight = (c - d)/u;

    return kernelWeight;
}

