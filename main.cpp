#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>


using namespace cv;
int main( int argc, char** argv )
{
    CascadeClassifier faceDetector;
    std::string faceCascadeFilename = "/opt/opencv/share/opencv4/lbpcascades/lbpcascade_frontalface_improved.xml";
    try {
        faceDetector.load(faceCascadeFilename);
    } catch (cv::Exception e) {
        std::cerr << e.what() << std::endl;
    }
    if ( faceDetector.empty()) {
        std::cerr << "ERROR: Couldn't load Face Detector (";
        std::cerr << faceCascadeFilename << ")!" << std::endl;
        exit(1);
    }


    VideoCapture camera;
    camera.open (0);
    if (!camera.isOpened()) {
        std::cerr << "ERROR: Could not access the camera or video!" <<std::endl;
        exit(1);
    }

    //set solution
    camera.set (CAP_PROP_FRAME_WIDTH, 640);
    camera.set (CAP_PROP_FRAME_HEIGHT, 480);
   // std::cout << camera.get(CAP_PROP_FRAME_COUNT) << std::endl;
    Mat cameraFrame;
   // while (true) {
   camera >> cameraFrame;
   if (cameraFrame.empty()) {
       std::cerr << "ERROR: Couldn't grab a camera frame." << std::endl;
       exit(1);
   }
    //gray scale conversion

    Mat gray, img;
    img = cameraFrame;
    /*
    VideoCapture camera2 = VideoCapture(0);
    camera2.read(img);
    imshow("ddd", img);
    waitKey(0);
     */
    if (img.channels() == 3) {
        cvtColor(img, gray, COLOR_BGR2GRAY);   // for 3 channel RGB
    } else if (img.channels() == 4) {
        cvtColor(img, gray, COLOR_BGR2GRAY);   // for 4 channel RGBA
    } else {
        gray = img; // directly
    }
    // shrink resize
    const int DETECTION_WIDTH = 320;
    Mat smallImg;
    float scale = gray.cols / (float) DETECTION_WIDTH;   // cols for width
    if (gray.cols > DETECTION_WIDTH) {
        // Shrink the image while keeping the same aspect ratio
        int scaleHeight = cvRound (gray.rows / scale);
        resize(gray, smallImg, Size(DETECTION_WIDTH, scaleHeight));
    } else {
        smallImg = gray; // no shrink
    }


    // ??? more clear
    Mat equalizedImg;
    equalizeHist(smallImg, equalizedImg);

    int flags = CASCADE_SCALE_IMAGE;
    Size minFeatureSize (20, 20);
    float searchScaleFactor = 1.1f;
    int minNeighbors = 4;
    std::vector<Rect> faces;
    faceDetector.detectMultiScale(equalizedImg, faces, searchScaleFactor, minNeighbors, flags, minFeatureSize);
    std::cout << faces.size();
    if (faces[0].width > 0) {
        std::cout << "We detected a face!" <<std::endl;
    }
    imshow("ddd", equalizedImg);
    waitKey(0);
    Mat faceImg = equalizedImg(faces[0]);
    imshow("ddd Rect", faceImg);
    waitKey(0);

    /*
    Mat image;
    image = imread( argv[1], 1 );
    if( argc != 2 || !image.data )
    {
        printf( "No image data \n" );
        return -1;
    }
    namedWindow( "Display Image", WINDOW_AUTOSIZE );
    imshow( "Display Image", image );
    waitKey(0);
     */
    return 0;
}