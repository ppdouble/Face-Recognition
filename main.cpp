#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/face.hpp>

// Relative coordinates for haarcascade_lefteye_2splits.xml
#define EYE_SX 0.12
#define EYE_SY 0.17
#define EYE_SW 0.37
#define EYE_SH 0.36

class loadFacePoints;

using namespace cv;

int main( int argc, char** argv )
{
    // face detect

    // Capture, gray scale convert, shrink - resize, equalizeHist, set detector parameter, detect
    // capture
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
    img = cameraFrame;  // original face image stores in img
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


    // ??? more clear equalizeHist
    Mat equalizedImg;
    equalizeHist(smallImg, equalizedImg);

    // load detector and set parameter and detect
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

    int flags = CASCADE_SCALE_IMAGE;
    Size minFeatureSize (20, 20);
    float searchScaleFactor = 1.1f;
    int minNeighbors = 4;
    std::vector<Rect> facesRectvector;
    faceDetector.detectMultiScale(equalizedImg, facesRectvector, searchScaleFactor, minNeighbors, flags, minFeatureSize);
    std::cout << facesRectvector.size();
    if (facesRectvector[0].width > 0) {
        std::cout << "We detected a face!" <<std::endl;
    }

    //imshow("ddd", equalizedImg);
    //waitKey(0);
    Mat faceImg1 = equalizedImg(facesRectvector[0]);
    imshow("ddd Rect", faceImg1);
    waitKey(0);
    imshow("ddd Rect", img(facesRectvector[0]));
    waitKey(0);

    // how to enlarge
    std::vector<Rect> objects;  // only one image
    objects = facesRectvector;
    if (img.cols > DETECTION_WIDTH) {
        objects[0].x = cvRound(objects[0].x * scale);
        objects[0].y = cvRound(objects[0].y * scale);
        objects[0].width = cvRound(objects[0].width * scale);
        objects[0].height = cvRound(objects[0].height * scale);
    }
    if (objects[0].x < 0) {
        objects[0].x = 0;
    }
    if (objects[0].y < 0) {
        objects[0].y = 0;
    }
    if (objects[0].x + objects[0].width > img.cols) {
        objects[0].x = img.cols - objects[0].width;
    }
    if (objects[0].y + objects[0].height > img.rows) {
        objects[0].y = img.rows - objects[0].height;
    }

    imshow("eee Rect", img(objects[0]));
    waitKey(0);

    Mat faceImg;
    resize(equalizedImg, faceImg, Size(objects[0].width, objects[0].height));
    imshow("eee Rect", faceImg);
    waitKey(0);
    
    // Face preprocessing
    // Eye detect
    // search Region of left eye and right eye

    int leftX = cvRound (faceImg.cols * EYE_SX);
    int topY = cvRound(faceImg.rows * EYE_SY);
    int widthX = cvRound(faceImg.cols * EYE_SW);
    int heightY = cvRound(faceImg.rows * EYE_SH);
    int rightX = cvRound(faceImg.cols * (1.0-EYE_SX-EYE_SW));
    Mat topLeftOfFace = faceImg(Rect(leftX, topY, widthX, heightY));
    Mat topRightOfFace = faceImg(Rect(rightX, topY, widthX, heightY));


    CascadeClassifier eyeDetectorleft1, eyeDetectorright1;
    std::string eyeCascadeFilenameL1 = "/opt/opencv/share/opencv4/haarcascades/haarcascade_lefteye_2splits.xml";
    //std::string eyeCascadeFilename2 = "/opt/opencv/share/opencv4/lbpcascades/lbpcascade_frontalface_improved.xml";
    std::string eyeCascadeFilenameR1 = "/opt/opencv/share/opencv4/haarcascades/haarcascade_righteye_2splits.xml";
    //std::string eyeCascadeFilename4 = "/opt/opencv/share/opencv4/lbpcascades/lbpcascade_frontalface_improved.xml";
    // left detector
    try {
        eyeDetectorleft1.load(eyeCascadeFilenameL1);
    } catch (cv::Exception e) {
        std::cerr << e.what() << std::endl;
    }
    if ( eyeDetectorleft1.empty()) {
        std::cerr << "ERROR: Couldn't load Left eye Detector (";
        std::cerr << eyeCascadeFilenameL1 << ")!" << std::endl;
        exit(1);
    }
    // right detector
    try {
        eyeDetectorright1.load(eyeCascadeFilenameR1);
    } catch (cv::Exception e) {
        std::cerr << e.what() << std::endl;
    }
    if ( eyeDetectorright1.empty()) {
        std::cerr << "ERROR: Couldn't load Right eye Detector (";
        std::cerr << eyeCascadeFilenameR1 << ")!" << std::endl;
        exit(1);
    }

    // detect left eye from a detected face
    //Rect leftEyeRect; // store left eye position rectangle
    std::vector<Rect> leftEyeRect, rightEyeRect;
    //eyeDetectorleft1.detectMultiScale(topLeftOfFace, leftEyeRect);
    minNeighbors = 1;
    eyeDetectorleft1.detectMultiScale(faceImg, leftEyeRect, searchScaleFactor, minNeighbors, flags, minFeatureSize);
    eyeDetectorright1.detectMultiScale(faceImg, rightEyeRect, searchScaleFactor, minNeighbors, flags, minFeatureSize);

    Point leftEye = Point(-1, -1);
    if (leftEyeRect[0].width > 0) {
        leftEye.x = leftEyeRect[0].x + leftEyeRect[0].width/2 + leftX;
        leftEye.y = leftEyeRect[0].y + leftEyeRect[0].width/2 + topY;
    }
    Point rightEye = Point(-1, -1);
    if (rightEyeRect[0].width > 0) {
        rightEye.x = rightEyeRect[0].x + rightEyeRect[0].width/2 + leftX;
        rightEye.y = rightEyeRect[0].y + rightEyeRect[0].width/2 + topY;
    }
    if (leftEye.x >= 0 && rightEye.x >=0) {
        std::cout << "Detected both eyes." << std::endl;
    }
    imshow("lefteye", faceImg(leftEyeRect[0]));
    imshow("righteye", faceImg(rightEyeRect[0]));
    waitKey(0);


    /*
    int num_components = 10;;
    double threshold = 10.0;
    Ptr<face::FaceRecognizer> model = face::EigenFaceRecognizer::create(num_components, threshold);
    std::vector<Mat> preprocessed;
    std::vector<int> labels;
    model->train(preprocessed, labels);
    Mat target;
    int predicted_label = model->predict(target);
     */



    return 0;
}