#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/face.hpp>

// Relative coordinates for haarcascade_lefteye_2splits.xml
#define EYE_SX 0.10
#define EYE_SY 0.19
#define EYE_SW 0.40
#define EYE_SH 0.36

class loadFacePoints;

using namespace cv;

double getSimilarity (const Mat A, const Mat B) {
    // less than 0.2 if image not move during capture process
    // higher than 0.4 if image move during capture process
    // set 0.3
    double errorL2 = norm(A, B, NORM_L2);
    double similarity = errorL2 / (double)(A.rows * A.cols);
    return similarity;

}
Mat converttoGray(Mat newImg) {
    Mat gray;
    if (newImg.channels() == 3) {
        cvtColor(newImg, gray, COLOR_BGR2GRAY);   // for 3 channel RGB
    } else if (newImg.channels() == 4) {
        cvtColor(newImg, gray, COLOR_BGR2GRAY);   // for 4 channel RGBA
    } else {
        gray = newImg; // directly
    }
    return gray;
}
void enlargeImg (Mat img, std::vector<Rect> &objvector, const int DETECTION_WIDTH, int scale) {
    if (img.cols > DETECTION_WIDTH) {
        objvector[0].x = cvRound(objvector[0].x * scale);
        objvector[0].y = cvRound(objvector[0].y * scale);
        objvector[0].width = cvRound(objvector[0].width * scale);
        objvector[0].height = cvRound(objvector[0].height * scale);
    }
    if (objvector[0].x < 0) {
        objvector[0].x = 0;
    }
    if (objvector[0].y < 0) {
        objvector[0].y = 0;
    }
    if (objvector[0].x + objvector[0].width > img.cols) {
        objvector[0].x = img.cols - objvector[0].width;
    }
    if (objvector[0].y + objvector[0].height > img.rows) {
        objvector[0].y = img.rows - objvector[0].height;
    }
}


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
    imshow("original image", img);
    waitKey(0);
    /*
    VideoCapture camera2 = VideoCapture(0);
    camera2.read(img);
    imshow("ddd", img);
    waitKey(0);
     */
    gray = converttoGray(img);
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
    } else {
        std::cout << "Failed to detect face " << std::endl;
        exit(-1);
    }

    //imshow("ddd", equalizedImg);
    //waitKey(0);
    Mat faceImg = equalizedImg(facesRectvector[0]);
    imshow("normal process detected face", faceImg);
    waitKey(0);

    // how to enlarge
    std::vector<Rect> objvector;  // only one image
    objvector = facesRectvector;
    enlargeImg (img, objvector, DETECTION_WIDTH, scale);
    Mat newImg = img(objvector[0]);
    imshow("enlarge", newImg);
    waitKey(0);
    // convert to gray again
    gray = converttoGray(newImg);

    imshow("enlarge2", gray);
    waitKey(0);

    faceImg = gray;

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
    std::string eyeCascadeFilenameL1 = "/opt/opencv/share/opencv4/haarcascades/haarcascade_eye.xml";
    //std::string eyeCascadeFilename2 = "/opt/opencv/share/opencv4/lbpcascades/lbpcascade_frontalface_improved.xml";
    std::string eyeCascadeFilenameR1 = "/opt/opencv/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
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
    eyeDetectorleft1.detectMultiScale(faceImg, rightEyeRect, searchScaleFactor, minNeighbors, flags, minFeatureSize);

    Point leftEye = Point(-1, -1);
    if (leftEyeRect[0].width > 0) {
        leftEye.x = leftEyeRect[0].x + leftEyeRect[0].width/2 + leftX;
        leftEye.y = leftEyeRect[0].y + leftEyeRect[0].width/2 + topY;
    }
    Point rightEye = Point(-1, -1);
    if (rightEyeRect[0].width > 0) {
        rightEye.x = rightEyeRect.at(0).x + rightEyeRect[0].width/2 + leftX;
        rightEye.y = rightEyeRect[0].y + rightEyeRect[0].width/2 + topY;
    }
    if (leftEye.x >= 0 && rightEye.x >=0) {
        std::cout << "Detected both eyes." << std::endl;
    } else {
        std::cout << "Failed to detect eyes" << std::endl;
        exit(-1);
    }
    imshow("lefteye", faceImg(leftEyeRect.at(0)));
    imshow("righteye", faceImg(rightEyeRect[0]));
    waitKey(0);

    // eyes geometrical transformation
    Point2f eyesCenter; // caculating eyes' center
    eyesCenter.x = (leftEye.x + rightEye.x) * 0.5f;
    eyesCenter.y = (leftEye.y + rightEye.y) * 0.5f;

    double dy = (rightEye.y - leftEye.y);
    double dx = (rightEye.x - leftEye.x);
    double len = sqrt(dx*dx + dy*dy); // length between lefteye center and righteye center
    double angle = atan2(dy, dx) * 180.0/CV_PI; // caculating the angle

    // Hand measurements ?? experience?? eye center should ideally be roughly at (0.16, 0.14) of a scaled face image
    const double DESIRED_LEFT_EYE_X = 0.16;
    const double DESIRED_RIGHT_EYE_X = (1.0f - 0.16);
    const double DESIRED_LEFT_EYE_Y = 0.14;
    const int DESIRED_FACE_WIDTH = 70;
    const int DESIRED_FACE_HEIGHT = 70;
    double desiredLen = (DESIRED_RIGHT_EYE_X - 0.16);
    double newscale = desiredLen * DESIRED_FACE_WIDTH / len;

    Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, newscale); // rotation matrix
    double ex = DESIRED_FACE_WIDTH * 0.5f - eyesCenter.x;   // shift new center on desired face
    double ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenter.y;
    rot_mat.at<double>(0, 2) += ex;
    rot_mat.at<double>(1,2) += ey;
    Mat warped = Mat(DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH, CV_8U, Scalar(128));  // empty face with gray color
    warpAffine(faceImg, warped, rot_mat, warped.size()); // rotation
    imshow("rotation", warped);
    waitKey(0);

    // histogram equalizaton seperately
    int w = warped.cols;
    int h = warped.cols;
    Mat wholeFace;
    equalizeHist(warped, wholeFace);
    int midX = w/2;
    Mat leftSide = warped(Rect(0,0,midX,h));
    Mat rightSide = warped(Rect(midX,0,w-midX,h));
    equalizeHist(leftSide, leftSide);   // equalize leftSide and stored into leftSide
    equalizeHist(rightSide, rightSide);
    imshow("whole sep equ", warped);
    imshow("left sep equ", leftSide);
    imshow("right sep equ", rightSide);
    waitKey(0);
    //combine three parts together
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            int v;
            if (x < w/4) {
                v = leftSide.at<uchar>(y, x);    // Left 25%
            } else if (x < w*2/4) { // Mid Left 25%
                int lv = leftSide.at<uchar>(y, x);
                int wv = wholeFace.at<uchar>(y, x);
                float f = (x - w*1/4) / (float)(w/4);
                v = cvRound((1.0f - f)*lv + f * wv);    // weighted by f
            } else if (x < w*3/4) { // Mid Right 25%
                int rv = rightSide.at<uchar>(y, x-midX);
                int wv = wholeFace.at<uchar>(y, x);
                float f = (x - w*2/4)/(float)(w/4);
                v = cvRound((1.0f - f) * wv + f * rv);
            } else {
                v = rightSide.at<uchar>(y, x-midX); // Right 25%
            }
            warped.at<uchar>(y, x) = v;
        }
    }
    imshow("merge equ", warped);
    waitKey(0);

    // Smoothing
    Mat filtered = Mat(warped.size(), CV_8U);
    bilateralFilter(warped, filtered, 0, 20.0, 2.0);
    imshow("smoothing", filtered);
    waitKey(0);
    // elliptical mask
    // create ellipse mask
    Mat mask = Mat(filtered.size(), CV_8UC1, Scalar(255));
    double dw = DESIRED_FACE_WIDTH;
    double dh = DESIRED_FACE_HEIGHT;
    Point faceCenter = Point(cvRound(dw * 0.5), cvRound((dh * 0.4)));   // ellipse center
    Size size = Size(cvRound(dw * 0.5), cvRound(dh * 0.8)); // ellipse size
    ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(0), FILLED);  // create ellipse mask with black color
    imshow("mask", mask);
    waitKey(0);
    filtered.setTo(Scalar(128), mask);  // applied mask on face image
    imshow("mergemask", filtered);
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