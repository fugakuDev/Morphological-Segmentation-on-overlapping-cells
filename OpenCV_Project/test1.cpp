#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat convert2bin(const Mat& grayscaleImage) {

    int thresholdValue = 128;
    Mat binaryImage = Mat::zeros(grayscaleImage.size(), CV_8UC1);

    for (int i = 0; i < grayscaleImage.rows; ++i) {
        for (int j = 0; j < grayscaleImage.cols; ++j) {
            //binaryImage.at<uchar>(i, j) = 255 - grayscaleImage.at<uchar>(i, j);
            if (grayscaleImage.at<uchar>(i, j) >= thresholdValue) {
                binaryImage.at<uchar>(i, j) = 0;
            }
            else {
                binaryImage.at<uchar>(i, j) = 255;
            }
        }
    }
    
    return binaryImage;

}

bool is_valid(int y, int x, int R, int C) {
    return (y >= 0 && y < R && x >= 0 && x < C);
}

Mat erodeImage(const Mat& original_image, int kernelSize, int background = 0) {
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(kernelSize, kernelSize));
    //kernel.setTo(255);

    Mat eroded_image = Mat::zeros(original_image.size(), original_image.type());

    int padding = kernelSize / 2;

    for (int y = 0; y < original_image.rows; ++y) {
        for (int x = 0; x < original_image.cols; ++x) {
            int pixelValue, minValue = 255;
            for (int ky = -padding; ky <= padding; ++ky) {
                for (int kx = -padding; kx <= padding; ++kx) {
                    if (!kernel.at<uchar>(ky + padding, kx + padding)) {          //Considering flat SEs
                        continue;
                    }
                    if (is_valid(y + ky, x + kx, original_image.rows, original_image.cols)) {
                        pixelValue = original_image.at<uchar>(y + ky, x + kx);
                    }
                    else {
                        pixelValue = background;
                    }
                    if (pixelValue < minValue) {
                        minValue = pixelValue;
                    }
                }
            }
            eroded_image.at<uchar>(y, x) = minValue;
        }
    }

    return eroded_image;
}


Mat dilateImage(const Mat& original_image, int kernelSize, int background = 0) {
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(kernelSize, kernelSize));
    //kernel.setTo(255);

    Mat dilated_image = Mat::zeros(original_image.size(), original_image.type());

    int padding = kernelSize / 2;

    for (int y = 0; y < original_image.rows; ++y) {
        for (int x = 0; x < original_image.cols; ++x) {
            int pixelValue, maxValue = 0;
            for (int ky = -padding; ky <= padding; ++ky) {
                for (int kx = -padding; kx <= padding; ++kx) {
                    if (!kernel.at<uchar>(ky + padding, kx + padding)) {          //Considering flat SEs
                        continue;
                    }
                    if (is_valid(y + ky, x + kx, original_image.rows, original_image.cols)) {
                        pixelValue = original_image.at<uchar>(y + ky, x + kx);
                    }
                    else {
                        pixelValue = background;
                    }
                    if (pixelValue > maxValue) {
                        maxValue = pixelValue;
                    }
                }
            }
            dilated_image.at<uchar>(y, x) = maxValue;
        }
    }

    return dilated_image;
}

Mat openImage(const Mat& original_image, int kernelSize, int background = 0) {
                                    //padding is dark in order to prevent small brighter features on border
    Mat eroded_image = erodeImage(original_image, kernelSize, background);
    return dilateImage(eroded_image, kernelSize, background);
}

Mat closeImage(const Mat& original_image, int kernelSize, int background = 255) {
                                    //padding is bright in order to prevent small darker features on border
    Mat dilated_image = dilateImage(original_image, kernelSize);
    return erodeImage(dilated_image, kernelSize, background);
}

Mat topHat(const Mat& original_image, int kernelSize) {
    kernelSize = 15;
    return (original_image - openImage(original_image, kernelSize));
}

Mat bottomHat(const Mat& original_image, int kernelSize) {
    return (closeImage(original_image, kernelSize) - original_image);
}

Mat enhance(const Mat& original_image, int kernelSize) {
    return (original_image + 2*topHat(original_image, kernelSize) - 3*bottomHat(original_image, kernelSize));
}

vector<vector<int>> delta{ {-1, 0}, {0, 1}, {1, 0}, {0, -1} };

void dfs(vector<pair<int, int>> &temp, int y, int x, const Mat& binary_image, vector<vector<int>> &visited) {
    
    visited[y][x] = 1;
    temp.push_back(make_pair(y, x));

    int r = binary_image.rows, c = binary_image.cols;

    for (int k = 0; k < delta.size(); k++) {
        int dy = delta[k][0], dx = delta[k][1];
        if (is_valid(y + dy, x + dx, r, c) && binary_image.at<uchar>(y + dy, x + dx) == binary_image.at<uchar>(y, x) && !visited[y + dy][x + dx]) {
            dfs(temp, y + dy, x + dx, binary_image, visited);
        }
    }
}

int convert01(int x) {
    return ((x == 0) ? 0 : 1);
}

void connectedComp(const Mat& binary_image, vector<set<pair<int, vector<pair<int, int>>>>> &cc) {

    int r = binary_image.rows, c = binary_image.cols;

    vector<vector<int>> visited(r, vector<int>(c, 0));

    for (int y = 0; y < r; ++y) {
        for (int x = 0; x < c; ++x) {
            if (!visited[y][x]) {
                vector<pair<int, int>> temp;
                dfs(temp, y, x, binary_image, visited);
                cc[convert01(binary_image.at<uchar>(y, x))].insert(make_pair(temp.size(), temp));
            }
        }
    }
}

Mat fillHoles(const Mat& original_image) {

    vector<set<pair<int, vector<pair<int, int>>>>> cc(2);
    connectedComp(original_image, cc);

    vector<vector<Point>> contours;
    findContours(original_image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat resultImage = original_image.clone();
    drawContours(resultImage, contours, -1, cv::Scalar(255), -1);


    imwrite("filled_image.png", resultImage);

    return resultImage;
}

Mat majority(const Mat& original_image, int background = 0) {           //Noise Reduction and edge smoothening
    
    Mat mod_image = Mat::zeros(original_image.size(), original_image.type());

    for (int y = 0; y < original_image.rows; ++y) {
        for (int x = 0; x < original_image.cols; ++x) {
            int pixelValue, cnt = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    if (ky == 0 && kx == 0) {
                        continue;
                    }
                    if (is_valid(y + ky, x + kx, original_image.rows, original_image.cols)) {
                        pixelValue = original_image.at<uchar>(y + ky, x + kx);
                    }
                    else {
                        pixelValue = background;
                    }
                    if (pixelValue) {
                        cnt++;
                    }
                }
            }
            if (cnt >= 5) {
                mod_image.at<uchar>(y, x) = 255;
            }
        }
    }
    return mod_image;
}


int main() {

    //Mat original_image = imread("grayscale3.png", IMREAD_GRAYSCALE);
    //Mat original_image = imread("enhanced_image.png", IMREAD_GRAYSCALE);
    Mat original_image = imread("output_binary.png", IMREAD_GRAYSCALE);
    //Mat original_image = imread("filled_image.png", IMREAD_GRAYSCALE);
    //Mat original_image = imread("opened_image.png", IMREAD_GRAYSCALE);

    if (original_image.empty()) {
        cerr << "Could not open or find the image." << endl;
        return 1;
    }

    int choice, kernelSize;
    cout << "For EROSION operation PRESS 1\nFor DILATION operation PRESS 2\nFor OPENING operation PRESS 3\n";
    cout << "For CLOSING operation PRESS 4\nFor TopHat operation PRESS 5\nFor BottomHat operation PRESS 6\n";
    cout << "For Enhancement of Contrast PRESS 7\nFor BINARY Conversion PRESS 8\nFor Filling holes PRESS 9\n";
    cout << "For Majority operation PRESS 10\nFor EXIT PRESS 0\n\n";
    cin >> choice;

    if (!choice) {
        return 0;
    }
    else if (choice == 8) {
        Mat binaryImage = convert2bin(original_image);
        imwrite("output_binary.png", binaryImage);
        imshow("Original Image", original_image);
        imshow("Output Binary Image", binaryImage);
        waitKey(0);
        return 0;
    }
    else if (choice == 9) {
        Mat filled_image = fillHoles(original_image);
        imwrite("filled_image.png", filled_image);
        imshow("Original Image", original_image);
        imshow("Holes filled Image", filled_image);
        waitKey(0);
        return 0;
    }
    else if (choice == 10) {
        Mat mod_image = majority(original_image);
        imwrite("mod_image.png", mod_image);
        imshow("Original Image", original_image);
        imshow("Mod filtered Image", mod_image);
        waitKey(0);
        return 0;
    }

    cout << "Enter kernel size (an odd integer): ";
    cin >> kernelSize;

    if (kernelSize % 2 != 1) {
        cout << "Kernel not of odd size!! Termination!!\n";
        return 0;
    }

    switch (choice) {

    case 1: {
        Mat eroded_image = erodeImage(original_image, kernelSize);
        imwrite("eroded_image.png", eroded_image);
        imshow("Original Image", original_image);
        imshow("Eroded Image", eroded_image);
        waitKey(0);
        break;
    }

    case 2: {
        Mat dilated_image = dilateImage(original_image, kernelSize);
        imwrite("dilated_image.png", dilated_image);
        imshow("Original Image", original_image);
        imshow("Dilated Image", dilated_image);
        waitKey(0);
        break;
    }

    case 3: {
        Mat opened_image = openImage(original_image, kernelSize);
        imwrite("opened_image.png", opened_image);
        imshow("Original Image", original_image);
        imshow("Opened Image", opened_image);
        waitKey(0);
        break;
    }

    case 4: {
        Mat closed_image = closeImage(original_image, kernelSize);
        imwrite("closed_image.png", closed_image);
        imshow("Original Image", original_image);
        imshow("Closed Image", closed_image);
        waitKey(0);
        break;
    }

    case 5: {
        Mat topHat_image = topHat(original_image, kernelSize);
        imwrite("TopHat_image.png", topHat_image);
        imshow("Original Image", original_image);
        imshow("TopHat Image", topHat_image);
        waitKey(0);
        break;
    }

    case 6: {
        Mat bottomHat_image = bottomHat(original_image, kernelSize);
        imwrite("BottomHat_image.png", bottomHat_image);
        imshow("Original Image", original_image);
        imshow("BottomHat Image", bottomHat_image);
        waitKey(0);
        break;
    }

    case 7: {   //45 (from testing; for cell image)
        Mat enhanced_image = enhance(original_image, kernelSize);
        imwrite("enhanced_image.png", enhanced_image);
        imshow("Original Image", original_image);
        imshow("Enhanced Image", enhanced_image);
        waitKey(0);
        break;
    }

    default: {
        cout << "Incorrect option choosed!!!\n";
        break;
    }
    }
    return 0;
}
