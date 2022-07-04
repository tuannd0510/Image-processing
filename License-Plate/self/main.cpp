#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/text/ocr.hpp>
#include <tesseract/baseapi.h>
// #include <leptonica/allheaders.h>


int main(int argc, char *argv[] ){
    cv::Mat image;
    std::string pathsrc = "Gs5sd.jpg";
    image = cv::imread(pathsrc, cv::IMREAD_COLOR);
    if(! image.data){
        std::cout<<"Could not open file" << std::endl;
        return -1;
    } 
    cv::imshow("img", image);
    cv::waitKey(0);

    /** Step 1: Resize the image to the required size and then grayscale it. 
     * img = cv2.resize(img, (620,480) )
     * gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
     */
    cv::Mat gray;
    resize(image,image, cv::Size(620,480));
    cv::imshow("resize", image);
    cv::waitKey(0);
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::imshow("grayscale", gray);
    cv::waitKey(0);
    
    /** Step 2: remove the unwanted details from an image. 
     * gray = cv2.bilateralFilter(gray, 15, 80, 80, cv2.BORDER_DEFAULT) 
     */
    // void bilateralFilter(cv::InputArray src, cv::OutputArray dst, int d, double sigmaColor, double sigmaSpace, int borderType = 4)
    cv::Mat bluring;
    cv::bilateralFilter(gray, bluring, 15,80,80, cv::BORDER_DEFAULT);
    cv::imshow("Bluring", bluring);
    cv::waitKey(0);    

    /** Step 3: Phat hien canh
     * edged = cv2.Canny(gray, 30, 200)
     */
    cv::Mat edged;
    cv::Canny(gray, edged, 100, 200);
    cv::imshow("edged 100 200", edged);
    cv::waitKey(0);

    /** Step 4: looking for contours on our image
     * cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     * cnts = imutils.grab_contours(cnts)
     * cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
     * screenCnt = None
     * # loop over our contours
     * for c in cnts:
     * # approximate the contour
     *  peri = cv2.arcLength(c, True)
     *  approx = cv2.approxPolyDP(c, 0.018 * peri, True,)
     * # if our approximated contour has four points, then
     * # we can assume that we have found our screen
     *  if len(approx) == 4:
     *      screenCnt = approx
     *      break
     */

    cv::threshold(edged, edged, 128, 255, cv::THRESH_BINARY);
    cv::imshow("edged", edged);
    cv::waitKey(0);
    

    /** Step 5: proceed with masking the entire picture except for the place where the number plate is
     * mask = np.zeros(gray.shape,np.uint8,)
     * new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
     * new_image = cv2.bitwise_and(img,img,mask=mask)
     */


    /** Step 6: crop
     * (x, y) = np.where(mask == 255)
     * print(type((x,y)))
     * (topx, topy) = (np.min(x), np.min(y))
     * (bottomx, bottomy) = (np.max(x), np.max(y))
     * Cropped = gray[topx:bottomx+1, topy:bottomy+1]
     */

    /** Step 7: Read the number plate use tesseract
     * text = pytesseract.image_to_string(Cropped, config='--psm 11')
     * print("Detected Number is:",text)
     */
    std::string numberplate;

    std::string outText, imPath = argv[1];
    cv::Mat im = cv::imread(imPath, cv::IMREAD_COLOR);
    tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();

    api->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
    api->SetPageSegMode(tesseract::PSM_AUTO);
    api->SetImage(im.data, im.cols, im.rows, 3, im.step);
    outText = std::string(api->GetUTF8Text());
    std::cout << outText;
    api->End();
    // std::cout << numberplate << std::endl;

    return 0;
}   