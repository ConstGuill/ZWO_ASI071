// analyse.cpp
#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <filesystem>
#include "ASI071.h"

cv::Mat convert_to_8bit(const cv::Mat &image);

std::pair<int, double> countPixelsAboveThreshold(const cv::Mat &image, const cv::Vec3f &circle, uint16_t threshold);

void csv_write(const std::string &filePath, const std::string &data, int code);

int main(int argc, char **argv)
{
/***************************************************************************************/
/*                                    Initialisation                                   */
/***************************************************************************************/

    int cameraIndex = 0;
    int width = ASI071_MAX_ROI_WIDTH;   // Set max width for the ROI
    int height = ASI071_MAX_ROI_HEIGHT; // Set max height for the ROI
    int bin = 1;
    int imageType = ASI_IMG_RAW16;
    int exposure = 1500;                // Exposure time in ms for the images discs
    int gain = 0;
    int capture_time = 2;               // 


    std::vector<int> exposures = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 500, 700, 900, 1000, 1100, 1300, 1500};

    cv::Scalar noise_mean, peak_mean;
    double minVal, maxVal, threshold;

    ASI071 ZWO_Cam(cameraIndex);

/***************************************************************************************/
/*                                    Configuration                                    */
/***************************************************************************************/

    if (ZWO_Cam.get_error_number() != ZWO_NO_ERROR)
    {
        std::cerr << "Camera initialization error." << std::endl;
        return 1;
    }
    ZWO_Cam.configure(width, height, bin, imageType);
    ZWO_Cam.set_camera_exposure(exposure);
    ZWO_Cam.set_camera_gain(gain);
    ZWO_Cam.run();
    std::this_thread::sleep_for(std::chrono::seconds(capture_time));
    ZWO_Cam.stop();

/***************************************************************************************/
/*                                     Calibration                                     */
/***************************************************************************************/

    cv::Mat image = ZWO_Cam.get_image().clone();

    cv::Mat extracted_square = ZWO_Cam.extractSquareRegion(image, image.cols / 4, image.rows / 2, 200);

    noise_mean = cv::mean(extracted_square);
    cv::minMaxLoc(extracted_square, &minVal, &threshold);

    // Conversion en niveaux de gris et floutage pour réduire le bruit
    cv::Mat grayImage;
    cv::GaussianBlur(image, grayImage, cv::Size(25, 25), 10, 10);

    // Binarisation de l'image
    cv::Mat bin_image;
    cv::threshold(convert_to_8bit(grayImage), bin_image, threshold, 255, cv::THRESH_OTSU);

    std::vector<cv::Vec3f> circles = ZWO_Cam.imageDiscConfig(image, threshold, 5);

    ZWO_Cam.set_circles(circles);

/***************************************************************************************/
/*                                       Capture                                       */
/***************************************************************************************/

    std::string csv_path = "../build/expo_ratio_filtred_" + std::to_string(exposures[0]) + "_" + std::to_string(exposures.back()) + ".csv";

    std::string expo = "Exposure Time";
    csv_write(csv_path, expo, 1);
    for (int exposure : exposures)
    {
        expo = expo + "," + std::to_string(exposure);
    }

    for (int i = 0; i < 3; i++)
    {

        std::string csv_noise = "Noise Level";
        std::string csv_peak = "Peak";
        std::string csv_ratio = "Contrast Ratio";
        std::string csv_stops = "Stops";
        std::string csv_peakSNR = "Peak SNR (dB)";
        std::string csv_max = "Max";
        std::string csv_ov_exp_pix = "Overexposed Pixels";
        std::string csv_ov_exp_pix_per = "Overexposed Percentage";
        for (int exposure : exposures)
        {
            ZWO_Cam.set_camera_exposure(exposure);
            ZWO_Cam.run();
            std::this_thread::sleep_for(std::chrono::seconds(capture_time));
            ZWO_Cam.stop();

            std::cout << "\n\n##############\texposure time : " << exposure << "\t##############\n"
                      << std::endl;

            cv::Mat image = ZWO_Cam.get_image().clone();

            cv::minMaxLoc(image, &minVal, &maxVal);

            cv::Vec3f center_circle = ZWO_Cam.get_circle(ZWO_CENTER_CIRCLE);

            peak_mean = ZWO_Cam.calculateMeanInsideCircle(image, center_circle);

            double noiseLevel = noise_mean[0];
            double peak = peak_mean[0];

            double contrast_ratio = peak / noiseLevel;
            double stops = std::log2(peak) - std::log10(noiseLevel);
            double peakSNR_dB = 20.0 * std::log10(peak / noiseLevel);

            auto [ov_exp_pix, ov_exp_pix_percentage] = countPixelsAboveThreshold(image, center_circle, 65527);

            std::cout << "Noise Level: " << noiseLevel << std::endl;
            std::cout << "Peak: " << peak << std::endl;
            std::cout << "Contrast Ratio: " << contrast_ratio << std::endl;
            std::cout << "Stops: " << stops << std::endl;
            std::cout << "Peak SNR (dB): " << peakSNR_dB << std::endl;
            std::cout << "Max: " << maxVal << std::endl;
            std::cout << "Overexposed Pixels: " << ov_exp_pix << std::endl;
            std::cout << "Overexposed Percentage: " << ov_exp_pix_percentage << std::endl;

            std::cout << "\n################################################\n"<< std::endl;

            csv_noise += "," + std::to_string(noiseLevel);
            csv_peak += "," + std::to_string(peak);
            csv_ratio += "," + std::to_string(contrast_ratio);
            csv_stops += "," + std::to_string(stops);
            csv_peakSNR += "," + std::to_string(peakSNR_dB);
            csv_max += "," + std::to_string(maxVal);
            csv_ov_exp_pix += "," + std::to_string(ov_exp_pix);
            csv_ov_exp_pix_per += "," + std::to_string(ov_exp_pix_percentage);

            ZWO_Cam.flush_queue();
        }
        csv_write(csv_path, expo);
        csv_write(csv_path, csv_noise);
        csv_write(csv_path, csv_peak);
        csv_write(csv_path, csv_ratio);
        csv_write(csv_path, csv_stops);
        csv_write(csv_path, csv_peakSNR);
        csv_write(csv_path, csv_max);
        csv_write(csv_path, csv_ov_exp_pix);
        csv_write(csv_path, csv_ov_exp_pix_per);
        csv_write(csv_path, "\n");
        cv::imshow("image", image);
        cv::waitKey(0);
    }
    ZWO_Cam.close();
}

// Fonction pour convertir une image en format 8 bits par canal
cv::Mat convert_to_8bit(const cv::Mat &image)
{
    cv::Mat result;
    if (image.depth() == CV_8U)
    {
        // L'image est déjà en 8 bits
        return image;
    }
    else if (image.depth() == CV_16U)
    {
        // Convertir une image en 16 bits en 8 bits
        double scale = 255.0 / 65535.0; // Convertir 16 bits en 8 bits
        image.convertTo(result, CV_8U, scale);
    }
    else if (image.depth() == CV_32F)
    {
        // Convertir une image en virgule flottante en 8 bits
        image.convertTo(result, CV_8U, 255.0);
    }
    else
    {
        throw std::runtime_error("Unsupported image depth for video writing");
    }
    return result;
}

std::pair<int, double> countPixelsAboveThreshold(const cv::Mat &image, const cv::Vec3f &circle, uint16_t threshold)
{
    // Extraire les coordonnées du centre et le rayon du cercle
    int x = cvRound(circle[0]);
    int y = cvRound(circle[1]);
    int radius = cvRound(circle[2]);

    // Créer un masque circulaire
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::circle(mask, cv::Point(x, y), radius, cv::Scalar(255), cv::FILLED);

    // Créer une image de seuil avec le même type que l'image originale
    cv::Mat threshImage;
    cv::threshold(image, threshImage, threshold, 65535, cv::THRESH_BINARY);

    // Appliquer le masque
    cv::Mat maskedImage;
    threshImage.copyTo(maskedImage, mask);

    // Compter le nombre de pixels non nuls (qui correspondent à des pixels supérieurs au seuil et dans le masque)
    int countAboveThreshold = cv::countNonZero(maskedImage);

    // Compter le nombre total de pixels dans le masque
    int countTotalPixelsInMask = cv::countNonZero(mask);

    // Calculer le pourcentage de pixels au-dessus du seuil par rapport au total dans le masque
    double percentage = (countTotalPixelsInMask > 0) ? (static_cast<double>(countAboveThreshold) / countTotalPixelsInMask) * 100.0 : 0.0;

    return {countAboveThreshold, percentage};
}

void csv_write(const std::string &filePath, const std::string &data, int code = 0)
{
    std::ofstream file;

    if (code == 0)
    {
        // Ouvrir le fichier en mode append (ajout à la fin)
        file.open(filePath, std::ios::app);
        if (!file.is_open())
        {
            std::cerr << "Erreur : Impossible d'ouvrir le fichier." << std::endl;
            return;
        }
        file << data << "\n";
    }
    else if (code == 1)
    {
        // Ouvrir le fichier en mode écriture avec troncature (efface le contenu existant)
        file.open(filePath, std::ofstream::out | std::ofstream::trunc);
        if (!file.is_open())
        {
            std::cerr << "Erreur : Impossible d'ouvrir le fichier." << std::endl;
            return;
        }
    }

    // Fermer le fichier après l'opération
    file.close();
}