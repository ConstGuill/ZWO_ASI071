// capture.cpp
#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <filesystem>
#include "ASI071.h"
#include <getopt.h>
#include <opencv2/opencv.hpp>

// Fonction de sauvegarde des images
void save_images_to_folder(const std::queue<cv::Mat> &images, const std::string &folder_path) {
    std::filesystem::create_directories(folder_path);
    int img_num = 0;
    int total_images = images.size();
    std::queue<cv::Mat> imgs = images;
    
    int step = 1;
    int displayNext = step;
    int percent = 0;

    std::cout << "Processing " << total_images << " images..." << std::endl;

    while (!imgs.empty()) {
        std::string filename = folder_path + "/image_" + std::to_string(img_num) + ".png";
        cv::imwrite(filename, imgs.front());
        imgs.pop();
        img_num++;

        // Calculer et afficher la barre de progression
        percent = (100 * img_num) / total_images;
        if (percent >= displayNext) {
            std::cout << "\r[" << std::string(percent / 5, (char)254u) << std::string(20 - percent / 5, ' ') << "]";
            std::cout << percent << "% [" << img_num << " of " << total_images << "]";
            std::cout.flush();
            displayNext += step;
        }
    }

    // Imprimer une nouvelle ligne après la fin de la sauvegarde
    std::cout << std::endl;
}

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
    uint32_t exposure = 50; // Exposure time in ms for the images discs
    int gain = 20;
    int capture_time = 2;

    bool demosaicing_test = false;
    std::string output_folder_path;

    // Options de ligne de commande
    static struct option long_options[] = {
        {"demosaicing", no_argument, 0, 'd'},
        {"output", required_argument, 0, 'o'},
        {"cameraIndex", required_argument, 0, 'i'},
        {"width", required_argument, 0, 'w'},
        {"height", required_argument, 0, 'h'},
        {"bin", required_argument, 0, 'b'},
        {"imageType", required_argument, 0, 't'},
        {"exposure", required_argument, 0, 'e'},
        {"gain", required_argument, 0, 'g'},
        {"capture_time", required_argument, 0, 'c'},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "do:i:w:h:b:t:e:g:c:", long_options, &option_index)) != -1) {
        switch (c) {
            case 'd':
                demosaicing_test = true;
                break;
            case 'o':
                output_folder_path = optarg;
                break;
            case 'i':
                cameraIndex = std::stoi(optarg);
                break;
            case 'w':
                width = std::stoi(optarg);
                break;
            case 'h':
                height = std::stoi(optarg);
                break;
            case 'b':
                bin = std::stoi(optarg);
                break;
            case 't':
                imageType = std::stoi(optarg);
                break;
            case 'e':
                exposure = std::stoi(optarg);
                break;
            case 'g':
                gain = std::stoi(optarg);
                break;
            case 'c':
                capture_time = std::stoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [options]\n"
                          << "Options:\n"
                          << "  --demosaicing\n"
                          << "  --output <folder_path>\n"
                          << "  --cameraIndex <index>\n"
                          << "  --width <width>\n"
                          << "  --height <height>\n"
                          << "  --bin <bin>\n"
                          << "  --imageType <type>\n"
                          << "  --exposure <exposure_ms>\n"
                          << "  --gain <gain>\n"
                          << "  --capture_time <time_s>\n";
                return 1;
        }
    }

    ASI071 ZWO_Cam(cameraIndex);
    if (ZWO_Cam.get_error_number() != ZWO_NO_ERROR)
    {
        std::cerr << "Camera initialization error." << std::endl;
        return 1;
    }

    ZWO_Cam.configure(width, height, bin, imageType);

    // Définir l'exposition
    if (ZWO_Cam.set_camera_exposure(exposure) != 0)
    {
        std::cerr << "Erreur lors de la définition de l'exposition." << std::endl;
        return 1;
    }

    // Démarrer la capture vidéo
    ZWO_Cam.run();

    // Attendre un moment pour capturer des images
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // Arrêter la capture vidéo
    ZWO_Cam.stop();

    // Obtenir les images capturées
    std::queue<cv::Mat> images = ZWO_Cam.flush_queue();
    std::cout << "Nombre d'images capturées : " << images.size() << std::endl;

    if (demosaicing_test)
    {
        std::queue<cv::Mat> bayer_images;
        while (!images.empty())
        {
            cv::Mat demosaiced_image = ZWO_Cam.demosaicing(images.front());
            bayer_images.push(demosaiced_image);
            images.pop();
        }
        images = bayer_images;
    }

    if (!output_folder_path.empty())
    {
        save_images_to_folder(images, output_folder_path);
    }
    ZWO_Cam.close();
    return 0;
}
