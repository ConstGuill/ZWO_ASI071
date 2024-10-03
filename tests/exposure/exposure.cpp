//capture.cpp
#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <filesystem>
#include "ASI071.h"

int main(int argc, char **argv)
{

    // Instanciation de la classe avec l'index de la caméra sélectionnée
    int cameraIndex = 0; // Index de la caméra, modifiez-le en fonction de votre configuration
    ASI071 ZWO_Cam(cameraIndex);
    if (ZWO_Cam.get_error_number() != ZWO_Cam_NO_ERROR)
    {
        std::cerr << "Camera initialization error." << std::endl;
        return 1;
    }

    // Configuration de la caméra
    int width = 0;
    int height = 0;
    int bin = 1;
    int imageType = ASI_IMG_RAW16;

    ZWO_Cam.configure(width, height, bin, imageType);

    // Définir l'exposition
    uint32_t exposureTime = 700; // Exposition en millisecondes
    if (ZWO_Cam.set_camera_exposure(exposureTime) != 0)
    {
        std::cerr << "Erreur lors de la définition de l'exposition." << std::endl;
        return 1;
    }

    // Démarrer la capture vidéo
    ZWO_Cam.run();

    // Attendre un moment pour capturer des images
    std::this_thread::sleep_for(std::chrono::seconds(5));

    exposureTime = 700; // Exposition en millisecondes
    ZWO_Cam.set_camera_exposure(exposureTime);

    // Attendre un moment pour capturer des images
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // Arrêter la capture vidéo
    ZWO_Cam.stop();

    // Obtenir les images capturées
    std::queue<cv::Mat> images = ZWO_Cam.flush_queue();
    std::cout << "Nombre d'images capturées : " << images.size() << std::endl;

#ifdef DEMOSAICING_TEST
    std::queue<cv::Mat> bayer_images = images;
    while (!bayer_images.empty())
    {
        ZWO_Cam.demosaicing(bayer_images.front());
        bayer_images.pop();
    }
#endif

#ifdef SAVING_TEST
    save_images_to_folder(images, "./output_img");
#endif

    std::queue<cv::Mat> f_images = filterQueue(images, 10);
}