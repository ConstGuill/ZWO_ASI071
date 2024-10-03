//ASI071.cpp
#include "ASI071.h"

// Constructor: Initializes the camera and creates an image buffer.
ASI071::ASI071(int selectedCamIndex)
{
    try
    {
        initializeCamera(selectedCamIndex);
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << e.what() << std::endl;
        // Handle initialization error and rethrow if necessary
        throw;
    }

    createImage();
}

// Destructor: Ensures proper cleanup by closing the camera.
ASI071::~ASI071()
{
    if (pImage != NULL)
    {
        cvReleaseImage(&pImage);
    }
    close(); // Close camera and release resources
}

void ASI071::initializeCamera(int selectedCamIndex)
{
    Error_Number = ZWO_NO_ERROR;

    // Get the number of connected cameras
    int numDevices = ASIGetNumOfConnectedCameras();
    if (numDevices <= 0)
    {
        Error_Number = ZWO_NO_CAMERA;
        throw std::runtime_error("No camera connected.");
    }

    // Validate the selected camera index
    if (selectedCamIndex < 0 || selectedCamIndex >= numDevices)
    {
        Error_Number = ZWO_WRONG_INDEX;
        throw std::runtime_error("Invalid camera index selected.");
    }

    // Set the selected camera index
    CamIndex = selectedCamIndex;

    // Retrieve camera properties
    check_status(ASIGetCameraProperty(&CamInfo, CamIndex), "Failed to get camera properties.");

    // Open and initialize the camera
    check_status(ASIOpenCamera(CamInfo.CameraID) || ASIInitCamera(CamInfo.CameraID), "OpenCamera error. Are you root?");
}

void ASI071::configure(int width, int height, int bin, int image_type)
{
    // Validate width and height
    if (width < 0 || height < 0)
    {
        throw std::invalid_argument("Width and height must be positive values.");
    }

    set_camera_exposure(100);
    set_camera_gain(0);

    set_dimensions(width, height);
    set_binning(bin);
    set_image_type(image_type);

    createImage();

    // Set ROI format
    check_status(ASISetROIFormat(CamInfo.CameraID, ROI_Width, ROI_Height, ROI_Bin, static_cast<ASI_IMG_TYPE>(ROI_Image_type)), "Failed to set ROI format.");

    // Set control values for exposure, gain, bandwidth overload, high speed mode, and white balance
    check_status(ASISetControlValue(CamInfo.CameraID, ASI_EXPOSURE, exposure * 1000, ASI_FALSE), "Failed to set exposure control value.");
    check_status(ASISetControlValue(CamInfo.CameraID, ASI_GAIN, gain, ASI_FALSE), "Failed to set gain control value.");
    check_status(ASISetControlValue(CamInfo.CameraID, ASI_BANDWIDTHOVERLOAD, 100, ASI_FALSE), "Failed to set bandwidth overload control value.");
    check_status(ASISetControlValue(CamInfo.CameraID, ASI_HIGH_SPEED_MODE, 0, ASI_FALSE), "Failed to set high speed mode control value.");
    check_status(ASISetControlValue(CamInfo.CameraID, ASI_WB_B, 90, ASI_FALSE), "Failed to set white balance blue control value.");
    check_status(ASISetControlValue(CamInfo.CameraID, ASI_WB_R, 48, ASI_TRUE), "Failed to set white balance red control value.");
}

void ASI071::run()
{
    // Start video capture
    check_status(ASIStartVideoCapture(CamInfo.CameraID), "Error starting video capture");

#ifdef _LIN
    // Create a capture thread for Linux
    if (pthread_create(&capture_thread, NULL, capture_helper, this) != 0)
    {
        std::cerr << "Error creating capture thread" << std::endl;
    }
#elif defined _WINDOWS
    // Create a capture thread for Windows
    capture_thread = (HANDLE)_beginthreadex(NULL, 0, capture_helper, this, 0, NULL);
    if (capture_thread == NULL)
    {
        std::cerr << "Error creating capture thread: " << GetLastError() << std::endl;
    }
#endif
}

void ASI071::stop()
{
    stop_capture = true;

#ifdef _LIN
    void *retval;
    pthread_join(capture_thread, &retval);
    stop_capture = false;
#elif defined _WINDOWS
    Sleep(50); // Allow some time for thread completion
#endif

    // Stop video capture
    check_status(ASIStopVideoCapture(CamInfo.CameraID), "Error stopping video capture");
}

std::queue<cv::Mat> ASI071::flush_queue()
{
    std::queue<cv::Mat> flushed_queue;

    // Empty the image queue and collect images
    while (!image_queue.empty())
    {
        // Create a copy of the image
        cv::Mat img = image_queue.front();

#ifdef ZWO_DISPLAY_CAPTURE
        // Display the captured image if ZWO_DISPLAY_CAPTURE is defined
        cv::imshow("Flushed Image", img);
        cv::waitKey(1); // Wait for a short time to display the image
#endif
        flushed_queue.push(img);
        image_queue.pop();
    }

    return flushed_queue;
}

bool ASI071::is_queue_empty()
{
    return image_queue.empty();
}

cv::Mat ASI071::demosaicing(cv::Mat image)
{
    cv::Mat output_image;
    // debayering images
    cv::demosaicing(image, output_image, cv::COLOR_BayerBG2BGR);

    return output_image;
}

void ASI071::close()
{
    stop(); // Stop capturing before closing

    // Close the camera
    check_status(ASICloseCamera(CamInfo.CameraID), "Error closing camera");

    // Flush any remaining images in the queue
    flush_queue();
}

void ASI071::capture()
{
    auto start_time = std::chrono::steady_clock::now();
    int count = 0;
    int iDropFrame = 0;

    // Get the camera mode once
    ASI_CAMERA_MODE mode;
    check_status(ASIGetCameraMode(CamInfo.CameraID, &mode), "Error getting camera mode");
    int wait_ms = (mode == ASI_MODE_NORMAL) ? 500 : 1000;

    while (!stop_capture)
    {
        // Capture image data
        if (ASIGetVideoData(CamInfo.CameraID, (unsigned char *)pImage->imageData, pImage->imageSize, wait_ms) == ASI_SUCCESS)
        {
            count++;
            // Convert IplImage* to cv::Mat
            cv::Mat img = cv::cvarrToMat(pImage).clone();
            image_queue.push(img);

#ifdef ZWO_DISPLAY_CAPTURE
            // Display the captured image if ZWO_DISPLAY_CAPTURE is defined
            cv::imshow("Captured Image", image_queue.back());
            cv::waitKey(1); // Wait for a short time to display the image
#endif
        }
        else
        {
            std::cerr << "Error capturing video data." << std::endl;
        }

        // Check elapsed time every second
        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - start_time;
        if (elapsed_seconds.count() > 1.0)
        {
            // Get and print dropped frames
            check_status(ASIGetDroppedFrames(CamInfo.CameraID, &iDropFrame), "Error getting dropped frames");
            std::cout << "fps: " << count << "\tdropped frames: " << iDropFrame << std::endl;

            // Reset counters and timer
            count = 0;
            start_time = current_time;
        }
    }
}

/*************************************** Getters ***************************************/

// Returns the error number
int ASI071::get_error_number()
{
    return Error_Number;
}

// Returns the current exposure time
uint32_t ASI071::get_exposure() const
{
    return exposure;
}

// Returns the current exposure time
uint32_t ASI071::get_gain() const
{
    return gain;
}

cv::Mat ASI071::get_image()
{
    if (image_queue.empty())
    {
        throw std::runtime_error("Image queue is empty.");
    }

    cv::Mat image = image_queue.front().clone();
    image_queue.pop();
    return image;
}

cv::Vec3f ASI071::get_circle(int index) const
{
    if (index >= 0 && index < circles.size())
    {
        return circles[index];
    }
    else
    {
        throw std::out_of_range("Index out of range");
    }
}

std::vector<cv::Vec3f> ASI071::get_circles() const
{
    return circles;
}

/*************************************** Setters ***************************************/

// Sets the dimensions of the image ROI
void ASI071::set_dimensions(int width, int height)
{
    if (width <= 0 || height <= 0)
    {
        ROI_Width = CamInfo.MaxWidth;
        ROI_Height = CamInfo.MaxHeight;
    }
    else
    {
        ROI_Width = width;
        ROI_Height = height;
    }
}

// Sets the binning factor for the image
void ASI071::set_binning(int bin)
{
    if (bin <= 0)
    {
        throw std::invalid_argument("Binning must be a positive value.");
    }
    ROI_Bin = bin;
}

void ASI071::set_image_type(int image_type)
{
    if (image_type < 0)
    {
        throw std::invalid_argument("Image type must be a non-negative value.");
    }
    ROI_Image_type = image_type;
}

void ASI071::set_exposure(uint32_t exposure_ms)
{
    if (exposure_ms == 0)
    {
        throw std::invalid_argument("Exposure time must be a positive value.");
    }
    exposure = exposure_ms * 1000;
}

void ASI071::set_gain(uint32_t gain_x10)
{
    if (gain_x10 < 0)
    {
        throw std::invalid_argument("Gain time must be a positive value.");
    }
    gain = gain_x10;
}

int ASI071::set_camera_exposure(uint32_t exposure_ms)
{
    set_exposure(exposure_ms);
    check_status(ASISetControlValue(CamInfo.CameraID, ASI_EXPOSURE, exposure_ms * 1000, ASI_FALSE), "Error setting exposure");

    return 0;
}

int ASI071::set_camera_gain(uint32_t gain_x10)
{
    set_gain(gain_x10);
    check_status(ASISetControlValue(CamInfo.CameraID, ASI_GAIN, gain, ASI_FALSE), "Error setting gain");

    return 0;
}

void ASI071::set_circle(int index, const cv::Vec3f &value)
{
    if (index >= 0 && index < circles.size())
    {
        circles[index] = value;
    }
    else
    {
        throw std::out_of_range("Index out of range");
    }
}

void ASI071::set_circles(const std::vector<cv::Vec3f> &newCircles)
{
    circles = newCircles;
}

/*************************************** Helpers ***************************************/

// Checks the status and prints an error message if there is an error
int ASI071::check_status(int status, const std::string &error_message)
{
    if (status != ASI_SUCCESS)
    {
        std::cerr << error_message << "\n\tError code: " << status << std::endl;
        throw std::runtime_error(error_message + " Error code: " + std::to_string(status));
    }
    return status;
}

// Creates an image buffer based on the image type
void ASI071::createImage()
{
    if (ROI_Image_type == ASI_IMG_RAW16)
        pImage = cvCreateImage(cvSize(CamInfo.MaxWidth, CamInfo.MaxHeight), IPL_DEPTH_16U, 1);
    else if (ROI_Image_type == ASI_IMG_RGB24)
        pImage = cvCreateImage(cvSize(CamInfo.MaxWidth, CamInfo.MaxHeight), IPL_DEPTH_8U, 3);
    else
        pImage = cvCreateImage(cvSize(CamInfo.MaxWidth, CamInfo.MaxHeight), IPL_DEPTH_8U, 1);

    if (pImage == NULL)
    {
        throw std::runtime_error("Failed to create image buffer.");
    }
}

cv::Mat ASI071::extractSquareRegion(const cv::Mat &image, int x, int y, int n)
{
    // Dimensions de l'image
    int h = image.rows;
    int w = image.cols;

    // Calculer les limites de la région à extraire
    int half_n = n / 2;
    int x_min = std::max(0, x - half_n);
    int x_max = std::min(w, x + half_n + 1);
    int y_min = std::max(0, y - half_n);
    int y_max = std::min(h, y + half_n + 1);

    // Extraire la région carrée
    cv::Rect roi(x_min, y_min, x_max - x_min, y_max - y_min);
    cv::Mat squareRegion = image(roi);

    // Si la région extraite n'est pas exactement n x n, compléter avec des zéros
    if (squareRegion.rows < n || squareRegion.cols < n)
    {
        cv::Mat borderedRegion;
        int top = std::max(0, half_n - y_min);
        int bottom = std::max(0, (y + half_n + 1) - y_max);
        int left = std::max(0, half_n - x_min);
        int right = std::max(0, (x + half_n + 1) - x_max);

        cv::copyMakeBorder(squareRegion, borderedRegion, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        return borderedRegion;
    }

    return squareRegion;
}

// Fonction de comparaison pour trier les cercles
bool ASI071::compareCircles(const cv::Vec3f &a, const cv::Vec3f &b)
{
    if (a[0] == b[0])
    {
        // Si les coordonnées x sont égales, trier par y
        return a[1] < b[1];
    }
    // Sinon, trier par x
    return a[0] < b[0];
}

std::vector<cv::Vec3f> ASI071::detectCircles(const cv::Mat &image)
{
    // Vérifier que l'image est en niveaux de gris
    cv::Mat grayImage;
    if (image.channels() == 3)
    {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    }
    else
    {
        grayImage = image;
    }

    // Détection des cercles avec la transformation de Hough
    std::vector<cv::Vec3f> circles;

    // Détection des cercles
    cv::HoughCircles(
        image,              // Image source
        circles,            // Vecteur pour les cercles détectés
        cv::HOUGH_GRADIENT, // Méthode de détection
        1,                  // Résolution de l'accumulateur
        1,                  // Distance minimale entre les centres des cercles
        100,                // Seuil pour la détection du gradient
        10,                 // Seuil pour la détection des centres
        image.rows / 10,    // Rayon minimum des cercles
        image.rows / 5      // Rayon maximum des cercles
    );

    if (circles.size() <= 0)
    {
        std::cerr << "Erreur : Aucun cercles detectes." << std::endl;
    }
    else
    {
        circles = removePartialCircles(image, circles);
    }

    return circles;
}

std::vector<std::vector<cv::Vec3f>> ASI071::clusterCircles(const std::vector<cv::Vec3f> &circles, int n)
{
    // Matrice de points (chaque point représente le centre d'un cercle)
    cv::Mat points(circles.size(), 2, CV_32F);

    for (size_t i = 0; i < circles.size(); i++)
    {
        points.at<float>(i, 0) = circles[i][0]; // Coordonnée x du centre
        points.at<float>(i, 1) = circles[i][1]; // Coordonnée y du centre
    }

    // Vecteur pour stocker les labels des clusters
    cv::Mat labels;
    // Matrice pour stocker les centres des clusters
    cv::Mat centers;

    // Appliquer K-means clustering
    cv::kmeans(points, n, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

    // Initialiser les groupes
    std::vector<std::vector<cv::Vec3f>> clusters(n);

    // Distribuer les cercles dans les groupes en fonction des labels obtenus
    for (size_t i = 0; i < circles.size(); i++)
    {
        int clusterIdx = labels.at<int>(i);
        clusters[clusterIdx].push_back(circles[i]);
    }

    return clusters;
}

std::vector<cv::Vec3f> ASI071::removePartialCircles(const cv::Mat &image, const std::vector<cv::Vec3f> &circles)
{
    std::vector<cv::Vec3f> filteredCircles;

    int imgWidth = image.cols;
    int imgHeight = image.rows;

    for (const auto &circle : circles)
    {
        float x = circle[0];
        float y = circle[1];
        float r = circle[2];

        // Vérifier si le cercle est entièrement contenu dans l'image
        if (x - r >= 0 && y - r >= 0 && x + r <= imgWidth && y + r <= imgHeight)
        {
            filteredCircles.push_back(circle);
        }
    }

    return filteredCircles;
}

// Fonction pour calculer la distance euclidienne entre deux points
double calculateDistance(const cv::Point2f &point1, const cv::Point2f &point2)
{
    return std::sqrt(std::pow(point1.x - point2.x, 2) + std::pow(point1.y - point2.y, 2));
}

cv::Point2f ASI071::calculateMeanCenter(const std::vector<cv::Vec3f> &circles)
{
    float sumX = 0;
    float sumY = 0;

    for (const auto &circle : circles)
    {
        sumX += circle[0];
        sumY += circle[1];
    }

    return cv::Point2f(sumX / circles.size(), sumY / circles.size());
}

// Fonction pour supprimer n% des cercles les plus éloignés du centre moyen
std::vector<cv::Vec3f> ASI071::removeFarCircles(const std::vector<cv::Vec3f> &circles, float percentage)
{
    // Calculer le centre moyen
    cv::Point2f meanCenter = calculateMeanCenter(circles);

    // Calculer les distances de chaque cercle au centre moyen
    std::vector<std::pair<double, cv::Vec3f>> distanceCirclePairs;
    for (const auto &circle : circles)
    {
        double distance = calculateDistance(meanCenter, cv::Point2f(circle[0], circle[1]));
        distanceCirclePairs.push_back(std::make_pair(distance, circle));
    }

    // Trier les cercles par distance décroissante
    std::sort(distanceCirclePairs.begin(), distanceCirclePairs.end(), [](const auto &a, const auto &b)
              { return a.first > b.first; });

    // Calculer le nombre de cercles à supprimer
    size_t numToRemove = static_cast<size_t>(percentage * circles.size() / 100.0);

    // Garder seulement les cercles les plus proches du centre moyen
    std::vector<cv::Vec3f> filteredCircles;
    for (size_t i = numToRemove; i < distanceCirclePairs.size(); i++)
    {
        filteredCircles.push_back(distanceCirclePairs[i].second);
    }

    return filteredCircles;
}

cv::Vec3f ASI071::findLargestCircle(const std::vector<cv::Vec3f> &circles)
{
    if (circles.empty())
    {
        throw std::runtime_error("La liste des cercles est vide.");
    }

    cv::Vec3f largestCircle = circles[0];

    for (const auto &circle : circles)
    {
        if (circle[2] > largestCircle[2])
        {
            largestCircle = circle;
        }
    }

    return largestCircle;
}

// Fonction pour trier les cercles
void ASI071::sortCircles(std::vector<cv::Vec3f> &circles)
{
    std::sort(circles.begin(), circles.end(), [this](const cv::Vec3f &a, const cv::Vec3f &b)
              { return this->compareCircles(a, b); });
}

cv::Vec3f ASI071::filterCircles(cv::Mat image, std::vector<cv::Vec3f> circles)
{
    // Supprimer les cercles qui ne sont pas completement inscrit dans l'image
    std::vector<cv::Vec3f> circles_set = removePartialCircles(image, circles);

    // Supprimer les cercles les plus éloignés du centre moyen
    circles_set = removeFarCircles(circles_set, 99.0);

    // Garder le cercle le plus grand
    cv::Vec3f largestCircle = findLargestCircle(circles_set);

    return largestCircle;
}

std::vector<cv::Vec3f> ASI071::imageDiscConfig(cv::Mat image, int threshold, int lens_numbers)
{
    // Conversion en niveaux de gris et floutage pour réduire le bruit
    cv::Mat grayImage;
    cv::GaussianBlur(image, grayImage, cv::Size(25, 25), 10, 10);

    // Binarisation de l'image
    cv::Mat bin_image;
    cv::threshold(convert_to_8bit(grayImage), bin_image, threshold, 255, cv::THRESH_OTSU);

    // Détection des cercles dans l'image binaire
    std::vector<cv::Vec3f> circles = detectCircles(bin_image);

    // Clustering des cercles détectés en fonction du nombre de lentilles
    std::vector<std::vector<cv::Vec3f>> clusters = clusterCircles(circles, lens_numbers);

    // Filtrage des cercles pour sélectionner le meilleur cercle de chaque cluster
    std::vector<cv::Vec3f> detected_circles;
    for (size_t i = 0; i < clusters.size(); i++)
    {
        detected_circles.push_back(filterCircles(image, clusters[i]));
    }

    // Tri des cercles détectés pour garantir un ordre spécifique
    sortCircles(circles);

    return detected_circles;
}

cv::Mat ASI071::convert_to_8bit(const cv::Mat &image)
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

cv::Scalar ASI071::calculateMeanInsideCircle(const cv::Mat &image, cv::Vec3f circle)
{
    // Extraire les coordonnées du centre et le rayon du cercle
    int x = cvRound(circle[0]);
    int y = cvRound(circle[1]);
    int radius = cvRound(circle[2]);

    // Créer un masque circulaire
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::circle(mask, cv::Point(x, y), radius, cv::Scalar(255), cv::FILLED);

    // Calculer la moyenne des pixels à l'intérieur du cercle
    cv::Scalar mean_value = cv::mean(image, mask);

    return mean_value;
}

// Returns the current tick count in milliseconds
unsigned long ASI071::GetTickCount()
{
#ifdef _MAC
    struct timeval now;
    gettimeofday(&now, NULL);
    unsigned long ul_ms = now.tv_usec / 1000 + now.tv_sec * 1000;
    return ul_ms;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / (1000 * 1000));
#endif
}

#ifdef _LIN
// Capture helper function for Linux
void *ASI071::capture_helper(void *arg)
{
    ASI071 *zwo = reinterpret_cast<ASI071 *>(arg);
    zwo->capture();
    return NULL;
}
#elif defined _WINDOWS
// Capture helper function for Windows
unsigned __stdcall ASI071::capture_helper(void *arg)
{
    ASI071 *zwo = reinterpret_cast<ASI071 *>(arg);
    zwo->capture();
    return 0;
}
#endif
