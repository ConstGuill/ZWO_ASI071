#ifndef ASI071_H
#define ASI071_H

#include "string.h"
#include <stdexcept>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core_c.h"
#include "opencv2/videoio/legacy/constants_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "ASICamera2.h"

#define _LIN
// #define ZWO_DISPLAY_CAPTURE

#define ASI071_MAX_ROI_WIDTH 4944
#define ASI071_MAX_ROI_HEIGHT 3284

typedef enum
{
    ZWO_NO_ERROR = 0, // format_change_imagetype
    ZWO_NO_CAMERA,    // format_change_bin
    ZWO_WRONG_INDEX,  // format_change_size_bigger
} ASI071_Error;

typedef enum
{
    ZWO_TOP_LEFT_CIRCLE = 0,
    ZWO_TOP_RIGHT_CIRCLE,
    ZWO_CENTER_CIRCLE,
    ZWO_BOTTOM_LEFT_CIRCLE,
    ZWO_BOTTOM_RIGHT_CIRCLE

} ZWO_CIRCLE_CODE;

class ASI071
{
public:
    /**
     * @brief Constructs a ASI071 object and initializes the camera.
     * @param selectedCamIndex Index of the camera to be selected.
     * @throws std::runtime_error if there is an error initializing the camera.
     */
    ASI071(int selectedCamIndex);

    /**
     * @brief Destroys the ASI071 object and closes the camera.
     */
    ~ASI071();

    /**
     * @brief Configures the camera settings.
     * @param width Width of the image.
     * @param height Height of the image.
     * @param bin Binning factor for the image.
     * @param image_type Type of the image to be captured.
     * @throws std::invalid_argument if width or height is negative.
     */
    void configure(int width, int height, int bin, int image_type);

    /**
     * @brief Starts video capture and creates a capture thread.
     */
    void run();

    /**
     * @brief Stops video capture and joins the capture thread.
     */
    void stop();

    /**
     * @brief Empties the image queue and returns all images.
     * @return A vector of images that were in the queue.
     */
    std::queue<cv::Mat> flush_queue();

    bool is_queue_empty();

    cv::Mat demosaicing(cv::Mat image);

    /**
     * @brief Closes the camera and flushes the image queue.
     */
    void close();

    /**
     * @brief Gets the current error number.
     * @return The current error number.
     */
    int get_error_number();

    /**
     * @brief Gets the current exposure time.
     * @return The current exposure time in milliseconds.
     */
    uint32_t get_exposure() const;

    /**
     * @brief Gets the current exposure time.
     * @return The current exposure time in milliseconds.
     */
    uint32_t get_gain() const;

    /**
     * @brief Gets an image from the queue.
     * @return The image at the front of the queue.
     * @throws std::runtime_error if the image queue is empty.
     */
    cv::Mat get_image();

    cv::Vec3f get_circle(int index) const;

    std::vector<cv::Vec3f> get_circles() const;

    /**
     * @brief Sets the dimensions of the image ROI.
     * @param width Width of the image ROI.
     * @param height Height of the image ROI.
     * @throws std::invalid_argument if width or height is not positive.
     */
    void set_dimensions(int width, int height);

    /**
     * @brief Sets the binning factor for the image.
     * @param bin Binning factor.
     * @throws std::invalid_argument if bin is not positive.
     */
    void set_binning(int bin);

    /**
     * @brief Sets the image type for capturing.
     * @param image_type Image type.
     * @throws std::invalid_argument if image_type is negative.
     */
    void set_image_type(int image_type);

    /**
     * @brief Sets the exposure time for capturing.
     * @param exposure_ms Exposure time in milliseconds.
     * @throws std::invalid_argument if exposure_ms is zero.
     */
    void set_exposure(uint32_t exposure_ms);

    /**
     * @brief Sets the exposure time for capturing.
     * @param gain_x10 Gain in 0.1dB.
     * @throws std::invalid_argument if exposure_ms is zero.
     */
    void set_gain(uint32_t gain_x10);

    /**
     * @brief Sets the exposure time for the camera and restarts video capture.
     * @param exposure_ms Exposure time in milliseconds.
     * @return Zero on success.
     */
    int set_camera_exposure(uint32_t exposure_ms);

    /**
     * @brief Sets the exposure time for the camera and restarts video capture.
     * @param gain_x10 Gain in 0.1dB.
     * @return Zero on success.
     */
    int set_camera_gain(uint32_t gain_x10);

    void set_circle(int index, const cv::Vec3f &value);

    void set_circles(const std::vector<cv::Vec3f> &newCircles);

    /**
     * @brief Creates an image buffer based on the image type.
     */
    void createImage();

    /**
     * @brief Extrait une région carrée de l'image autour d'un point donné, avec padding si nécessaire.
     *
     * Cette fonction extrait une région carrée de taille `n x n` centrée sur le point `(x, y)` dans l'image fournie.
     * Si la région demandée dépasse les limites de l'image, elle sera complétée avec des bordures noires pour atteindre
     * la taille spécifiée.
     *
     * @param image L'image source à partir de laquelle extraire la région (doit être en niveaux de gris ou couleur).
     * @param x La coordonnée x du centre de la région carrée à extraire.
     * @param y La coordonnée y du centre de la région carrée à extraire.
     * @param n La taille de la région carrée à extraire. La région résultante aura une taille de `n x n`.
     * @return cv::Mat La région carrée extraite, complétée avec des bordures noires si nécessaire pour atteindre
     *         la taille spécifiée `n x n`.
     */
    cv::Mat extractSquareRegion(const cv::Mat &image, int x, int y, int n);

    // Fonction pour trier les cercles
    void sortCircles(std::vector<cv::Vec3f> &circles);

    cv::Point2f calculateMeanCenter(const std::vector<cv::Vec3f> &circles);

    std::vector<cv::Vec3f> imageDiscConfig(cv::Mat image, int threshold, int lens_numbers);

    cv::Scalar calculateMeanInsideCircle(const cv::Mat &image, cv::Vec3f circle);

    // Getter pour le thread de capture
#ifdef _LIN
    pthread_t get_capture_thread() const
    {
        return capture_thread;
    }
#elif defined _WINDOWS
    HANDLE get_capture_thread() const
    {
        return thread_setgainexp;
    }
#endif

#ifdef _LIN
    /**
     * @brief Capture helper function for Linux to use capture function in a thread.
     * @param arg Pointer to the ASI071 instance.
     * @return Null pointer.
     */
    static void *capture_helper(void *arg); // For Linux
#elif defined _WINDOWS
    /**
     * @brief Capture helper function for Windows to use capture function in a thread..
     * @param arg Pointer to the ASI071 instance.
     * @return Status code.
     */
    static unsigned __stdcall capture_helper(void *arg); // For Windows
#endif

protected:
    /**
     * @brief Captures video data and processes images.
     */
    void capture();

private:
    /**
     * @brief Initializes the camera with the given index.
     * @param selectedCamIndex Index of the camera to be initialized.
     * @throws std::runtime_error if there is an error during initialization.
     */
    void initializeCamera(int selectedCamIndex);

    /**
     * @brief Checks the status of an operation and prints an error message if there is an error.
     * @param status Status code of the operation.
     * @param error_message Error message to be printed if the status indicates failure.
     * @return The status code.
     */
    int check_status(int status, const std::string &error_message);

    /**
     * @brief Returns the current tick count in milliseconds.
     * @return The current tick count in milliseconds.
     */
    unsigned long GetTickCount();

    bool compareCircles(const cv::Vec3f &a, const cv::Vec3f &b);

    std::vector<cv::Vec3f> detectCircles(const cv::Mat &image);

    std::vector<std::vector<cv::Vec3f>> clusterCircles(const std::vector<cv::Vec3f> &circles, int n);

    cv::Vec3f filterCircles(cv::Mat image, std::vector<cv::Vec3f> circles);

    std::vector<cv::Vec3f> removePartialCircles(const cv::Mat &image, const std::vector<cv::Vec3f> &circles);

    std::vector<cv::Vec3f> removeFarCircles(const std::vector<cv::Vec3f> &circles, float percentage);

    cv::Vec3f findLargestCircle(const std::vector<cv::Vec3f> &circles);

    cv::Mat convert_to_8bit(const cv::Mat &image);

private:
    int Error_Number;
    int CamIndex;
    int ROI_Width;
    int ROI_Height;
    int ROI_Bin;
    int ROI_Image_type;
    uint32_t exposure;
    uint32_t gain;
    bool stop_capture = false;
    IplImage *pImage;
    ASI_CAMERA_INFO CamInfo;
    std::queue<cv::Mat> image_queue;
    std::vector<cv::Vec3f> circles;

#ifdef _LIN
    pthread_t capture_thread;
#elif defined _WINDOWS
    HANDLE thread_setgainexp;
#endif
};

#endif
