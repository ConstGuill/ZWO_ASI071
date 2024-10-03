cmake_minimum_required(VERSION 3.10)
project(ASI071)

# Définir la version du standard C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Trouver les bibliothèques nécessaires
find_package(OpenCV REQUIRED)

# Find the pthread library
find_package(Threads REQUIRED)

# Définir les chemins pour ASICamera2
set(ASICAMERA2_PATH /opt/Driver_ZWO_ASI/ASI_Camera_SDK/ASI_linux_mac_SDK_V1.35)
set(ASICAMERA2_INCLUDE_DIR ${ASICAMERA2_PATH}/include)
#set(ASICAMERA2_LIBRARY ${ASICAMERA2_PATH}/lib/x64/) # Remplacez par le fichier correct si nécessaire

# Ajouter les répertoires d'en-tête
include_directories(${OpenCV_INCLUDE_DIRS} ${ASICAMERA2_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/include)

link_directories(/opt/Driver_ZWO_ASI/ASI_Camera_SDK/ASI_linux_mac_SDK_V1.35/lib/x64/)

# Ajouter les sous-répertoires
add_subdirectory(src)
add_subdirectory(tests)