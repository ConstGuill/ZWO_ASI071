# Créer l'exécutable pour les tests
add_executable(exposure_test exposure.cpp)

# Lier l'exécutable avec la bibliothèque ZWO_class et les bibliothèques nécessaires
target_link_libraries(exposure_test ASI071 
                        ${OpenCV_LIBS} 
                        ASICamera2
                        Threads::Threads)
