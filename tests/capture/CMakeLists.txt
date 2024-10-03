# Créer l'exécutable pour les tests
add_executable(capture_test capture.cpp)

# Lier l'exécutable avec la bibliothèque ZWO_class et les bibliothèques nécessaires
target_link_libraries(capture_test ASI071 
                        ${OpenCV_LIBS} 
                        ASICamera2
                        Threads::Threads)
