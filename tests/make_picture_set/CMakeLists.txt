# Créer l'exécutable pour les tests
add_executable(make_picture_set_test make_picture_set.cpp)

# Lier l'exécutable avec la bibliothèque ZWO_class et les bibliothèques nécessaires
target_link_libraries(make_picture_set_test ASI071 
                        ${OpenCV_LIBS} 
                        ASICamera2
                        Threads::Threads)
