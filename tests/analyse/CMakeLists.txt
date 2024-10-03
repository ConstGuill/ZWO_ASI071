# Créer l'exécutable pour les tests
add_executable(analyse_test analyse.cpp)

# Lier l'exécutable avec la bibliothèque ZWO_class et les bibliothèques nécessaires
target_link_libraries(analyse_test ASI071 
                        ${OpenCV_LIBS} 
                        ASICamera2
                        Threads::Threads)
