# Créer la bibliothèque ZWO_class
add_library(ASI071 ASI071.cpp)

# Lier la bibliothèque avec OpenCV et ASICamera2
target_link_libraries(ASI071 
                        ASICamera2
                        ${OpenCV_LIBS}
                        Threads::Threads)
