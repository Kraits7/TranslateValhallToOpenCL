cmake_minimum_required(VERSION 3.22.1)
project(UnisocCompute)

# Стандарт C++20 для продвинутых фишек
set(CMAKE_CXX_STANDARD 20)

# Добавляем наш основной файл (создадим его следующим шагом)
add_library(unisoc_compute SHARED native-lib.cpp)

# Ищем системные библиотеки Android
find_library(log-lib log)
find_library(dl-lib dl)

# Линкуем
target_link_libraries(unisoc_compute ${log-lib} ${dl-lib})

# Оптимизация под архитектуру твоего чипсета
target_compile_options(unisoc_compute PRIVATE -O3 -mfloat-abi=softfp -mfpu=neon)
