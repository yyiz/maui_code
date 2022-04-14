nvcc -arch=sm_60 -x cu -Iinc -dc src/gputils.cpp -o obj/gputils.obj
nvcc -arch=sm_60 -x cu -Iinc -dc src/test.cpp -o obj/test.obj
nvcc -arch=sm_60 -x cu -Iinc -dc src/mcml.cpp -o obj/mcml.obj
nvcc -arch=sm_60 -x cu -Iinc -dc src/main.cpp -o obj/main.obj
nvcc -arch=sm_60 obj/gputils.obj obj/test.obj obj/mcml.obj obj/main.obj -o maui
cmd /K maui.exe
