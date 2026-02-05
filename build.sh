#!/bin/bash

# Crear directorio build si no existe
mkdir -p build

# Ir a build
cd build

# Configurar y compilar
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Volver a raíz
cd ..

echo "Compilación completada. Ejecutar con: ./build/snn_simulator"