#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include "network.hpp"

class Simulator {
private:
    Network net;
    double dt;  // Paso de tiempo (típicamente 0.1-1.0 ms)
    int spikeCount;
    
public:
    Simulator(double timestep = 1.0);
    
    void generateTopology();

    // Ejecutar simulación
    void startSimulation(
        //const std::vector<double>& inputs,
        double weight,
        double duration
        //PoissonEncoder* encoder
        //Decoder* decoder
    );
    /*
    std::vector<double> startSimulation(
        const std::vector<double>& inputs,
        double duration,
        Encoder* encoder
        //Decoder* decoder
    );
    */
    
    //void loadNetwork(const WANNGenome& genome, double weight);
};

#endif