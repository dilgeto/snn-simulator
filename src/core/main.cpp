#include <CLI/CLI.hpp>
#include <iostream>
#include <vector>
#include "../../include/core/network.hpp"

int main(int argc, char** argv) {
    CLI::App app{"Spiking Neural Network Simulator"};

    // Variables a utilizar
    double time = 100.0;
    double dt = 1.0;
    double input1 = 20.0;
    double input2 = 15.0;

    app.add_option("-t,--total_time", time, "Total time of simulation")
        ->check(CLI::PositiveNumber);
    app.add_option("-d,--timestep", dt, "Timestep")
        ->check(CLI::PositiveNumber);
    app.add_option("-i,--input1", input1, "First neuron input")
        ->check(CLI::PositiveNumber);
    app.add_option("-y,--input2", input2, "Second neuron input")
        ->check(CLI::PositiveNumber);

    CLI11_PARSE(app, argc, argv);

    // Se crea la red con el timestep definido
    Network net(dt);
    
    // Construir red: 2 inputs -> 2 hidden -> 1 output
    std::cout << "Construyendo red..." << std::endl;
    
    int in1 = net.addInputNeuron();
    int in2 = net.addInputNeuron();
    
    int h1 = net.addHiddenNeuron(NeuronType::REGULAR_SPIKING);
    int h2 = net.addHiddenNeuron(NeuronType::REGULAR_SPIKING);
    
    int out = net.addOutputNeuron();
    
    // Conexiones (sin pesos)
    net.addSynapse(in1, h1);
    net.addSynapse(in1, h2);
    net.addSynapse(in2, h1);
    net.addSynapse(in2, h2);
    net.addSynapse(h1, out);
    net.addSynapse(h2, out);
    
    net.printInfo();
    net.printTopology();
    
    // Evaluar con diferentes pesos (típico en WANN)
    //std::vector<double> test_weights = {-2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0};
    std::vector<double> test_weights = {20.0};
    
    for (double weight : test_weights) {
        std::cout << "\n=== Evaluando con peso compartido: " << weight << " ===" << std::endl;
        
        net.reset();
        
        int spike_count = 0;
        
        // Simular por 100ms
        for (double t = 0; t < time; t += 1.0) {
            // Aplicar corriente constante a las entradas
            //net.setInputCurrents({20.0, 15.0});
            net.setInputCurrents({input1, input2});
            
            // Paso con peso compartido
            net.step(weight);
            
            // Contar spikes de salida
            auto output_spikes = net.getOutputSpikes();
            if (output_spikes[0]) {
                spike_count++;
            }
        }
        
        std::cout << "Total output spikes: " << spike_count << std::endl;
    }
    
    return 0;
}