#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <vector>
#include "../../include/core/network.hpp"

TEST_CASE("Neural network creation") {
    // Variables
    //double I = 100.0;
    double dt = 1.0;
    Network net(dt);

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

    REQUIRE(net.getNeuronCount() == 5);
    REQUIRE(net.getSynapseCount() == 6);
    REQUIRE(net.getInputCount() == 2);
    REQUIRE(net.getOutputCount() == 1);
}

TEST_CASE("Sinaptic Links") {
    // Variables
    double dt = 1.0;
    double I = 10.0;
    double w = 20.0;
    double simulationTime = 100.0;

    int spikeCount = 0;

    // Red
    Network net(dt);

    // Neuronas
    int input = net.addInputNeuron();
    int output = net.addOutputNeuron();

    // Conexiones sinápticas
    net.addSynapse(input, output);

    // 
    for (double t = 0; t < simulationTime; t += 1.0) {
        net.setInputCurrent(input, I);
        net.step(w);

        auto outputSpikes = net.getOutputSpikes();
        if (outputSpikes[0]) {
            spikeCount++;
        }
    }
    
    std::cout << "Spikes count: " << spikeCount << std::endl;

    REQUIRE(spikeCount > 0);
}