#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include "../../include/core/neuron.hpp"

/**
 * 
 */
TEST_CASE("Single Neuron Test") {
    // Variables
    double I = 10.0;
    double simulationTime = 200.0;
    double dt = 1.0;

    int spikeCount = 0;

    IzhikevichNeuron neuron = IzhikevichNeuron(NeuronType::REGULAR_SPIKING);
    for (double t = 0; t < simulationTime; t+= dt) {
        neuron.addCurrent(I);
        neuron.step(dt);
        if (neuron.hasSpiked()) spikeCount++;
    }

    std::cout << "The neuron has spiked: " << spikeCount << " times." << std::endl;
    REQUIRE(spikeCount > 0);
}
