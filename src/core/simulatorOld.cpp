#include <iostream>
#include "../../include/core/simulatorOld.hpp"

Simulator::Simulator(double timestep) : dt(timestep), spikeCount(0) {
    Network net(dt);
}

void Simulator::generateTopology() {
    // Input layer
    int in0 = net.addInputNeuron(NeuronType::REGULAR_SPIKING);
    int in1 = net.addInputNeuron(NeuronType::REGULAR_SPIKING);
    int in2 = net.addInputNeuron(NeuronType::REGULAR_SPIKING);

    // Hidden layers
    // Layer 0
    int h0_0 = net.addHiddenNeuron(NeuronType::REGULAR_SPIKING);
    int h0_1 = net.addHiddenNeuron(NeuronType::REGULAR_SPIKING);
    int h0_2 = net.addHiddenNeuron(NeuronType::REGULAR_SPIKING);
    int h0_3 = net.addHiddenNeuron(NeuronType::REGULAR_SPIKING);
    int h0_4 = net.addHiddenNeuron(NeuronType::REGULAR_SPIKING);

    /*
    // Layer 1
    int h1_0 = net.addHiddenNeuron();
    int h1_1 = net.addHiddenNeuron();
    int h1_2 = net.addHiddenNeuron();
    int h1_3 = net.addHiddenNeuron();
    int h1_4 = net.addHiddenNeuron();

    // Layer 2
    int h2_0 = net.addHiddenNeuron();
    int h2_1 = net.addHiddenNeuron();
    int h2_2 = net.addHiddenNeuron();
    int h2_3 = net.addHiddenNeuron();
    int h2_4 = net.addHiddenNeuron();
    */
    
    // Output layer
    int out = net.addOutputNeuron(NeuronType::REGULAR_SPIKING);
    
    // Synapses
    net.addSynapse(in0, h0_0);
    net.addSynapse(in0, h0_1);
    net.addSynapse(in0, h0_2);
    net.addSynapse(in0, h0_3);
    net.addSynapse(in0, h0_4);
    net.addSynapse(in1, h0_0);
    net.addSynapse(in1, h0_1);
    net.addSynapse(in1, h0_2);
    net.addSynapse(in1, h0_3);
    net.addSynapse(in1, h0_4);
    net.addSynapse(in2, h0_0);
    net.addSynapse(in2, h0_1);
    net.addSynapse(in2, h0_2);
    net.addSynapse(in2, h0_3);
    net.addSynapse(in2, h0_4);
    net.addSynapse(h0_0, out);
    net.addSynapse(h0_1, out);
    net.addSynapse(h0_2, out);
    net.addSynapse(h0_3, out);
    net.addSynapse(h0_4, out);

    std::cout << "Topology generated!" << std::endl;
    net.printInfo();
    net.printTopology();
}

void Simulator::startSimulation(
    double weight,
    double duration
    //PoissonEncoder* encoder
    ) {
    for (double t = 0; t < duration; t += 1.0) {
        net.setInputCurrents({20.0, 20.0});

        net.step(weight);

        auto outputSpikes = net.getOutputSpikes();
        if (outputSpikes[0]) {
            spikeCount++;
        }
    }

    std::cout << "Total output spikes: " << spikeCount << std::endl;
}