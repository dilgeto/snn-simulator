#include <CLI/CLI.hpp>
#include "../../include/core/network.hpp"

int main(int argc, char** argv) {
    CLI::App app{"Spiking Neural Network Simulator"};

    // Variables a utilizar
    double duration = 100.0;
    double dt = 1.0;
    double weight = 1.0;

    app.add_option("-t,--duration", duration, "Total duration of simulation")
        ->check(CLI::PositiveNumber);
    app.add_option("-d,--timestep", dt, "Timestep")
        ->check(CLI::PositiveNumber);
    app.add_option("-w,--weight", weight, "Weight")
        ->check(CLI::Number);

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
        double input1 = 10.0, input2 = 10.0;

        // Simular por 100ms
        for (double t = 0; t < duration; t += 1.0) {
            // Aplicar corriente constante a las entradas
            net.setInputCurrents(std::vector<double>{input1, input2});
            
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

/*
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include "../../include/core/simulator.hpp"
#include "../../include/encoding/rlEncoder.hpp"
#include "../../include/decoding/decoder.hpp"

namespace rlt = rl_tools;

int main() {
    // 1. Configurar environment de RLTools
    using T = float;
    using TI = typename rlt::rl::environments::pendulum::Specification<T, TI>::INDEX;
    using Environment = rlt::rl::environments::Pendulum<rlt::rl::environments::pendulum::Specification<T, TI>>;
    
    Environment env;
    typename Environment::State state;
    typename Environment::Parameters env_params;
    
    // 2. Tu SNN + WANN
    Network net(1.0);
    
    // Construir topología (esto vendría del genoma WANN)
    int in1 = net.addInputNeuron();
    int in2 = net.addInputNeuron();
    int in3 = net.addInputNeuron();  // Pendulum tiene 3 observaciones
    
    int h1 = net.addHiddenNeuron();
    int h2 = net.addHiddenNeuron();
    
    int out = net.addOutputNeuron();  // 1 acción (torque continuo)
    
    // Conexiones
    net.addSynapse(in1, h1);
    net.addSynapse(in2, h1);
    net.addSynapse(in3, h2);
    net.addSynapse(h1, out);
    net.addSynapse(h2, out);
    
    // 3. Encoder/Decoder
    RLEncoder encoder(RLEncoder::EncodingType::RATE, 100.0);
    RLDecoder decoder(RLDecoder::DecodingType::RATE, 50.0);
    
    Simulator sim(&net);
    sim.setEncoder(std::make_unique<RLEncoder>(encoder));
    sim.setDecoder(std::make_unique<RLDecoder>(decoder));
    
    // 4. EVALUAR un individuo WANN con un peso específico
    double weight = 5.0;  // Este vendría de WANN
    
    double total_reward = 0.0;
    int max_steps = 200;
    
    // Reset del environment (RLTools)
    rlt::sample_initial_state(env, env_params, state, rng);
    
    for (int step = 0; step < max_steps; step++) {
        // a) Obtener observación del environment
        std::vector<double> observation = {
            state.theta,      // Ángulo
            state.theta_dot   // Velocidad angular
            // RLTools ya normaliza estos valores
        };
        
        // b) Tu SNN procesa la observación
        net.reset();
        auto result = sim.simulateEncoded(observation, weight, 50.0);
        
        // c) Decodificar acción
        double action = decoder.decodeContinuousAction(result.output_spike_times[0]);
        
        // d) Ejecutar acción en el environment (RLTools)
        typename Environment::Action rl_action;
        rl_action.value = action;
        
        T reward;
        bool terminated = false;
        rlt::step(env, env_params, state, rl_action, state, reward, terminated, rng);
        
        total_reward += reward;
        
        if (terminated) break;
    }
    
    std::cout << "Fitness (total reward): " << total_reward << std::endl;
    
    return 0;
}
*/