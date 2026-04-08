#include "../../include/core/neuron.hpp"

// Definición de miembros estáticos thread-local
thread_local std::mt19937 IzhikevichNeuron::rng{std::random_device{}()};
thread_local std::uniform_real_distribution<double> IzhikevichNeuron::uniform{-0.5, 0.5};

IzhikevichNeuron::IzhikevichNeuron(NeuronType type, double i_offset, double noise)
        : I_syn(0.0), i_offset(i_offset), noise(noise), spiked(false) {
    switch(type) {
        case NeuronType::REGULAR_SPIKING:
            a = 0.02; b = 0.2; c = -65.0; d = 8.0;
            break;
        case NeuronType::FAST_SPIKING:
            a = 0.1; b = 0.2; c = -65.0; d = 2.0;
            break;
        case NeuronType::INTRINSICALLY_BURSTING:
            a = 0.02; b = 0.2; c = -55.0; d = 4.0;
            break;
        case NeuronType::CHATTERING:
            a = 0.02; b = 0.2; c = -50.0; d = 2.0;
            break;
        case NeuronType::LOW_THRESHOLD_SPIKING:
            a = 0.02; b = 0.25; c = -65.0; d = 2.0;
            break;
        case NeuronType::RESONATOR:
            a = 0.1; b = 0.26; c = -65.0; d = 2.0;
            break;
        default:
            a = 0.02; b = 0.2; c = -65.0; d = 8.0;
            break;
    }

    v = c;
    u = b * v;
}

/**
 *  Inicializador para una neurona personalizada
 */
IzhikevichNeuron::IzhikevichNeuron(double a, double b, double c, double d,
                                   double i_offset, double noise)
        : a(a), b(b), c(c), d(d), v(c), u(b*c),
          I_syn(0.0), i_offset(i_offset), noise(noise), spiked(false) {}

/**
 *  Método de Euler (dos sub-pasos de dt/2 por precisión, como Izhikevich recomienda)
 *  Corriente total = I_syn + i_offset + noise * Uniform(-0.5, 0.5)  [igual que ANNarchy]
 */
void IzhikevichNeuron::step(double dt) {
    spiked = false;

    // Corriente total: sináptica + offset constante + ruido uniforme
    double I_total = I_syn + i_offset + noise * uniform(rng);

    for (int i = 0; i < 2; i++) {
        double dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I_total) * (dt / 2.0);
        v += dv;

        double du = a * (b * v - u) * (dt / 2.0);
        u += du;
    }

    if (v >= V_THRESHOLD) {
        spiked = true;
        v = c;
        u += d;
    }

    I_syn = 0.0;
}

/**
 *  Agrega corriente a la neurona
 */
void IzhikevichNeuron::addCurrent(double current) {
    I_syn += current;
}

/**
 *  Restaura los valores iniciales de la neurona
 */
void IzhikevichNeuron::reset() {
    v = c;
    u = b * v;
    I_syn = 0.0;
    spiked = false;
}
