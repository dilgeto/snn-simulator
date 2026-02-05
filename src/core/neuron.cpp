#include "../../include/core/neuron.hpp"

/**
 *
 */
IzhikevichNeuron::IzhikevichNeuron(NeuronType type) : I_syn(0.0), spiked(false) {
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

    // Inicializar estado en reposo
    v = c;
    u = b * v;
}

/**
 *  Inicializador para una neurona personalizada
 */
IzhikevichNeuron::IzhikevichNeuron(double a, double b, double c, double d) 
        : a(a), b(b), c(c), d(d), v(c), u(b*c), I_syn(0.0), spiked(false) {}

/**
 *  Método de Euler
 */
void IzhikevichNeuron::step(double dt) {
    spiked = false;

    // verificar si hay spike
    if (v >= V_THRESHOLD) {
        spiked = true;
        v = c; // Reset del voltaje
        u += d; // Reset de la recuperación
    }

    // Ecuaciones de Izhikevich
    // dv/dt = 0.04*v^2 + 5*v + 140 - u + I
    // du/dt = a * (b*v - u)

    // Integración de v (dos pasos para mayor precisión)
    // Izhikevich recomienda usar dt = 0.5ms o ods pasos de 0.5ms

    for (int i = 0; i < 2; i++) {
        double dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I_syn) * (dt / 2.0);
        v += dv;

        // Actualizar u
        double du = a * (b * v - u) * (dt / 2.0);
        u += du;
    }

    // Reset de corriente sináptica (se acumula cada paso)
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
