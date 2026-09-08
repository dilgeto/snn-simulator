#include "../../include/core/neuron.hpp"

#include <algorithm>

// Definición de miembros estáticos thread-local
thread_local std::mt19937 IzhikevichNeuron::rng{std::random_device{}()};
thread_local std::uniform_real_distribution<double> IzhikevichNeuron::uniform{-0.5, 0.5};

IzhikevichNeuron::IzhikevichNeuron(NeuronType type, double i_offset, double noise,
                                   double tau_exc, double tau_inh)
        : i_offset(i_offset), noise(noise), spiked(false), enabled(true),
          g_exc(0.0), g_inh(0.0),
          tau_exc(tau_exc), tau_inh(tau_inh) {
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
        // Izhikevich (2004), Fig. 1 — 20 patrones de disparo.
        case NeuronType::TONIC_SPIKING:
            a = 0.02; b = 0.2; c = -65.0; d = 6.0;
            break;
        case NeuronType::PHASIC_SPIKING:
            a = 0.02; b = 0.25; c = -65.0; d = 6.0;
            break;
        case NeuronType::TONIC_BURSTING:
            a = 0.02; b = 0.2; c = -50.0; d = 2.0;
            break;
        case NeuronType::PHASIC_BURSTING:
            a = 0.02; b = 0.25; c = -55.0; d = 0.05;
            break;
        case NeuronType::MIXED_MODE:
            a = 0.02; b = 0.2; c = -55.0; d = 4.0;
            break;
        case NeuronType::SPIKE_FREQUENCY_ADAPTATION:
            a = 0.01; b = 0.2; c = -65.0; d = 8.0;
            break;
        case NeuronType::CLASS1_EXCITABLE:
            a = 0.02; b = -0.1; c = -55.0; d = 6.0;
            break;
        case NeuronType::CLASS2_EXCITABLE:
            a = 0.2; b = 0.26; c = -65.0; d = 0.0;
            break;
        case NeuronType::SPIKE_LATENCY:
            a = 0.02; b = 0.2; c = -65.0; d = 6.0;
            break;
        case NeuronType::SUBTHRESHOLD_OSCILLATIONS:
            a = 0.05; b = 0.26; c = -60.0; d = 0.0;
            break;
        case NeuronType::INTEGRATOR:
            a = 0.02; b = -0.1; c = -55.0; d = 6.0;
            break;
        case NeuronType::REBOUND_SPIKE:
            a = 0.03; b = 0.25; c = -60.0; d = 4.0;
            break;
        case NeuronType::REBOUND_BURST:
            a = 0.03; b = 0.25; c = -52.0; d = 0.0;
            break;
        case NeuronType::THRESHOLD_VARIABILITY:
            a = 0.03; b = 0.25; c = -60.0; d = 4.0;
            break;
        case NeuronType::BISTABILITY:
            a = 1.0; b = 1.5; c = -60.0; d = 0.0;
            break;
        case NeuronType::DEPOLARIZING_AFTERPOTENTIAL:
            a = 1.0; b = 0.2; c = -60.0; d = -21.0;
            break;
        case NeuronType::ACCOMMODATION:
            a = 0.02; b = 1.0; c = -55.0; d = 4.0;
            break;
        case NeuronType::INHIBITION_INDUCED_SPIKING:
            a = -0.02; b = -1.0; c = -60.0; d = 8.0;
            break;
        case NeuronType::INHIBITION_INDUCED_BURSTING:
            a = -0.026; b = -1.0; c = -45.0; d = -2.0;
            break;
        default:
            a = 0.02; b = 0.2; c = -65.0; d = 8.0;
            break;
    }

    v = c;
    u = b * v;
}

IzhikevichNeuron::IzhikevichNeuron(double a, double b, double c, double d,
                                   double i_offset, double noise,
                                   double tau_exc, double tau_inh)
        : a(a), b(b), c(c), d(d), v(c), u(b*c),
          i_offset(i_offset), noise(noise), spiked(false), enabled(true),
          g_exc(0.0), g_inh(0.0),
          tau_exc(tau_exc), tau_inh(tau_inh) {}

void IzhikevichNeuron::setNeuronType(NeuronType type) {
    switch(type) {
        case NeuronType::REGULAR_SPIKING:
            a = 0.02; b = 0.2; c = -65.0; d = 8.0; break;
        case NeuronType::FAST_SPIKING:
            a = 0.1;  b = 0.2; c = -65.0; d = 2.0; break;
        case NeuronType::INTRINSICALLY_BURSTING:
            a = 0.02; b = 0.2; c = -55.0; d = 4.0; break;
        case NeuronType::CHATTERING:
            a = 0.02; b = 0.2; c = -50.0; d = 2.0; break;
        case NeuronType::LOW_THRESHOLD_SPIKING:
            a = 0.02; b = 0.25; c = -65.0; d = 2.0; break;
        case NeuronType::RESONATOR:
            a = 0.1;  b = 0.26; c = -65.0; d = 2.0; break;

        // Izhikevich (2004), Fig. 1 — 20 patrones de disparo.
        case NeuronType::TONIC_SPIKING:
            a = 0.02; b = 0.2;  c = -65.0; d = 6.0;   break;
        case NeuronType::PHASIC_SPIKING:
            a = 0.02; b = 0.25; c = -65.0; d = 6.0;   break;
        case NeuronType::TONIC_BURSTING:
            a = 0.02; b = 0.2;  c = -50.0; d = 2.0;   break;
        case NeuronType::PHASIC_BURSTING:
            a = 0.02; b = 0.25; c = -55.0; d = 0.05;  break;
        case NeuronType::MIXED_MODE:
            a = 0.02; b = 0.2;  c = -55.0; d = 4.0;   break;
        case NeuronType::SPIKE_FREQUENCY_ADAPTATION:
            a = 0.01; b = 0.2;  c = -65.0; d = 8.0;   break;
        case NeuronType::CLASS1_EXCITABLE:
            a = 0.02; b = -0.1; c = -55.0; d = 6.0;   break;
        case NeuronType::CLASS2_EXCITABLE:
            a = 0.2;  b = 0.26; c = -65.0; d = 0.0;   break;
        case NeuronType::SPIKE_LATENCY:
            a = 0.02; b = 0.2;  c = -65.0; d = 6.0;   break;
        case NeuronType::SUBTHRESHOLD_OSCILLATIONS:
            a = 0.05; b = 0.26; c = -60.0; d = 0.0;   break;
        case NeuronType::INTEGRATOR:
            a = 0.02; b = -0.1; c = -55.0; d = 6.0;   break;
        case NeuronType::REBOUND_SPIKE:
            a = 0.03; b = 0.25; c = -60.0; d = 4.0;   break;
        case NeuronType::REBOUND_BURST:
            a = 0.03; b = 0.25; c = -52.0; d = 0.0;   break;
        case NeuronType::THRESHOLD_VARIABILITY:
            a = 0.03; b = 0.25; c = -60.0; d = 4.0;   break;
        case NeuronType::BISTABILITY:
            a = 1.0;  b = 1.5;  c = -60.0; d = 0.0;   break;
        case NeuronType::DEPOLARIZING_AFTERPOTENTIAL:
            a = 1.0;  b = 0.2;  c = -60.0; d = -21.0; break;
        case NeuronType::ACCOMMODATION:
            a = 0.02; b = 1.0;  c = -55.0; d = 4.0;   break;
        case NeuronType::INHIBITION_INDUCED_SPIKING:
            a = -0.02;  b = -1.0; c = -60.0; d = 8.0;  break;
        case NeuronType::INHIBITION_INDUCED_BURSTING:
            a = -0.026; b = -1.0; c = -45.0; d = -2.0; break;

        default:
            a = 0.02; b = 0.2; c = -65.0; d = 8.0; break;
    }
    v = c;
    u = b * v;
    g_exc = 0.0;
    g_inh = 0.0;
    spiked = false;
}

/**
 *  Método de Euler (dos sub-pasos de dt/2 por precisión, como Izhikevich recomienda)
 *
 *  Corriente total (como ANNarchy):
 *    I_total = g_exc - g_inh + I_syn + i_offset + noise * Uniform(-0.5, 0.5)
 *
 *  Decaimiento de conductancias (Euler de tau*dg/dt = -g):
 *    g_exc -= g_exc / tau_exc * dt
 *    g_inh -= g_inh / tau_inh * dt
 */
void IzhikevichNeuron::step(double dt) {
    spiked = false;


    // Decaimiento exponencial de conductancias (tau * dg/dt = -g)
    g_exc *= exp( -dt / tau_exc);
    g_inh *= exp( -dt / tau_inh);

    // Integración de v y u (dos sub-pasos de dt/2)
    for (int i = 0; i < 2; i++) {
        // Corriente total: excitatoria - inhibitoria + directa + offset + ruido
        double I_total = g_exc - g_inh + i_offset + noise * uniform(rng);

        double dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I_total) * (dt / 2.0);
        v += dv;
        // Red de seguridad, no el límite de disparo: ver comentario junto a
        // V_SAFETY_MAX/MIN en neuron.hpp. No usar V_THRESHOLD acá.
        v = std::clamp(v, V_SAFETY_MIN, V_SAFETY_MAX);

        double du = a * (b * v - u) * (dt / 2.0);
        u += du;
        u = std::clamp(u, -U_SAFETY_LIMIT, U_SAFETY_LIMIT);
    }

    if (v >= V_THRESHOLD) {
        spiked = true;
        v = c;
        u += d;
    }

}

/**
 *  Acumula conductancia excitatoria (AMPA) — llamar antes de step()
 *  cuando dispara una neurona presináptica excitatoria
 */
void IzhikevichNeuron::addExcConductance(double g) {
    g_exc += g;
}

/**
 *  Acumula conductancia inhibitoria (INH) — llamar antes de step()
 *  cuando dispara una neurona presináptica inhibitoria
 */
void IzhikevichNeuron::addInhConductance(double g) {
    g_inh += g;
}

/**
 *  Restaura los valores iniciales de la neurona
 */
void IzhikevichNeuron::reset() {
    v = c;
    u = b * v;
    g_exc = 0.0;
    g_inh = 0.0;
    spiked = false;
}
