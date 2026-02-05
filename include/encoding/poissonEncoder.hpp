#ifndef POISSON_ENCODER_HPP
#define POISSON_ENCODER_HPP

#include "encoder.hpp"

// Codificación usando proceso de Poisson inhomogéneo
// Permite variación temporal de la tasa de disparo
class PoissonEncoder : public Encoder {
private:
    double max_rate;
    bool use_refractory;      // Usar período refractario
    double refractory_period; // Período refractario en ms
    
public:
    PoissonEncoder(double max_rate = 100.0, 
                   bool use_refractory = true,
                   double refractory_period = 2.0);
    
    std::vector<double> encode(double value, double duration, double dt) const override;
    
    // Codificar con tasa variable en el tiempo
    std::vector<double> encodeTemporalPattern(
        const std::vector<double>& rate_pattern,
        double dt) const;
    
    void setMaxRate(double rate) { max_rate = rate; }
    void setRefractoryPeriod(double period) { refractory_period = period; }
};

#endif