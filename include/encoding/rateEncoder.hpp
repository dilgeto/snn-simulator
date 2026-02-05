#ifndef RATE_ENCODER_HPP
#define RATE_ENCODER_HPP

#include "encoder.hpp"

// Codifica valores como frecuencia de disparo
// Valor alto = más spikes por segundo
class RateEncoder : public Encoder {
private:
    double max_rate;  // Frecuencia máxima en Hz
    
public:
    // max_rate: frecuencia máxima de disparo (típicamente 100-200 Hz)
    explicit RateEncoder(double max_rate = 100.0);
    
    std::vector<double> encode(double value, double duration, double dt) const override;
    
    void setMaxRate(double rate) { max_rate = rate; }
    double getMaxRate() const { return max_rate; }
};

#endif