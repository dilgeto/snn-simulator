// Constantes del solver, de b2Settings.h (Box2D 2.3.x). Están en float como en Box2D.
#pragma once

namespace bipedal_walker::physics {

    template <typename T>
    struct Settings {
        static constexpr T PI = static_cast<T>(3.14159265359f);
        static constexpr T LINEAR_SLOP = static_cast<T>(0.005f);
        static constexpr T ANGULAR_SLOP = static_cast<T>(2.0f / 180.0f * 3.14159265359f);
        static constexpr T POLYGON_RADIUS = static_cast<T>(2.0f * 0.005f);
        static constexpr T VELOCITY_THRESHOLD = static_cast<T>(1.0f);
        static constexpr T MAX_LINEAR_CORRECTION = static_cast<T>(0.2f);
        static constexpr T MAX_ANGULAR_CORRECTION = static_cast<T>(8.0f / 180.0f * 3.14159265359f);
        static constexpr T MAX_TRANSLATION = static_cast<T>(2.0f);
        static constexpr T MAX_TRANSLATION_SQUARED = MAX_TRANSLATION * MAX_TRANSLATION;
        static constexpr T MAX_ROTATION = static_cast<T>(0.5f * 3.14159265359f);
        static constexpr T MAX_ROTATION_SQUARED = MAX_ROTATION * MAX_ROTATION;
        static constexpr T BAUMGARTE = static_cast<T>(0.2f);
    };

}
