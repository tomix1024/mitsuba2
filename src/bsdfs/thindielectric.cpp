#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/render/reflection.h>

NAMESPACE_BEGIN(mitsuba)

class ThinDielectric : public BSDF {
public:
    ThinDielectric(const Properties &props) {
        m_flags = (ENull | EDeltaReflection | EFrontSide | EBackSide);
        m_components.clear();
        m_components.push_back(EDeltaReflection | EFrontSide | EBackSide);
        m_components.push_back(ENull | EFrontSide | EBackSide);

        // Specifies the internal index of refraction at the interface
        Float int_ior = lookup_ior(props, "int_ior", "bk7");
        // Specifies the external index of refraction at the interface
        Float ext_ior = lookup_ior(props, "ext_ior", "air");

        if (int_ior < 0.0f || ext_ior < 0.0f)
            Throw("The interior and exterior indices of "
                  "refraction must be positive!");

        Properties props_eta("uniform");
        props_eta.set_float("value", int_ior / ext_ior);
        m_eta = (ContinuousSpectrum *) PluginManager::instance()
                ->create_object<ContinuousSpectrum>(props_eta).get();

        m_specular_reflectance   = props.spectrum("specular_reflectance",   1.0f);
        m_specular_transmittance = props.spectrum("specular_transmittance", 1.0f);
    }

    template <typename SurfaceInteraction,
              typename BSDFSample = BSDFSample<typename SurfaceInteraction::Point3>,
              typename Value      = typename SurfaceInteraction::Value,
              typename Point2     = typename SurfaceInteraction::Point2,
              typename Spectrum   = Spectrum<Value>>
    std::pair<BSDFSample, Spectrum> sample_impl(
            const BSDFContext &ctx, const SurfaceInteraction &si,
            Value sample1, const Point2 &/*sample2*/, mask_t<Value> active) const {
        using Index   = typename BSDFSample::Index;
        using Vector3 = typename BSDFSample::Vector3;
        using Frame   = Frame<Vector3>;
        using Mask    = mask_t<Value>;

        bool sample_reflection   = ctx.is_enabled(EDeltaReflection, 0);
        bool sample_transmission = ctx.is_enabled(ENull, 1);

        Value R, cos_theta_t, eta_ti;
        std::tie(R, cos_theta_t, std::ignore, eta_ti) = fresnel(
            abs(Frame::cos_theta(si.wi)),
            luminance(m_eta->eval(si.wavelengths, active), si.wavelengths));
        Value T = 1.0f - R;

        // Account for internal reflections: R' = R + TRT + TR^3T + ..
        masked(R, R < 1.0f) += T * T * R / (1.0f - R * R);

        BSDFSample bs;
        Spectrum value;

        if (sample_transmission && sample_reflection) {
            Mask reflection = (sample1 <= R);

            bs.sampled_component = select(reflection, Index(0), Index(1));
            bs.sampled_type =
                select(reflection, Index(EDeltaReflection), Index(ENull));
            bs.wo = select(reflection, reflect(si.wi),
                           refract(si.wi, cos_theta_t, eta_ti));

            bs.pdf = select(reflection, R, 1.0f - R);
            value  = select(reflection,
                            m_specular_reflectance->eval(si.wavelengths,   active),
                            m_specular_transmittance->eval(si.wavelengths, active));
        } else if (sample_reflection) {
            bs.sampled_component = 0;
            bs.sampled_type = EDeltaReflection;
            bs.wo = reflect(si.wi);

            bs.pdf = 1.0f;
            value = m_specular_reflectance->eval(si.wavelengths, active) * R;
        } else if (sample_transmission) {
            bs.sampled_component = 1;
            bs.sampled_type = ENull;
            bs.wo = refract(si.wi, cos_theta_t, eta_ti);

            bs.pdf = 1.0f;
            value = m_specular_transmittance->eval(si.wavelengths, active) * (1.0f - R);
        }

        return { bs, value };
    }

    template <typename SurfaceInteraction,
              typename Value    = typename SurfaceInteraction::Value,
              typename Vector3  = typename SurfaceInteraction::Vector3,
              typename Spectrum = Spectrum<Value>>
    Spectrum eval_impl(const BSDFContext &ctx, const SurfaceInteraction &si,
                       const Vector3 &wo, const mask_t<Value> &active) const {
        using Frame = Frame<Vector3>;
        using Mask    = mask_t<Value>;

        bool sample_reflection   = ctx.is_enabled(EDeltaReflection, 0);
        bool sample_transmission = ctx.is_enabled(ENull, 1);

        Value R, cos_theta_t, eta_ti;
        std::tie(R, cos_theta_t, std::ignore, eta_ti) =
            fresnel(abs(Frame::cos_theta(si.wi)),
                    luminance(m_eta->eval(si.wavelengths, active),
                              si.wavelengths, active));
        Value T = 1.0f - R;

        // Account for internal reflections: R' = R + TRT + TR^3T + ..
        masked(R, R < 1.0f) += T * T * R / (1.0f - R * R);

        Mask reflection =
            (Frame::cos_theta(si.wi) * Frame::cos_theta(wo) >= 0.0f);
        auto refracted = refract(si.wi, cos_theta_t, eta_ti);
        Mask valid_direction =
            active &&
                (Mask(sample_reflection) && reflection &&
                 (abs(dot(reflect(si.wi), wo) - 1.0f) <= math::DeltaEpsilon)) ||
            (Mask(sample_transmission) && !reflection &&
             (abs(dot(refracted, wo) - 1.0f) <= math::DeltaEpsilon));

        Spectrum value
            = select(reflection,
                 m_specular_reflectance->eval(si.wavelengths,   active) * R,
                 m_specular_transmittance->eval(si.wavelengths, active) * (1.0f - R));

        return select(valid_direction, value, 0.0f);
    }

    template <typename SurfaceInteraction,
              typename Value    = typename SurfaceInteraction::Value,
              typename Vector3  = typename SurfaceInteraction::Vector3>
    Value pdf_impl(const BSDFContext &ctx, const SurfaceInteraction &si,
                   const Vector3 &wo, const mask_t<Value> &active) const {
        using Frame = Frame<Vector3>;
        using Mask  = mask_t<Value>;

        bool sample_reflection   = ctx.is_enabled(EDeltaReflection, 0);
        bool sample_transmission = ctx.is_enabled(ENull, 1);

        Value R, cos_theta_t, eta_ti;
        std::tie(R, cos_theta_t, std::ignore, eta_ti) =
            fresnel(abs(Frame::cos_theta(si.wi)),
                    luminance(m_eta->eval(si.wavelengths, active),
                              si.wavelengths, active));
        Value T = 1.0f - R;

        // Account for internal reflections: R' = R + TRT + TR^3T + ..
        masked(R, R < 1.0f) += T * T * R / (1.0f - R * R);

        Mask reflection =
            (Frame::cos_theta(si.wi) * Frame::cos_theta(wo) >= 0.0f);
        auto refracted = refract(si.wi, cos_theta_t, eta_ti);
        Mask valid_direction =
            active &&
                (Mask(sample_reflection) && reflection &&
                 (abs(dot(reflect(si.wi), wo) - 1.0f) <= math::DeltaEpsilon)) ||
            (Mask(sample_transmission) && !reflection &&
             (abs(dot(refracted, wo) - 1.0f) <= math::DeltaEpsilon));

        Value pdf = select(reflection, R, (1.0f - R));
        return select(valid_direction, pdf, 0.0f);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "ThinDielectric[" << std::endl
            << "  eta = "                    << m_eta                    << "," << std::endl
            << "  specular_reflectance = "   << m_specular_reflectance   << "," << std::endl
            << "  specular_transmittance = " << m_specular_transmittance << std::endl
            << "]";
        return oss.str();
    }

    MTS_IMPLEMENT_BSDF()
    MTS_DECLARE_CLASS()
private:
    ref<ContinuousSpectrum> m_eta;
    ref<ContinuousSpectrum> m_specular_transmittance;
    ref<ContinuousSpectrum> m_specular_reflectance;
};

MTS_IMPLEMENT_CLASS(ThinDielectric, BSDF)
MTS_EXPORT_PLUGIN(ThinDielectric, "Thin dielectric BRDF")

NAMESPACE_END(mitsuba)