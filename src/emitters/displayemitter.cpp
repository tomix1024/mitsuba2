#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/ior.h>

NAMESPACE_BEGIN(mitsuba)


template <typename Float>
static std::tuple<Float, Float> fresnel_polarized_power_transmittance(Float cos_theta_i, Float cos_theta_t, Float eta) {
    auto outside_mask = cos_theta_i >= 0.f;

    Float rcp_eta = rcp(eta),
          eta_it = select(outside_mask, eta, rcp_eta),
          eta_ti = select(outside_mask, rcp_eta, eta);

    /* Find the absolute cosines of the incident/transmitted rays */
    Float cos_theta_i_abs = abs(cos_theta_i);
    Float cos_theta_t_abs = abs(cos_theta_t);

    auto index_matched = eq(eta, 1.f),
         special_case  = index_matched || eq(cos_theta_i_abs, 0.f);

    Float T_sc = select(index_matched, Float(1.f), Float(0.f));

    /* Amplitudes of reflected waves */
    /*
    Float a_s = fnmadd(eta_it, cos_theta_t_abs, cos_theta_i_abs) /
                 fmadd(eta_it, cos_theta_t_abs, cos_theta_i_abs);

    Float a_p = fnmadd(eta_it, cos_theta_i_abs, cos_theta_t_abs) /
                 fmadd(eta_it, cos_theta_i_abs, cos_theta_t_abs);
    */
    Float t_s = 2*cos_theta_i_abs / fmadd(eta_it, cos_theta_t_abs, cos_theta_i_abs);
    Float t_p = 2*cos_theta_i_abs / fmadd(eta_it, cos_theta_i_abs, cos_theta_t_abs);

    Float T_s = sqr(t_s);
    Float T_p = sqr(t_p);

    masked(T_s, special_case) = T_sc;
    masked(T_p, special_case) = T_sc;

    return { T_s, T_p };
}


template <typename Float, typename Spectrum>
class DisplayEmitter final : public Emitter<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Emitter, m_flags, m_shape, m_medium)
    MTS_IMPORT_TYPES(Scene, Shape, Texture)

    DisplayEmitter(const Properties &props) : Base(props) {

        ScalarFloat scene_ior = lookup_ior(props, "scene_ior", "air"); // water
        ScalarFloat display_ior = lookup_ior(props, "display_ior", "bk7");

        m_eta = scene_ior / display_ior;

        m_thickness = props.float_("thickness", 0);

        m_polarization_dir_x = props.float_("polarization_x", 1.0f);
        m_polarization_dir_y = props.float_("polarization_y", -1.0f);
        auto length = sqrt(sqr(m_polarization_dir_x) + sqr(m_polarization_dir_y));
        m_polarization_dir_x /= length;
        m_polarization_dir_y /= length;

        m_emission_coeff_0 = props.float_("emission_coeff_0", 1.0f);
        m_emission_coeff_1 = props.float_("emission_coeff_1", 0.0f);
        m_emission_coeff_2 = props.float_("emission_coeff_2", 0.0f);
        m_emission_coeff_3 = props.float_("emission_coeff_3", 0.0f);
        m_emission_coeff_4 = props.float_("emission_coeff_4", 0.0f);
        m_emission_coeff_5 = props.float_("emission_coeff_5", 0.0f);

        m_radiance = props.texture<Texture>("radiance", Texture::D65(1.f));

        m_flags = +EmitterFlags::Surface;
        if (m_radiance->is_spatially_varying())
            m_flags |= +EmitterFlags::SpatiallyVarying;
    }

    Spectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        // Evaluate the Fresnel equations for unpolarized illumination
        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        Float cos_theta_i_abs = abs(cos_theta_i);

        /* Using Snell's law, calculate the squared sine of the
        angle between the surface normal and the transmitted ray */
        Float cos_theta_t_sqr = 1.0f - m_eta * m_eta * (1.0f - cos_theta_i * cos_theta_i);
        Float cos_theta_t_abs = safe_sqrt(cos_theta_t_sqr);
        Float cos_theta_t = mulsign_neg(cos_theta_t_abs, cos_theta_i);

        // Vector3f wo = refract(si.wi, cos_theta_t, Float(m_eta));

        auto [T_s, T_p] = fresnel_polarized_power_transmittance(cos_theta_i, cos_theta_t, Float(m_eta));

        auto T = 0.5*(T_s + T_p) * cos_theta_t_abs / cos_theta_i_abs;

        /*
        Vector2f parallel_dir = Vector2f(si.wi.x(), si.wi.y()) / (1.0f - Frame3f::cos_theta_2(si.wi));
        Float px = parallel_dir.x();
        Float py = parallel_dir.y();

        // displayToTransmissionPlaneMat = torch.stack([px, py, -py, px], dim=-1).view(*parallel_dir.shape[:-1], 2, 2)
        // J = fct.matvecmul(displayToTransmissionPlaneMat, args.displayPolarization)
        Float J_p = px*m_polarization_dir_x + py*m_polarization_dir_y; // Parallel
        Float J_s = px*m_polarization_dir_y - py*m_polarization_dir_x; // Senkrecht

        Float T = (sqr(J_s)*T_s + sqr(J_p)*T_p) * cos_theta_t_abs / cos_theta_i_abs;
        */

        /*
        auto [ r, cos_theta_t, eta_it, eta_ti ] = fresnel(cos_theta_i, Float(m_eta));
        Float T = 1-r;
        */

        /*
        Float E = 1+ 0 * fmadd(
                        fmadd(
                            fmadd(
                                fmadd(
                                    fmadd(
                                        m_emission_coeff_5,
                                    cos_theta_t_abs, m_emission_coeff_4),
                                cos_theta_t_abs, m_emission_coeff_3),
                            cos_theta_t_abs, m_emission_coeff_2),
                        cos_theta_t_abs, m_emission_coeff_1),
                    cos_theta_t_abs, m_emission_coeff_0);
        */

        // TODO modify si!!
        return T * m_radiance->eval(si, active);
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &sample2, const Point2f &sample3,
                                          Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        // TODO sample direction on emitter surface and transform into scene
        return {};
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        // TODO NEE manifold exploration here...
        return {};
    }

    Float pdf_direction(const Interaction3f &it, const DirectionSample3f &ds,
                        Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        // TODO evaluate pdf for sampled direction
        // this might be difficult!!
        return 0; // TODO
    }

    ScalarBoundingBox3f bbox() const override { return m_shape->bbox(); }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("radiance", m_radiance.get());
        callback->put_parameter("thickness", m_thickness);
        callback->put_parameter("eta", m_eta);
        callback->put_parameter("polarization_x", m_polarization_dir_x);
        callback->put_parameter("polarization_y", m_polarization_dir_y);
        callback->put_parameter("emission_coeff_0", m_emission_coeff_0);
        callback->put_parameter("emission_coeff_1", m_emission_coeff_1);
        callback->put_parameter("emission_coeff_2", m_emission_coeff_2);
        callback->put_parameter("emission_coeff_3", m_emission_coeff_3);
        callback->put_parameter("emission_coeff_4", m_emission_coeff_4);
        callback->put_parameter("emission_coeff_5", m_emission_coeff_5);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "DisplayEmitter[" << std::endl;
        oss << std::endl << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()

private:
    ScalarFloat  m_thickness;
    ScalarFloat  m_eta;
    ScalarFloat  m_polarization_dir_x;
    ScalarFloat  m_polarization_dir_y;

    ScalarFloat  m_emission_coeff_0;
    ScalarFloat  m_emission_coeff_1;
    ScalarFloat  m_emission_coeff_2;
    ScalarFloat  m_emission_coeff_3;
    ScalarFloat  m_emission_coeff_4;
    ScalarFloat  m_emission_coeff_5;

    ref<Texture> m_radiance;

};

MTS_IMPLEMENT_CLASS_VARIANT(DisplayEmitter, Emitter)
MTS_EXPORT_PLUGIN(DisplayEmitter, "Display emitter")
NAMESPACE_END(mitsuba)
