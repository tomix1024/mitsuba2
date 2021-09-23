#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/texture.h>
//#include <mitsuba/render/medium.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _phase-hg:

Henyey-Greenstein phase function (:monosp:`hg`)
-----------------------------------------------

.. list-table::
 :widths: 20 15 65
 :header-rows: 1
 :class: paramstable

 * - Parameter
   - Type
   - Description
 * - g
   - |Volume|
   - This parameter must be somewhere in the range -1 to 1
     (but not equal to -1 or 1). It denotes the *mean cosine* of scattering
     interactions. A value greater than zero indicates that medium interactions
     predominantly scatter incident light into a similar direction (i.e. the
     medium is *forward-scattering*), whereas values smaller than zero cause
     the medium to be scatter more light in the opposite direction.

This plugin implements the phase function model proposed by
Henyey and Greenstein |nbsp| :cite:`Henyey1941Diffuse`. It is
parameterizable from backward- (g<0) through
isotropic- (g=0) to forward (g>0) scattering.

*/
template <typename Float, typename Spectrum>
class HGPhaseFunctionVarying final : public PhaseFunction<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(PhaseFunction, m_flags)
    MTS_IMPORT_TYPES(PhaseFunctionContext, Volume)

    HGPhaseFunctionVarying(const Properties &props) : Base(props) {
        //m_g = props.float_("g", 0.8f);
        m_g = props.volume<Volume>("g", 0.8f);
/*
        if (m_g >= 1 || m_g <= -1)
            Log(Error, "The asymmetry parameter must lie in the interval (-1, 1)!");
*/
        m_flags = +PhaseFunctionFlags::Anisotropic;
    }

    MTS_INLINE Float eval_hg(Float cos_theta, MediumInteraction3f mi, Mask active) const {
        auto mg_eval = m_g->eval_1(mi, active);  // TODO this will only work in monochrome mode, disregard for now
        Float temp = 1.0f + mg_eval * mg_eval + 2.0f * mg_eval * cos_theta;
        return math::InvFourPi<ScalarFloat> * (1 - mg_eval * mg_eval) / (temp * enoki::sqrt(temp));
    }

    std::pair<Vector3f, Float> sample(const PhaseFunctionContext & /* ctx */,
                                      const MediumInteraction3f &mi, const Point2f &sample,
                                      Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

	Mask my_mask = enoki::abs(m_g) < math::Epsilon<ScalarFloat>;

	Float sqr_term = (1 - m_g * m_g) / (1 - m_g + 2 * m_g * sample.x());
	Float cos_theta_2 = (1 + m_g * m_g - sqr_term * sqr_term) / (2 * m_g);
	Float cos_theta_1 = 1 - 2 * sample.x();

	Float cos_theta = enoki::select(my_mask, cos_theta_1, cos_theta_2);
        Float sin_theta = enoki::safe_sqrt(1.0f - cos_theta * cos_theta);
        auto [sin_phi, cos_phi] = enoki::sincos(2 * math::Pi<ScalarFloat> * sample.y());
        auto wo = Vector3f(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        wo = mi.to_world(wo);
        Float pdf = eval_hg(-cos_theta, mi, active);
        return std::make_pair(wo, pdf);
    }

    Float eval(const PhaseFunctionContext & /* ctx */, const MediumInteraction3f &mi,
               const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);
        return eval_hg(dot(wo, mi.wi), mi, active);
    }

    void traverse(TraversalCallback *callback) override {
        //callback->put_parameter("g", m_g);
        callback->put_object("g", m_g.get());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "HGPhaseFunctionVarying[" << std::endl
            << "  g = " << string::indent(m_g) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Volume> m_g;

};

MTS_IMPLEMENT_CLASS_VARIANT(HGPhaseFunctionVarying, PhaseFunction)
MTS_EXPORT_PLUGIN(HGPhaseFunctionVarying, "Henyey-Greenstein phase function with spacial variance")
NAMESPACE_END(mitsuba)