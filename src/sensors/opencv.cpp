#include <mitsuba/render/sensor.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/bbox.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sensor-opencv:

OpenCV pinhole camera (:monosp:`opencv`)
--------------------------------------------------

.. pluginparameters::

 * - to_world
   - |transform|
   - Specifies an optional camera-to-world transformation.
     (Default: none (i.e. camera space = world space))
 * - fx
   - |float|
   - Denotes the camera's horizontal focal length parameter.
 * - fy
   - |float|
   - Denotes the camera's vertical focal length parameter.
 * - cx
   - |float|
   - Denotes the camera's horizontal principal point parameter.
 * - cy
   - |float|
   - Denotes the camera's vertical principal point parameter.
 * - near_clip, far_clip
   - |float|
   - Distance to the near/far clip planes. (Default: :monosp:`near_clip=1e-2` (i.e. :monosp:`0.01`)
     and :monosp:`far_clip=1e4` (i.e. :monosp:`10000`))

This plugin implements a simple idealizied perspective camera model used in OpenCV, which
has an infinitely small aperture. This creates an infinite depth of field,
i.e. no optical blurring occurs.

 */

template <typename Float, typename Spectrum>
class OpenCVCamera final : public ProjectiveCamera<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(ProjectiveCamera, m_world_transform, m_needs_sample_3,
                    m_film, m_sampler, m_resolution, m_shutter_open,
                    m_shutter_open_time, m_near_clip, m_far_clip)
    MTS_IMPORT_TYPES()

    // =============================================================
    //! @{ \name Constructors
    // =============================================================

    OpenCVCamera(const Properties &props) : Base(props) {
        ScalarVector2i size = m_film->size();
        m_fx = props.float_("fx");
        m_fy = props.float_("fy");
        m_cx = props.float_("cx");
        m_cy = props.float_("cy");

        if (m_world_transform->has_scale())
            Throw("Scale factors in the camera-to-world transformation are not allowed!");

        update_camera_transforms();
    }

    ScalarTransform4f compute_camera_to_sample() {
        /*
        return perspective_projection(
            m_film->size(), m_film->crop_size(), m_film->crop_offset(),
            m_x_fov, m_near_clip, m_far_clip);
        */

        ScalarVector2f film_size = ScalarVector2f(Vector<int, 2>(m_film->size()));

        ScalarFloat aspect = film_size.x() / film_size.y();

        ScalarVector2f crop_size = ScalarVector2f(Vector<int, 2>(m_film->crop_size()));
        ScalarVector2f crop_offset = ScalarVector2f(Vector<int, 2>(m_film->crop_offset()));

        ScalarVector2f rel_size    = crop_size / film_size,
                 rel_offset  = crop_offset / film_size;

        ScalarTransform4f crop_image_transform =
            ScalarTransform4f::scale(
               ScalarVector3f(1.f / rel_size.x(), 1.f / rel_size.y(), 1.f)) *
            ScalarTransform4f::translate(
               ScalarVector3f(-rel_offset.x(), -rel_offset.y(), 0.f));

        /*
        ScalarTransform4f scale_image = ScalarTransform4f::scale(ScalarVector3f(-0.5f, -0.5f * aspect, 1.f));
        ScalarTransform4f translate_image = ScalarTransform4f::translate(ScalarVector3f(-1.f, -1.f / aspect, 0.f));

        ScalarTransform4f perspective = ScalarTransform4f::perspective(m_x_fov, m_near_clip, m_far_clip);
        */

        ScalarTransform4f image_to_viewport_transform =
            ScalarTransform4f::scale(
                ScalarVector3f(1.0f / film_size.x(), 1.0f / film_size.y(), 1.0f));
        /*
            ScalarTransform4f::translate(
                ScalarVector3f(-1.0f, -1.0f, 0.0f)) *
            ScalarTransform4f::scale(
                ScalarVector3f(2.0f / film_size.x(), 2.0f / film_size.y(), 1.0f));
        */


        ScalarFloat recip = 1.f / (m_far_clip - m_near_clip);
        ScalarMatrix4f opencv_camera_mat = ScalarMatrix4f::from_rows(
            ScalarVector4f(m_fx, 0.f, m_cx, 0.f),
            ScalarVector4f(0.f, m_fy, m_cy, 0.f),
            ScalarVector4f(0.f, 0.f, m_far_clip * recip, -m_near_clip * m_far_clip * recip),
            ScalarVector4f(0.f, 0.f, 1.f, 0.f)
        );

        ScalarTransform4f opencv_camera_transform = ScalarTransform4f(opencv_camera_mat);

        ScalarTransform4f axis_flip_transform = ScalarTransform4f::scale(ScalarVector3f(-1.0f, -1.0f, 1.f));

        return
           crop_image_transform *
           image_to_viewport_transform *
           opencv_camera_transform *
           axis_flip_transform;
    }

    void update_camera_transforms() {
        m_camera_to_sample = compute_camera_to_sample();

        m_sample_to_camera = m_camera_to_sample.inverse();

        // Position differentials on the near plane
        m_dx = m_sample_to_camera * ScalarPoint3f(1.f / m_resolution.x(), 0.f, 0.f) -
               m_sample_to_camera * ScalarPoint3f(0.f);
        m_dy = m_sample_to_camera * ScalarPoint3f(0.f, 1.f / m_resolution.y(), 0.f)
             - m_sample_to_camera * ScalarPoint3f(0.f);

        /* Precompute some data for importance(). Please
           look at that function for further details. */
        ScalarPoint3f pmin(m_sample_to_camera * ScalarPoint3f(0.f, 0.f, 0.f)),
                      pmax(m_sample_to_camera * ScalarPoint3f(1.f, 1.f, 0.f));

        m_image_rect.reset();
        m_image_rect.expand(ScalarPoint2f(pmin.x(), pmin.y()) / pmin.z());
        m_image_rect.expand(ScalarPoint2f(pmax.x(), pmax.y()) / pmax.z());
        m_normalization = 1.f / m_image_rect.volume();
        m_needs_sample_3 = false;
    }

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Sampling methods (Sensor interface)
    // =============================================================

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &position_sample,
                                          const Point2f & /*aperture_sample*/,
                                          Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        auto [wavelengths, wav_weight] = sample_wavelength<Float, Spectrum>(wavelength_sample);
        Ray3f ray;
        ray.time = time;
        ray.wavelengths = wavelengths;

        // Compute the sample position on the near plane (local camera space).
        Point3f near_p = m_sample_to_camera *
                         Point3f(position_sample.x(), position_sample.y(), 0.f);

        // Convert into a normalized ray direction; adjust the ray interval accordingly.
        Vector3f d = normalize(Vector3f(near_p));

        Float inv_z = rcp(d.z());
        ray.mint = m_near_clip * inv_z;
        ray.maxt = m_far_clip * inv_z;

        auto trafo = m_world_transform->eval(ray.time, active);
        ray.o = trafo.translation();
        ray.d = trafo * d;
        ray.update();

        return std::make_pair(ray, wav_weight);
    }

    std::pair<RayDifferential3f, Spectrum>
    sample_ray_differential(Float time, Float wavelength_sample, const Point2f &position_sample,
                            const Point2f & /*aperture_sample*/, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        auto [wavelengths, wav_weight] = sample_wavelength<Float, Spectrum>(wavelength_sample);
        RayDifferential3f ray;
        ray.time = time;
        ray.wavelengths = wavelengths;

        // Compute the sample position on the near plane (local camera space).
        Point3f near_p = m_sample_to_camera *
                         Point3f(position_sample.x(), position_sample.y(), 0.f);

        // Convert into a normalized ray direction; adjust the ray interval accordingly.
        Vector3f d = normalize(Vector3f(near_p));
        Float inv_z = rcp(d.z());
        ray.mint = m_near_clip * inv_z;
        ray.maxt = m_far_clip * inv_z;

        auto trafo = m_world_transform->eval(ray.time, active);
        ray.o = trafo.transform_affine(Point3f(0.f));
        ray.d = trafo * d;
        ray.update();

        ray.o_x = ray.o_y = ray.o;

        ray.d_x = trafo * normalize(Vector3f(near_p) + m_dx);
        ray.d_y = trafo * normalize(Vector3f(near_p) + m_dy);
        ray.has_differentials = true;

        return std::make_pair(ray, wav_weight);
    }

    ScalarBoundingBox3f bbox() const override {
        return m_world_transform->translation_bounds();
    }

    /**
     * \brief Compute the directional sensor response function of the camera
     * multiplied with the cosine foreshortening factor associated with the
     * image plane
     *
     * \param d
     *     A normalized direction vector from the aperture position to the
     *     reference point in question (all in local camera space)
     */
    Float importance(const Vector3f &d) const {
        /* How is this derived? Imagine a hypothetical image plane at a
           distance of d=1 away from the pinhole in camera space.

           Then the visible rectangular portion of the plane has the area

              A = (2 * tan(0.5 * xfov in radians))^2 / aspect

           Since we allow crop regions, the actual visible area is
           potentially reduced:

              A' = A * (cropX / filmX) * (cropY / filmY)

           Perspective transformations of such aligned rectangles produce
           an equivalent scaled (but otherwise undistorted) rectangle
           in screen space. This means that a strategy, which uniformly
           generates samples in screen space has an associated area
           density of 1/A' on this rectangle.

           To compute the solid angle density of a sampled point P on
           the rectangle, we can apply the usual measure conversion term:

              d_omega = 1/A' * distance(P, origin)^2 / cos(theta)

           where theta is the angle that the unit direction vector from
           the origin to P makes with the rectangle. Since

              distance(P, origin)^2 = Px^2 + Py^2 + 1

           and

              cos(theta) = 1/sqrt(Px^2 + Py^2 + 1),

           we have

              d_omega = 1 / (A' * cos^3(theta))
        */

        Float ct = Frame3f::cos_theta(d), inv_ct = rcp(ct);

        // Compute the position on the plane at distance 1
        Point2f p(d.x() * inv_ct, d.y() * inv_ct);

        /* Check if the point lies to the front and inside the
           chosen crop rectangle */
        Mask valid = ct > 0 && m_image_rect.contains(p);

        return select(valid, m_normalization * inv_ct * inv_ct * inv_ct, 0.f);
    }

    //! @}
    // =============================================================

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        // TODO x_fov
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        Base::parameters_changed(keys);
        // TODO
    }

    std::string to_string() const override {
        using string::indent;

        std::ostringstream oss;
        oss << "PerspectiveCamera[" << std::endl
            << "  fx = " << m_fx << "," << std::endl
            << "  fy = " << m_fy << "," << std::endl
            << "  cx = " << m_cx << "," << std::endl
            << "  cy = " << m_cy << "," << std::endl
            << "  near_clip = " << m_near_clip << "," << std::endl
            << "  far_clip = " << m_far_clip << "," << std::endl
            << "  film = " << indent(m_film) << "," << std::endl
            << "  sampler = " << indent(m_sampler) << "," << std::endl
            << "  resolution = " << m_resolution << "," << std::endl
            << "  shutter_open = " << m_shutter_open << "," << std::endl
            << "  shutter_open_time = " << m_shutter_open_time << "," << std::endl
            << "  world_transform = " << indent(m_world_transform) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ScalarTransform4f m_camera_to_sample;
    ScalarTransform4f m_sample_to_camera;
    ScalarBoundingBox2f m_image_rect;
    ScalarFloat m_normalization;
    ScalarFloat m_fx;
    ScalarFloat m_fy;
    ScalarFloat m_cx;
    ScalarFloat m_cy;
    ScalarVector3f m_dx, m_dy;
};

MTS_IMPLEMENT_CLASS_VARIANT(OpenCVCamera, ProjectiveCamera)
MTS_EXPORT_PLUGIN(OpenCVCamera, "OpenCV Camera");
NAMESPACE_END(mitsuba)
