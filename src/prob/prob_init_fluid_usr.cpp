#include <incflo.H>

using namespace amrex;

void incflo::column_collapse_granular (Box const& vbx, Box const& nbx,
                                 Array4<Real> const& density,
                                 Array4<Real> const& tracer,
                                 Array4<Real> const& pressure,
                                 Box const& /*domain*/,
                                 GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                 GpuArray<Real, AMREX_SPACEDIM> const& problo,
                                 GpuArray<Real, AMREX_SPACEDIM> const& probhi) const
{
    amrex::ignore_unused<Box>(nbx);
    amrex::ignore_unused<Array4<Real>>(pressure);
    // Ensure it is set to two_fluid
    if (!m_two_fluid) amrex::Abort("probtype 531 involves two fluids");

    Vector<Real> granlen_vec{AMREX_D_DECL(Real(0.),Real(0.),Real(0.))};
    ParmParse pp;
    pp.getarr("granular_length",granlen_vec,0,AMREX_SPACEDIM);
    GpuArray<Real,AMREX_SPACEDIM> granLen{AMREX_D_DECL(
        granlen_vec[0],granlen_vec[1],granlen_vec[2])};
    granLen[0] += problo[0];
    granLen[1] += problo[1];
    if (granLen[0] > probhi[0])
        amrex::Abort("Granular length along x is larger than setup");
    if (granLen[1] > probhi[1])
        amrex::Abort("Granular length along y is larger than setup");
#if (AMREX_SPACEDIM==3)
    granLen[2] += problo[2];
    if (granLen[2] > probhi[2])
        amrex::Abort("Granular length along z is larger than setup");
#endif
    Real rho_1 = m_ro_0; Real rho_2 = m_ro_0_second;
    if (rho_1 > rho_2)
        amrex::Abort("Primary fluid must be lighter than second fluid");
    // Density
    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real x = problo[0] + (Real(i)+Real(0.5))*dx[0];
        Real y = problo[1] + (Real(j)+Real(0.5))*dx[1];
#if (AMREX_SPACEDIM == 3)
        Real z = problo[2] + (Real(k)+Real(0.5))*dx[2];
#endif
        if ( (x <= granLen[0]) and (y <= granLen[1])
#if (AMREX_SPACEDIM == 3)
            and (z <= granLen[2])
#endif
           ) {
            density(i,j,k) = rho_2;
            tracer(i,j,k,0) = Real(1.0);
        } else {
            density(i,j,k) = rho_1;
            tracer(i,j,k,0) = Real(0.0);
        }
    });

    if (m_initial_iterations == 0 )
        amrex::Abort("Include non-zero initial_iterations as pressure is not set");
    /*
    // Pressure
    GpuArray<Real,3> grav{0.,0.,0.};
    if (m_gravity[0]*m_gravity[0] + m_gravity[1]*m_gravity[1]+
        m_gravity[2]*m_gravity[2] > Real(0.)) {
        grav[0] = std::abs(m_gravity[0]);
        grav[1] = std::abs(m_gravity[1]);
        grav[2] = std::abs(m_gravity[2]);
    } else if (m_gp0[0]*m_gp0[0] + m_gp0[1]*m_gp0[1]+
        m_gp0[2]*m_gp0[2] > Real(0.)) {
        grav[0] = std::abs(m_gp0[0]);
        grav[1] = std::abs(m_gp0[1]);
        grav[2] = std::abs(m_gp0[2]);
    }
    else {
        amrex::Abort("Body force required in probtype 531");
    }

    amrex::ParallelFor(nbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        // Taking pressure based on the last dimension
        Real x = problo[0] + Real(i)*dx[0];
        Real y = problo[1] + Real(j)*dx[1];
        Real p_h = y;
#if (AMREX_SPACEDIM == 3)
        Real z = problo[2] + Real(k)*dx[2];
        p_h = z;
#endif
        int p_dim = AMREX_SPACEDIM - 1;
        if ( (x <= granLen[0]) and (y <= granLen[1])
#if (AMREX_SPACEDIM == 3)
            and (z <= granLen[2])
#endif
           ) {
            pressure(i,j,k) = rho_1*grav[p_dim]*(probhi[p_dim]-granLen[p_dim])
                              + rho_2*grav[p_dim]*(granLen[p_dim]-p_h);
        } else {
            pressure(i,j,k) = rho_1*grav[p_dim]*(probhi[p_dim]-p_h);
        }
    });
    */
}

void incflo::smooth_column_collapse_granular (Box const& vbx, Box const& nbx,
                                 Array4<Real> const& density,
                                 Array4<Real> const& tracer,
                                 Array4<Real> const& pressure,
                                 Box const& /*domain*/,
                                 GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                 GpuArray<Real, AMREX_SPACEDIM> const& problo,
                                 GpuArray<Real, AMREX_SPACEDIM> const& probhi,
                                 Real smoothing_factor) const
{
    amrex::ignore_unused<Box>(nbx);
    amrex::ignore_unused<Array4<Real>>(pressure);
    // Ensure it is set to two_fluid
    if (!m_two_fluid) amrex::Abort("probtype 532 involves two fluids");

    Vector<Real> granlen_vec{AMREX_D_DECL(Real(0.),Real(0.),Real(0.))};
    ParmParse pp;
    pp.getarr("granular_length",granlen_vec,0,AMREX_SPACEDIM);
    GpuArray<Real,AMREX_SPACEDIM> granLen{AMREX_D_DECL(
        granlen_vec[0],granlen_vec[1],granlen_vec[2])};
    granLen[0] += problo[0];
    granLen[1] += problo[1];
    if (granLen[0] > probhi[0])
        amrex::Abort("Granular length along x is larger than setup");
    if (granLen[1] > probhi[1])
        amrex::Abort("Granular length along y is larger than setup");
#if (AMREX_SPACEDIM==3)
    granLen[2] += problo[2];
    if (granLen[2] > probhi[2])
        amrex::Abort("Granular length along z is larger than setup");
    if (granLen[1] != probhi[1])
        amrex::Abort(
        "For 3D case, domain length and granular_length along y need to be same");
#endif
    Real rho_1 = m_ro_0; Real rho_2 = m_ro_0_second;
    if (rho_1 > rho_2)
        amrex::Abort("Primary fluid must be lighter than second fluid");
    // Density
    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real x = problo[0] + (Real(i)+Real(0.5))*dx[0];
#if (AMREX_SPACEDIM == 2)
        Real y = problo[1] + (Real(j)+Real(0.5))*dx[1];
        density(i,j,k) = std::tanh((granLen[1]-y)/(smoothing_factor*dx[1]))
                            + Real(1.0);
#else
        Real z = problo[2] + (Real(k)+Real(0.5))*dx[2];
        density(i,j,k) = std::tanh((granLen[2]-z)/(smoothing_factor*dx[2]))
                            + Real(1.0);

#endif
        density(i,j,k) *= (std::tanh((granLen[0]-x)/(smoothing_factor*dx[0]))
                            + Real(1.0));
        // Tracer is 1 in heavier fluid
        tracer(i,j,k) = density(i,j,k)*Real(0.25);
        density(i,j,k) *= Real(0.25)*(rho_2-rho_1);
        density(i,j,k) += rho_1;
    });

    if (m_initial_iterations == 0 )
        amrex::Abort("Include non-zero initial_iterations as pressure is not set");
    /*
    // Pressure
    GpuArray<Real,3> grav{0.,0.,0.};
    if (m_gravity[0]*m_gravity[0] + m_gravity[1]*m_gravity[1]+
        m_gravity[2]*m_gravity[2] > Real(0.)) {
        grav[0] = std::abs(m_gravity[0]);
        grav[1] = std::abs(m_gravity[1]);
        grav[2] = std::abs(m_gravity[2]);
    } else if (m_gp0[0]*m_gp0[0] + m_gp0[1]*m_gp0[1]+
        m_gp0[2]*m_gp0[2] > Real(0.)) {
        grav[0] = std::abs(m_gp0[0]);
        grav[1] = std::abs(m_gp0[1]);
        grav[2] = std::abs(m_gp0[2]);
    }
    else {
        amrex::Abort("Body force required in probtype 531");
    }

    amrex::ParallelFor(nbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        // Taking pressure based on the last dimension
        Real x = problo[0] + Real(i)*dx[0];
        Real y = problo[1] + Real(j)*dx[1];
        Real p_grav = grav[1];
        Real p_h = y;
        Real p_h_max = probhi[1];
        Real p_h_gran = granLen[1];
        // This is a multiplication factor along gravity
        Real p_fctr = smoothing_factor*dx[1];
#if (AMREX_SPACEDIM == 3)
        Real z = problo[2] + Real(k)*dx[2];
        p_grav = grav[2];
        p_h = z;
        p_h_max = probhi[2];
        p_h_gran = granLen[2];
        p_fctr = smoothing_factor*dx[2];
#endif
        Real fctr_x = std::tanh((granLen[0]-x)/(smoothing_factor*dx[0]))
                        + Real(1.0);
        fctr_x *= ((rho_2-rho_1)*p_grav*Real(0.25));

        pressure(i,j,k) = std::cosh((p_h_gran-p_h)/p_fctr);
        pressure(i,j,k) /= std::cosh((p_h_gran-p_h_max)/p_fctr);
        pressure(i,j,k) = std::log(pressure(i,j,k));
        pressure(i,j,k) *= (fctr_x*p_fctr);
        pressure(i,j,k) += (fctr_x*(p_h_max-p_h));
        pressure(i,j,k) += (rho_1*p_grav*(p_h_max-p_h));
    });
    */
}

#if (AMREX_SPACEDIM == 3)
void incflo::cylinder_collapse_granular (Box const& vbx, Box const& nbx,
                                 Array4<Real> const& density,
                                 Array4<Real> const& pressure,
                                 Box const& /*domain*/,
                                 GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                 GpuArray<Real, AMREX_SPACEDIM> const& problo,
                                 GpuArray<Real, AMREX_SPACEDIM> const& probhi) const
{
    amrex::ignore_unused<Box>(nbx);
    amrex::ignore_unused<Array4<Real>>(pressure);
    amrex::ignore_unused<GpuArray<Real,AMREX_SPACEDIM>>(probhi);
    // Ensure it is set to two_fluid
    if (!m_two_fluid) amrex::Abort("probtype 533 involves two fluids");

    Real cyl_rad, cyl_h;
    Vector<Real> cyl_center{Real(0.),Real(0.),Real(0.)};
    Vector<Real> cyl_axis{Real(0.),Real(0.),Real(0.)};
    ParmParse pp;
    pp.get("cylinder_radius",cyl_rad);
    pp.get("cylinder_height",cyl_h);
    pp.getarr("cylinder_center",cyl_center,0,3);
    pp.getarr("cylinder_axis",cyl_axis,0,3);

    Real axis_mag = cyl_axis[0]*cyl_axis[0] + cyl_axis[1]*cyl_axis[1] +
                    cyl_axis[2]*cyl_axis[2];
    axis_mag = std::sqrt(axis_mag);
    AMREX_ALWAYS_ASSERT(axis_mag > Real(0.));

    GpuArray<Real,3> cyl_ctr{cyl_center[0],cyl_center[1],cyl_center[2]};
    GpuArray<Real,3> cyl_axs{cyl_axis[0]/axis_mag, cyl_axis[1]/axis_mag,
                             cyl_axis[2]/axis_mag};

    Real rho_1 = m_ro_0; Real rho_2 = m_ro_0_second;
    // Density
    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real x = problo[0] + (Real(i)+Real(0.5))*dx[0];
        x -= cyl_ctr[0];
        Real y = problo[1] + (Real(j)+Real(0.5))*dx[1];
        y -= cyl_ctr[1];
        Real z = problo[2] + (Real(k)+Real(0.5))*dx[2];
        z -= cyl_ctr[2];

        // Distance along axial direction
        Real r_nrml = x*cyl_axs[0] + y*cyl_axs[1]+ z*cyl_axs[2];
        x -= (r_nrml*cyl_axs[0]);
        y -= (r_nrml*cyl_axs[1]);
        z -= (r_nrml*cyl_axs[2]);
        // Transverse distance
        Real r_trns = x*x + y*y + z*z;
        r_trns = std::sqrt(r_trns);
        r_nrml = std::abs(r_nrml);

        if ((r_nrml <= Real(0.5)*cyl_h) and (r_trns <= cyl_rad))
        {
            density(i,j,k) = rho_2;
        }
        else {
            density(i,j,k) = rho_1;
        }

    });

    if (m_initial_iterations == 0 )
        amrex::Abort("Include non-zero initial_iterations as pressure is not set");
}
#endif

void incflo::single_layer_inclined_plane_granular (Box const& vbx, Box const& nbx,
                                 Array4<Real> const& density,
                                 Array4<Real> const& tracer,
                                 Array4<Real> const& pressure,
                                 Array4<Real> const& velocity,
                                 Box const& /*domain*/,
                                 GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                 GpuArray<Real, AMREX_SPACEDIM> const& problo,
                                 GpuArray<Real, AMREX_SPACEDIM> const& probhi) const
{
    amrex::ignore_unused<Box>(nbx);
    amrex::ignore_unused<Array4<Real>>(pressure);
    // Ensure it is set to two_fluid
    if (!m_two_fluid) amrex::Abort("probtype 535 leverages two_fluid infrastructure");
    Real grav_mag = m_gravity[0]*m_gravity[0] + m_gravity[1]*m_gravity[1]+
                    m_gravity[2]*m_gravity[2];
    grav_mag = std::sqrt(grav_mag);
    if (grav_mag ==  Real(0.)) {
        amrex::Abort("Gravity is required in probtype 535");
    }
    Real zeta_0; // Inclination angle used in initialization
    Real I_zeta_0; // Inertial number corresponding to initialization
    ParmParse pp;
    pp.get("zeta_0", zeta_0);
    pp.get("I_zeta_0",I_zeta_0);
    amrex::Print() << "zeta_0: "<<zeta_0<<" , I_zeta_0: "<< I_zeta_0<<"\n";

    Real rho_2 = m_ro_0_second;
    Real rho_p = m_ro_grain_second; // Granular particle density
    Real diam_p = m_diam_second; // Granular particle mean diameter
    Real H_c = probhi[1]; // Column Height
#if (AMREX_SPACEDIM == 3)
    H_c = probhi[2];
#endif

    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real x = problo[0] + (Real(i)+Real(0.5))*dx[0];
        Real y = problo[1] + (Real(j)+Real(0.5))*dx[1];
#if (AMREX_SPACEDIM == 3)
        Real z = problo[2] + (Real(k)+Real(0.5))*dx[2];
#endif
        density(i,j,k) = rho_2;
        tracer(i,j,k,0) = Real(1.0);
        // Bagnold profile based on zeta_0
        velocity(i,j,k,0) = std::sqrt(grav_mag*std::cos(zeta_0)*rho_2/rho_p);
        velocity(i,j,k,0) *= (Real(4.0)*I_zeta_0/(Real(3.0)*diam_p));
#if (AMREX_SPACEDIM == 2)
        velocity(i,j,k,0) *= (std::pow(H_c,Real(1.5)) - std::pow(H_c-y,Real(1.5)));
#else
        velocity(i,j,k,0) *= (std::pow(H_c,Real(1.5)) - std::pow(H_c-z,Real(1.5)));
#endif
        velocity(i,j,k,1) = Real(0.0);
#if (AMREX_SPACEDIM == 3)
        velocity(i,j,k,2) = Real(0.0);
#endif
    });

    if (m_initial_iterations == 0 )
        amrex::Abort("Include non-zero initial_iterations as pressure is not set");
}

void incflo::double_layer_inclined_plane_granular (Box const& vbx, Box const& nbx,
                                 Array4<Real> const& density,
                                 Array4<Real> const& tracer,
                                 Array4<Real> const& pressure,
                                 Array4<Real> const& velocity,
                                 Box const& /*domain*/,
                                 GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                 GpuArray<Real, AMREX_SPACEDIM> const& problo,
                                 GpuArray<Real, AMREX_SPACEDIM> const& probhi) const
{
    amrex::ignore_unused<Box>(nbx);
    amrex::ignore_unused<Array4<Real>>(pressure);
    // Ensure it is set to two_fluid
    if (!m_two_fluid) amrex::Abort("probtype 536 leverages two_fluid infrastructure");
    Real grav_mag = m_gravity[0]*m_gravity[0] + m_gravity[1]*m_gravity[1]+
                    m_gravity[2]*m_gravity[2];
    grav_mag = std::sqrt(grav_mag);
    if (grav_mag ==  Real(0.)) {
        amrex::Abort("Gravity is required in probtype 536");
    }
    Real zeta_0; // Inclination angle used in initialization
    Real I_zeta_0; // Inertial number corresponding to initialization
    ParmParse pp;
    pp.get("zeta_0", zeta_0);
    pp.get("I_zeta_0",I_zeta_0);
    amrex::Print() << "zeta_0: "<<zeta_0<<" , I_zeta_0: "<< I_zeta_0<<"\n";

    Vector<Real> granlen_vec{AMREX_D_DECL(Real(0.),Real(0.),Real(0.))};
    pp.getarr("granular_length",granlen_vec,0,AMREX_SPACEDIM);
    GpuArray<Real,AMREX_SPACEDIM> granLen{AMREX_D_DECL(
        granlen_vec[0],granlen_vec[1],granlen_vec[2])};
    granLen[0] += problo[0];
    granLen[1] += problo[1];
    if (granLen[0] != probhi[0])
        amrex::Abort("Granular length along x needs to be equal to setup");
#if (AMREX_SPACEDIM == 2)
    if (granLen[1] >= probhi[1])
        amrex::Abort("Granular length along y needs to be smaller than setup");
#else
    if (granLen[1] != probhi[1])
        amrex::Abort("Granular length along y needs to be equal to setup");

    granLen[2] += problo[2];
    if (granLen[2] >= probhi[2])
        amrex::Abort("Granular length along z needs to be smaller than setup");
#endif
    Real rho_1 = m_ro_0; Real rho_2 = m_ro_0_second;
    Real rho_p = m_ro_grain_second; // Granular particle density
    Real diam_p = m_diam_second; // Granular particle mean diameter
    Real H_c = granLen[1]; // Column Height
    Real H_d = probhi[1]; // Domain Height
#if (AMREX_SPACEDIM == 3)
    H_c = granLen[2];
    H_d = probhi[2];
#endif
    // Create a variable to store max Bagnold velocity
    Real u_bag_max = std::sqrt(grav_mag*std::cos(zeta_0)*rho_2/rho_p);
    u_bag_max *= (Real(4.0)*I_zeta_0/(Real(3.0)*diam_p));
    u_bag_max *= std::pow(H_c,Real(1.5));

    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real x = problo[0] + (Real(i)+Real(0.5))*dx[0];
        Real y = problo[1] + (Real(j)+Real(0.5))*dx[1];
#if (AMREX_SPACEDIM == 3)
        Real z = problo[2] + (Real(k)+Real(0.5))*dx[2];
#endif
        if ( (x <= granLen[0]) and (y <= granLen[1])
#if (AMREX_SPACEDIM == 3)
            and (z <= granLen[2])
#endif
           ) {
            density(i,j,k) = rho_2;
            tracer(i,j,k,0) = Real(1.0);
            // Bagnold profile based on zeta_0
            velocity(i,j,k,0) = std::sqrt(grav_mag*std::cos(zeta_0)*rho_2/rho_p);
            velocity(i,j,k,0) *= (Real(4.0)*I_zeta_0/(Real(3.0)*diam_p));
#if (AMREX_SPACEDIM == 2)
            velocity(i,j,k,0) *= (std::pow(H_c,Real(1.5)) - std::pow(H_c-y,Real(1.5)));
#else
            velocity(i,j,k,0) *= (std::pow(H_c,Real(1.5)) - std::pow(H_c-z,Real(1.5)));
#endif
        } else {
            density(i,j,k) = rho_1;
            tracer(i,j,k,0) = Real(0.0);
            // Linear profile for lighter fluid
#if (AMREX_SPACEDIM == 2)
            velocity(i,j,k,0) = (H_d - y)/(H_d - H_c);
#else
            velocity(i,j,k,0) = (H_d - z)/(H_d - H_c);
#endif
            velocity(i,j,k,0) *= u_bag_max;
        }

        velocity(i,j,k,1) = Real(0.0);
#if (AMREX_SPACEDIM == 3)
        velocity(i,j,k,2) = Real(0.0);
#endif
    });

    if (m_initial_iterations == 0 )
        amrex::Abort("Include non-zero initial_iterations as pressure is not set");
}

void incflo::smooth_double_layer_inclined_plane_granular (Box const& vbx, Box const& nbx,
                                 Array4<Real> const& density,
                                 Array4<Real> const& tracer,
                                 Array4<Real> const& pressure,
                                 Array4<Real> const& velocity,
                                 Box const& /*domain*/,
                                 GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                 GpuArray<Real, AMREX_SPACEDIM> const& problo,
                                 GpuArray<Real, AMREX_SPACEDIM> const& probhi) const
{
    amrex::ignore_unused<Box>(nbx);
    amrex::ignore_unused<Array4<Real>>(pressure);
    // Ensure it is set to two_fluid
    if (!m_two_fluid) amrex::Abort("probtype 537 requires two_fluid");
    Real grav_mag = m_gravity[0]*m_gravity[0] + m_gravity[1]*m_gravity[1]+
                    m_gravity[2]*m_gravity[2];
    grav_mag = std::sqrt(grav_mag);
    if (grav_mag ==  Real(0.)) {
        amrex::Abort("Gravity is required in probtype 537");
    }
    Real zeta_0; // Inclination angle used in initialization
    Real I_zeta_0; // Inertial number corresponding to initialization
    ParmParse pp;
    pp.get("zeta_0", zeta_0);
    pp.get("I_zeta_0",I_zeta_0);
    amrex::Print() << "zeta_0: "<<zeta_0<<" , I_zeta_0: "<< I_zeta_0<<"\n";

    Real smoothing_factor;
    pp.get("smoothing_factor",smoothing_factor);
    Vector<Real> granlen_vec{AMREX_D_DECL(Real(0.),Real(0.),Real(0.))};
    pp.getarr("granular_length",granlen_vec,0,AMREX_SPACEDIM);
    GpuArray<Real,AMREX_SPACEDIM> granLen{AMREX_D_DECL(
        granlen_vec[0],granlen_vec[1],granlen_vec[2])};
    granLen[0] += problo[0];
    granLen[1] += problo[1];
    if (granLen[0] != probhi[0])
        amrex::Abort("Granular length along x needs to be equal to setup");
#if (AMREX_SPACEDIM == 2)
    if (granLen[1] >= probhi[1])
        amrex::Abort("Granular length along y needs to be smaller than setup");
#else
    if (granLen[1] != probhi[1])
        amrex::Abort("Granular length along y needs to be equal to setup");

    granLen[2] += problo[2];
    if (granLen[2] >= probhi[2])
        amrex::Abort("Granular length along z needs to be smaller than setup");
#endif
    Real rho_1 = m_ro_0; Real rho_2 = m_ro_0_second;
    Real rho_p = m_ro_grain_second; // Granular particle density
    Real diam_p = m_diam_second; // Granular particle mean diameter
    Real H_c = granLen[1]; // Column Height
    Real H_d = probhi[1]; // Domain Height
#if (AMREX_SPACEDIM == 3)
    H_c = granLen[2];
    H_d = probhi[2];
#endif
    // Create a variable to store max Bagnold velocity
    Real u_bag_max = std::sqrt(grav_mag*std::cos(zeta_0)*rho_2/rho_p);
    u_bag_max *= (Real(4.0)*I_zeta_0/(Real(3.0)*diam_p));
    u_bag_max *= std::pow(H_c,Real(1.5));

    bool init_vel = true;
    pp.query("initialize_velocity", init_vel);

    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        // Tracer is 1 in heavier fluid
        Real x = problo[0] + (Real(i)+Real(0.5))*dx[0];
        Real y = problo[1] + (Real(j)+Real(0.5))*dx[1];
#if (AMREX_SPACEDIM == 2)
        tracer(i,j,k,0) = std::tanh((granLen[1]-y)/(smoothing_factor*dx[1]))
                            + Real(1.0);
#else
        Real z = problo[2] + (Real(k)+Real(0.5))*dx[2];
        tracer(i,j,k,0) = std::tanh((granLen[2]-z)/(smoothing_factor*dx[2]))
                            + Real(1.0);

#endif
        tracer(i,j,k,0) *= Real(0.5);
        density(i,j,k)  = tracer(i,j,k,0)*(rho_2-rho_1);
        density(i,j,k)  += rho_1;

        if (init_vel) {
           if ( (x <= granLen[0]) and (y <= granLen[1])
#if (AMREX_SPACEDIM == 3)
               and (z <= granLen[2])
#endif
              ) {
                // Bagnold profile based on zeta_0
                velocity(i,j,k,0) = std::sqrt(grav_mag*std::cos(zeta_0)*rho_2/rho_p);
                velocity(i,j,k,0) *= (Real(4.0)*I_zeta_0/(Real(3.0)*diam_p));
#if (AMREX_SPACEDIM == 2)
                velocity(i,j,k,0) *= (std::pow(H_c,Real(1.5)) - std::pow(H_c-y,Real(1.5)));
#else
                velocity(i,j,k,0) *= (std::pow(H_c,Real(1.5)) - std::pow(H_c-z,Real(1.5)));
#endif
           } else {
               // Linear profile for lighter fluid
#if (AMREX_SPACEDIM == 2)
               velocity(i,j,k,0) = (H_d - y)/(H_d - H_c);
#else
               velocity(i,j,k,0) = (H_d - z)/(H_d - H_c);
#endif
               velocity(i,j,k,0) *= u_bag_max;
           }
        } else {
           velocity(i,j,k,0) = Real(0.0);
        }
        velocity(i,j,k,1) = Real(0.0);
#if (AMREX_SPACEDIM == 3)
        velocity(i,j,k,2) = Real(0.0);
#endif
    });

    if (m_initial_iterations == 0 )
        amrex::Abort("Include non-zero initial_iterations as pressure is not set");
}

void incflo::initialize_entire_domain_with_second_fluid (Box const& vbx,
                                   Array4<Real> const& density,
                                   Array4<Real> const& tracer) const
{
    // Ensure it is set to two_fluid
    if (!m_two_fluid) amrex::Abort("probtype 538 requires two_fluid");
    Real rho_2 = m_ro_0_second;
    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        tracer(i,j,k,0) = Real(1.0);
        density(i,j,k)  = rho_2;
    });
}

void incflo::init_taylor_couette (Box const& vbx,
                                  Array4<Real> const& velocity,
                                  GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                  GpuArray<Real, AMREX_SPACEDIM> const& problo) const
{
     if (!m_eb_flow.has_rotation) {
         amrex::Abort("This probtype needs rotation\n");
     }
     Real l_rf_omega = m_eb_flow.omega_mag;
     GpuArray<Real,3> l_rf_axis{m_eb_flow.omega_unit_vec[0],
                                m_eb_flow.omega_unit_vec[1],
                                m_eb_flow.omega_unit_vec[2]};

     GpuArray<Real,3> l_rf_center{m_eb_flow.rotation_center[0],
                                  m_eb_flow.rotation_center[1],
                                  m_eb_flow.rotation_center[2]};
     // Hard-coded values
     Real l_r1 = 0.30; Real l_r2 = 0.375;
     Real l_eta = l_r1/l_r2;
     Real l_A = -l_rf_omega*(l_eta*l_eta/(1.0-l_eta*l_eta));
     Real l_B = (l_rf_omega*l_r1*l_r1)/(1.0-l_eta*l_eta);

     amrex::ParallelFor(vbx, [=]
         AMREX_GPU_DEVICE (int i, int j, int k) noexcept
         {
           Real l_x = problo[0] + (i+0.5)*dx[0] - l_rf_center[0];
           Real l_y = problo[1] + (j+0.5)*dx[1] - l_rf_center[1];
           Real l_z = problo[2] + (k+0.5)*dx[2] - l_rf_center[2];
           // Get projected radius
           Real r_x = l_rf_axis[1]*l_z - l_rf_axis[2]*l_y;
           Real r_y = l_rf_axis[2]*l_x - l_rf_axis[0]*l_z;
           Real r_z = l_rf_axis[0]*l_y - l_rf_axis[1]*l_x;
           Real r_mag = std::sqrt(r_x*r_x + r_y*r_y + r_z*r_z);

           if (r_mag > 0.) {
             Real vel_mag = l_A*r_mag + l_B/r_mag;
             velocity(i,j,k,0) = (vel_mag/r_mag)*(l_rf_axis[1]*l_z - l_rf_axis[2]*l_y);
             velocity(i,j,k,1) = (vel_mag/r_mag)*(l_rf_axis[2]*l_x - l_rf_axis[0]*l_z);
             velocity(i,j,k,2) = (vel_mag/r_mag)*(l_rf_axis[0]*l_y - l_rf_axis[1]*l_x);
           }
         });
}
