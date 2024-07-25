#include <incflo.H>

using namespace amrex;

void incflo::column_collapse_granular (Box const& vbx, Box const& nbx,
                                 Array4<Real> const& density,
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
        } else {
            density(i,j,k) = rho_1;
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
