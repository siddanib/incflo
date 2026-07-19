#include <incflo.H>

using namespace amrex;

void incflo::update_velocity (StepType step_type, Vector<MultiFab>& vel_eta, Vector<MultiFab>& vel_forces)
{
    BL_PROFILE("incflo::update_velocity");

    // Related to modified timestepping. Better code structure needed
    Vector<MultiFab> timestepping_alpha, timestepping_divtau_o;
    if (m_gran_rheo_modified_time_stepping) {
        for (int lev=0; lev<= finest_level; lev++) {
            auto const& divtau_o = m_leveldata[lev]->divtau_o;
            timestepping_divtau_o.emplace_back(divtau_o.boxArray(),
                divtau_o.DistributionMap(), divtau_o.nComp(),
                divtau_o.nGrow(), MFInfo(), divtau_o.Factory());

            timestepping_alpha.emplace_back(vel_eta[lev].boxArray(),
                vel_eta[lev].DistributionMap(), vel_eta[lev].nComp(),
                vel_eta[lev].nGrow(), MFInfo(), vel_eta[lev].Factory());

            timestepping_divtau_o[lev].setVal(Real(0.));
            timestepping_alpha[lev].setVal(Real(0.));
            // Choosing alpha based on eta_1
            MultiFab::Saxpy(timestepping_alpha[lev],
                m_modified_time_stepping_constant, vel_eta[lev],
                0, 0, vel_eta[lev].nComp(), vel_eta[lev].nGrow());
        }
        compute_divtau(GetVecOfPtrs(timestepping_divtau_o),
                       get_velocity_old_const(), get_density_old_const(),
                       GetVecOfConstPtrs(timestepping_alpha));
    }

    Real new_time = m_cur_time + m_dt;

    Real l_dt   = m_dt;
    Real l_half = Real(0.5);

    if (step_type == StepType::Predictor) {

        // *************************************************************************************
        // Define (or if advection_type != "MOL", re-define) the forcing terms, without the viscous terms
        //    and using the half-time density
        // *************************************************************************************
        compute_vel_forces(GetVecOfPtrs(vel_forces), get_velocity_old_const(),
                           get_density_nph_const(), get_tracer_old_const(), get_tracer_new_const());

        // *************************************************************************************
        // Update the velocity
        // *************************************************************************************
        for (int lev = 0; lev <= finest_level; lev++)
        {
        auto& ld = *m_leveldata[lev];
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(ld.velocity,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            Array4<Real> const& vel = ld.velocity.array(mfi);
            Array4<Real const> const& dvdt = ld.conv_velocity_o.const_array(mfi);
            Array4<Real const> const& vel_f = vel_forces[lev].const_array(mfi);
            Array4<Real const> const& rho_old  = ld.density_o.const_array(mfi);
            Array4<Real const> const& rho_new  = ld.density.const_array(mfi);
            Array4<Real const> const& rho_nph  = ld.density_nph.const_array(mfi);

            if (m_diff_type == DiffusionType::Implicit
                && (!m_gran_rheo_modified_time_stepping)) {

                if (use_tensor_correction)
                {
                    Array4<Real const> const& divtau_o = ld.divtau_o.const_array(mfi);
                    // Here divtau_o is the difference of tensor and scalar divtau_o!
                    if (m_advect_momentum) {
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            AMREX_D_TERM(vel(i,j,k,0) *= rho_old(i,j,k);,
                                         vel(i,j,k,1) *= rho_old(i,j,k);,
                                         vel(i,j,k,2) *= rho_old(i,j,k););
                            AMREX_D_TERM(vel(i,j,k,0) += l_dt*(dvdt(i,j,k,0)+rho_nph(i,j,k)*vel_f(i,j,k,0)+divtau_o(i,j,k,0));,
                                         vel(i,j,k,1) += l_dt*(dvdt(i,j,k,1)+rho_nph(i,j,k)*vel_f(i,j,k,1)+divtau_o(i,j,k,1));,
                                         vel(i,j,k,2) += l_dt*(dvdt(i,j,k,2)+rho_nph(i,j,k)*vel_f(i,j,k,2)+divtau_o(i,j,k,2)););
                            AMREX_D_TERM(vel(i,j,k,0) /= rho_new(i,j,k);,
                                         vel(i,j,k,1) /= rho_new(i,j,k);,
                                         vel(i,j,k,2) /= rho_new(i,j,k););
                        });
                    } else {
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            AMREX_D_TERM(vel(i,j,k,0) += l_dt*(dvdt(i,j,k,0)+vel_f(i,j,k,0)+divtau_o(i,j,k,0));,
                                         vel(i,j,k,1) += l_dt*(dvdt(i,j,k,1)+vel_f(i,j,k,1)+divtau_o(i,j,k,1));,
                                         vel(i,j,k,2) += l_dt*(dvdt(i,j,k,2)+vel_f(i,j,k,2)+divtau_o(i,j,k,2)););
                        });
                    }
                } else {
                    if (m_advect_momentum) {
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            AMREX_D_TERM(vel(i,j,k,0) *= rho_old(i,j,k);,
                                         vel(i,j,k,1) *= rho_old(i,j,k);,
                                         vel(i,j,k,2) *= rho_old(i,j,k););
                            AMREX_D_TERM(vel(i,j,k,0) += l_dt*(dvdt(i,j,k,0)+rho_nph(i,j,k)*vel_f(i,j,k,0));,
                                         vel(i,j,k,1) += l_dt*(dvdt(i,j,k,1)+rho_nph(i,j,k)*vel_f(i,j,k,1));,
                                         vel(i,j,k,2) += l_dt*(dvdt(i,j,k,2)+rho_nph(i,j,k)*vel_f(i,j,k,2)););
                            AMREX_D_TERM(vel(i,j,k,0) /= rho_new(i,j,k);,
                                         vel(i,j,k,1) /= rho_new(i,j,k);,
                                         vel(i,j,k,2) /= rho_new(i,j,k););
                        });
                    } else {
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            AMREX_D_TERM(vel(i,j,k,0) += l_dt*(dvdt(i,j,k,0)+vel_f(i,j,k,0));,
                                         vel(i,j,k,1) += l_dt*(dvdt(i,j,k,1)+vel_f(i,j,k,1));,
                                         vel(i,j,k,2) += l_dt*(dvdt(i,j,k,2)+vel_f(i,j,k,2)););
                        });
                    }
                }
            }
            else if (m_diff_type == DiffusionType::Crank_Nicolson
                     && (!m_gran_rheo_modified_time_stepping))
            {

                Array4<Real const> const& divtau_o = ld.divtau_o.const_array(mfi);
                if (m_advect_momentum) {
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        AMREX_D_TERM(vel(i,j,k,0) *= rho_old(i,j,k);,
                                     vel(i,j,k,1) *= rho_old(i,j,k);,
                                     vel(i,j,k,2) *= rho_old(i,j,k););
                        AMREX_D_TERM(vel(i,j,k,0) += l_dt*(dvdt(i,j,k,0)+rho_nph(i,j,k)*vel_f(i,j,k,0)+l_half*divtau_o(i,j,k,0));,
                                     vel(i,j,k,1) += l_dt*(dvdt(i,j,k,1)+rho_nph(i,j,k)*vel_f(i,j,k,1)+l_half*divtau_o(i,j,k,1));,
                                     vel(i,j,k,2) += l_dt*(dvdt(i,j,k,2)+rho_nph(i,j,k)*vel_f(i,j,k,2)+l_half*divtau_o(i,j,k,2)););
                        AMREX_D_TERM(vel(i,j,k,0) /= rho_new(i,j,k);,
                                     vel(i,j,k,1) /= rho_new(i,j,k);,
                                     vel(i,j,k,2) /= rho_new(i,j,k););
                    });
                } else {
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        AMREX_D_TERM(vel(i,j,k,0) += l_dt*(dvdt(i,j,k,0)+vel_f(i,j,k,0)+l_half*divtau_o(i,j,k,0));,
                                     vel(i,j,k,1) += l_dt*(dvdt(i,j,k,1)+vel_f(i,j,k,1)+l_half*divtau_o(i,j,k,1));,
                                     vel(i,j,k,2) += l_dt*(dvdt(i,j,k,2)+vel_f(i,j,k,2)+l_half*divtau_o(i,j,k,2)););
                    });
                }
            }
            else if (velocity_uses_explicit_diffusion_terms())
            {
                Array4<Real const> const& divtau_o = ld.divtau_o.const_array(mfi);
                if (m_advect_momentum) {
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        AMREX_D_TERM(vel(i,j,k,0) *= rho_old(i,j,k);,
                                     vel(i,j,k,1) *= rho_old(i,j,k);,
                                     vel(i,j,k,2) *= rho_old(i,j,k););
                        AMREX_D_TERM(vel(i,j,k,0) += l_dt*(dvdt(i,j,k,0)+rho_nph(i,j,k)*vel_f(i,j,k,0)+divtau_o(i,j,k,0));,
                                     vel(i,j,k,1) += l_dt*(dvdt(i,j,k,1)+rho_nph(i,j,k)*vel_f(i,j,k,1)+divtau_o(i,j,k,1));,
                                     vel(i,j,k,2) += l_dt*(dvdt(i,j,k,2)+rho_nph(i,j,k)*vel_f(i,j,k,2)+divtau_o(i,j,k,2)););
                        AMREX_D_TERM(vel(i,j,k,0) /= rho_new(i,j,k);,
                                     vel(i,j,k,1) /= rho_new(i,j,k);,
                                     vel(i,j,k,2) /= rho_new(i,j,k););
                    });
                } else {
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        AMREX_D_TERM(vel(i,j,k,0) += l_dt*(dvdt(i,j,k,0)+vel_f(i,j,k,0)+divtau_o(i,j,k,0));,
                                     vel(i,j,k,1) += l_dt*(dvdt(i,j,k,1)+vel_f(i,j,k,1)+divtau_o(i,j,k,1));,
                                     vel(i,j,k,2) += l_dt*(dvdt(i,j,k,2)+vel_f(i,j,k,2)+divtau_o(i,j,k,2)););
                    });
                }
                if (m_gran_rheo_modified_time_stepping) {
                   Array4<Real const> const& ts_divtau_o = timestepping_divtau_o[lev].const_array(mfi);
                   if (m_advect_momentum) {
                       ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                       {
                           AMREX_D_TERM(vel(i,j,k,0) -= l_dt*(ts_divtau_o(i,j,k,0)/rho_new(i,j,k));,
                                        vel(i,j,k,1) -= l_dt*(ts_divtau_o(i,j,k,1)/rho_new(i,j,k));,
                                        vel(i,j,k,2) -= l_dt*(ts_divtau_o(i,j,k,2)/rho_new(i,j,k)););
                       });
                   } else {
                       ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                       {
                           AMREX_D_TERM(vel(i,j,k,0) -= l_dt*ts_divtau_o(i,j,k,0);,
                                        vel(i,j,k,1) -= l_dt*ts_divtau_o(i,j,k,1);,
                                        vel(i,j,k,2) -= l_dt*ts_divtau_o(i,j,k,2););
                       });
                   }
                }
            }
        } // mfi
        } // lev

    } else if (step_type == StepType::Corrector) {

        // *************************************************************************************
        // Define the forcing terms to use in the final update (using half-time density)
        // *************************************************************************************
        compute_vel_forces(GetVecOfPtrs(vel_forces), get_velocity_new_const(),
                           get_density_nph_const(), get_tracer_old_const(), get_tracer_new_const());

        for (int lev = 0; lev <= finest_level; lev++)
        {
            auto& ld = *m_leveldata[lev];

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(ld.velocity,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            Array4<Real> const& vel = ld.velocity.array(mfi);
            Array4<Real const> const& vel_o = ld.velocity_o.const_array(mfi);
            Array4<Real const> const& dvdt = ld.conv_velocity.const_array(mfi);
            Array4<Real const> const& dvdt_o = ld.conv_velocity_o.const_array(mfi);
            Array4<Real const> const& vel_f = vel_forces[lev].const_array(mfi);

            Array4<Real const> const& rho_old  = ld.density_o.const_array(mfi);
            Array4<Real const> const& rho_new  = ld.density.const_array(mfi);
            Array4<Real const> const& rho_nph  = ld.density_nph.const_array(mfi);

            if (velocity_uses_explicit_diffusion_terms())
            {
                Array4<Real const> const& divtau_o = ld.divtau_o.const_array(mfi);
                Array4<Real const> const& divtau   = ld.divtau.const_array(mfi);

                if (m_advect_momentum) {
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        AMREX_D_TERM(vel(i,j,k,0) = rho_old(i,j,k) * vel_o(i,j,k,0) + l_dt * (
                                                    l_half*(  dvdt_o(i,j,k,0)+  dvdt(i,j,k,0))
                                                  + l_half*(divtau_o(i,j,k,0)+divtau(i,j,k,0))
                                                  + rho_nph(i,j,k) * vel_f(i,j,k,0) );,
                                     vel(i,j,k,1) =  rho_old(i,j,k) * vel_o(i,j,k,1) + l_dt * (
                                                     l_half*(  dvdt_o(i,j,k,1)+  dvdt(i,j,k,1))
                                                   + l_half*(divtau_o(i,j,k,1)+divtau(i,j,k,1))
                                                   + rho_nph(i,j,k) * vel_f(i,j,k,1) );,
                                     vel(i,j,k,2) =  rho_old(i,j,k) * vel_o(i,j,k,2) + l_dt * (
                                                     l_half*(  dvdt_o(i,j,k,2)+  dvdt(i,j,k,2))
                                                   + l_half*(divtau_o(i,j,k,2)+divtau(i,j,k,2))
                                                   + rho_nph(i,j,k) * vel_f(i,j,k,2) ););

                        AMREX_D_TERM(vel(i,j,k,0) /= rho_new(i,j,k);,
                                     vel(i,j,k,1) /= rho_new(i,j,k);,
                                     vel(i,j,k,2) /= rho_new(i,j,k););
                    });
                } else {
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        AMREX_D_TERM(vel(i,j,k,0) = vel_o(i,j,k,0) + l_dt * (
                                                    l_half*(  dvdt_o(i,j,k,0)+  dvdt(i,j,k,0))
                                                  + l_half*(divtau_o(i,j,k,0)+divtau(i,j,k,0))
                                                  + vel_f(i,j,k,0) );,
                                     vel(i,j,k,1) = vel_o(i,j,k,1) + l_dt * (
                                                    l_half*(  dvdt_o(i,j,k,1)+  dvdt(i,j,k,1))
                                                  + l_half*(divtau_o(i,j,k,1)+divtau(i,j,k,1))
                                                  + vel_f(i,j,k,1) );,
                                     vel(i,j,k,2) = vel_o(i,j,k,2) + l_dt * (
                                                    l_half*(  dvdt_o(i,j,k,2)+  dvdt(i,j,k,2))
                                                  + l_half*(divtau_o(i,j,k,2)+divtau(i,j,k,2))
                                                  + vel_f(i,j,k,2) ););
                    });
                }
            }
            else if (m_diff_type == DiffusionType::Crank_Nicolson
                     && (!m_gran_rheo_modified_time_stepping))
            {
                Array4<Real const> const& divtau_o = ld.divtau_o.const_array(mfi);

                if (m_advect_momentum) {
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        AMREX_D_TERM(vel(i,j,k,0) = rho_old(i,j,k) * vel_o(i,j,k,0) + l_dt * (
                                                    l_half*(  dvdt_o(i,j,k,0)+dvdt(i,j,k,0))
                                                   +l_half*(divtau_o(i,j,k,0) ) + rho_nph(i,j,k)*vel_f(i,j,k,0) );,
                                     vel(i,j,k,1) = rho_old(i,j,k) * vel_o(i,j,k,1) + l_dt * (
                                                    l_half*(  dvdt_o(i,j,k,1)+dvdt(i,j,k,1))
                                                   +l_half*(divtau_o(i,j,k,1) ) + rho_nph(i,j,k)*vel_f(i,j,k,1) );,
                                     vel(i,j,k,2) = rho_old(i,j,k) * vel_o(i,j,k,2) + l_dt * (
                                                    l_half*(  dvdt_o(i,j,k,2)+dvdt(i,j,k,2))
                                                   +l_half*(divtau_o(i,j,k,2) ) + rho_nph(i,j,k)*vel_f(i,j,k,2) ););

                        AMREX_D_TERM(vel(i,j,k,0) /= rho_new(i,j,k);,
                                     vel(i,j,k,1) /= rho_new(i,j,k);,
                                     vel(i,j,k,2) /= rho_new(i,j,k););
                    });
                } else {
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        AMREX_D_TERM(vel(i,j,k,0) = vel_o(i,j,k,0) + l_dt * (
                                                    l_half*(  dvdt_o(i,j,k,0)+dvdt(i,j,k,0))
                                                  + l_half* divtau_o(i,j,k,0) + vel_f(i,j,k,0) );,
                                     vel(i,j,k,1) = vel_o(i,j,k,1) + l_dt * (
                                                    l_half*(  dvdt_o(i,j,k,1)+dvdt(i,j,k,1))
                                                  + l_half* divtau_o(i,j,k,1) + vel_f(i,j,k,1) );,
                                     vel(i,j,k,2) = vel_o(i,j,k,2) + l_dt * (
                                                    l_half*(  dvdt_o(i,j,k,2)+dvdt(i,j,k,2))
                                                  + l_half* divtau_o(i,j,k,2) + vel_f(i,j,k,2) ););
                    });
                }
            }
            else if (m_diff_type == DiffusionType::Implicit)
            {
                if (use_tensor_correction)
                {
                    Array4<Real const> const& divtau   = ld.divtau.const_array(mfi);
                    if (m_advect_momentum)
                    {
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            AMREX_D_TERM(vel(i,j,k,0) = rho_old(i,j,k) * vel_o(i,j,k,0) + l_dt * (
                                                        l_half*(dvdt_o(i,j,k,0) +   dvdt(i,j,k,0))
                                                      + rho_nph(i,j,k) * vel_f(i,j,k,0) + divtau(i,j,k,0));,
                                         vel(i,j,k,1) = rho_old(i,j,k) * vel_o(i,j,k,1) + l_dt * (
                                                        l_half*(dvdt_o(i,j,k,1) +   dvdt(i,j,k,1))
                                                      + rho_nph(i,j,k) * vel_f(i,j,k,1) + divtau(i,j,k,1));,
                                         vel(i,j,k,2) = rho_old(i,j,k) * vel_o(i,j,k,2) + l_dt * (
                                                        l_half*(dvdt_o(i,j,k,2) +   dvdt(i,j,k,2))
                                                      + rho_nph(i,j,k) * vel_f(i,j,k,2) + divtau(i,j,k,2)););

                            AMREX_D_TERM(vel(i,j,k,0) /= rho_new(i,j,k);,
                                         vel(i,j,k,1) /= rho_new(i,j,k);,
                                         vel(i,j,k,2) /= rho_new(i,j,k););
                        });
                    } else {
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            AMREX_D_TERM(vel(i,j,k,0) = vel_o(i,j,k,0) + l_dt * (
                                                        l_half*(  dvdt_o(i,j,k,0) +   dvdt(i,j,k,0))
                                                      + vel_f(i,j,k,0) + divtau(i,j,k,0));,
                                         vel(i,j,k,1) = vel_o(i,j,k,1) + l_dt * (
                                                        l_half*(dvdt_o(i,j,k,1) +   dvdt(i,j,k,1))
                                                      + vel_f(i,j,k,1) + divtau(i,j,k,1) );,
                                         vel(i,j,k,2) = vel_o(i,j,k,2) + l_dt * (
                                                        l_half*(  dvdt_o(i,j,k,2) +   dvdt(i,j,k,2))
                                                      + vel_f(i,j,k,2) + divtau(i,j,k,2) ););
                        });
                    }
                } else {
                    if (m_advect_momentum)
                    {
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            AMREX_D_TERM(vel(i,j,k,0) = rho_old(i,j,k) * vel_o(i,j,k,0) + l_dt * (
                                                        l_half*(  dvdt_o(i,j,k,0)+dvdt(i,j,k,0))
                                                      + rho_nph(i,j,k) * vel_f(i,j,k,0) );,
                                         vel(i,j,k,1) = rho_old(i,j,k) * vel_o(i,j,k,1) + l_dt * (
                                                        l_half*(  dvdt_o(i,j,k,1)+dvdt(i,j,k,1))
                                                      + rho_nph(i,j,k) * vel_f(i,j,k,1) );,
                                         vel(i,j,k,2) = rho_old(i,j,k) * vel_o(i,j,k,2) + l_dt * (
                                                        l_half*(  dvdt_o(i,j,k,2)+dvdt(i,j,k,2))
                                                      + rho_nph(i,j,k) * vel_f(i,j,k,2) ););
                            AMREX_D_TERM(vel(i,j,k,0) /= rho_new(i,j,k);,
                                         vel(i,j,k,1) /= rho_new(i,j,k);,
                                         vel(i,j,k,2) /= rho_new(i,j,k););
                        });
                    } else {
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            AMREX_D_TERM(vel(i,j,k,0) = vel_o(i,j,k,0) + l_dt * (
                                                        l_half*(  dvdt_o(i,j,k,0)+dvdt(i,j,k,0)) + vel_f(i,j,k,0) );,
                                         vel(i,j,k,1) = vel_o(i,j,k,1) + l_dt * (
                                                        l_half*(  dvdt_o(i,j,k,1)+dvdt(i,j,k,1)) + vel_f(i,j,k,1) );,
                                         vel(i,j,k,2) = vel_o(i,j,k,2) + l_dt * (
                                                        l_half*(  dvdt_o(i,j,k,2)+dvdt(i,j,k,2)) + vel_f(i,j,k,2) ););
                        });
                    }
                }
            }
        }
        } // lev
    } // Corrector

    // *************************************************************************************
    // Solve diffusion equation for u* but using eta_old at old time
    // *************************************************************************************
    if ((m_diff_type == DiffusionType::Crank_Nicolson || m_diff_type == DiffusionType::Implicit)
        &&
        (!m_gran_rheo_modified_time_stepping))
    {
        const int ng_diffusion = 1;
        for (int lev = 0; lev <= finest_level; ++lev) {
            fillphysbc_velocity(lev, new_time, m_leveldata[lev]->velocity, ng_diffusion);
            fillphysbc_density (lev, new_time, m_leveldata[lev]->density , ng_diffusion);
        }

        Real dt_diff = (m_diff_type == DiffusionType::Implicit) ? m_dt : l_half*m_dt;
        diffuse_velocity(get_velocity_new(), get_density_new(), GetVecOfConstPtrs(vel_eta), dt_diff);

#ifdef AMREX_USE_EB
        if (m_probtype ==  537) {
            auto const& bc_vel = get_velocity_bcrec_device_ptr();
            for (int lev = 0; lev <= finest_level; lev++)
            {
                auto& ld = *m_leveldata[lev];
                const auto& fact = EBFactory(lev);
                // Intentionally setting u component negative values to small positive values
                auto const& flags = fact.getMultiEBCellFlagFab();
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(ld.velocity,TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    Box const& bx = mfi.tilebox();
                    Array4<Real> const& vel = ld.velocity.array(mfi);
                    auto const& flag_fab = flags[mfi];
                    auto typ = flag_fab.getType(bx);
                    if (typ != FabType::singlevalued) {
                        continue;
                    }
                    auto const& flag_arr = flag_fab.const_array();
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        if (flag_arr(i,j,k).isSingleValued()) {
                            if (vel(i,j,k,0) < Real(0.)) {
                                vel(i,j,k,0) = Real(1.0e-18);
                            }
                        }
                    });
                }
                // Applying StateRedistribution
                auto& vel_main = ld.velocity;
                MultiFab vel_tmp;
                vel_tmp.define(vel_main.boxArray(), vel_main.DistributionMap(),
                        AMREX_SPACEDIM, vel_main.nGrow(), MFInfo(),
                        vel_main.Factory());
                MultiFab::Copy(vel_tmp, vel_main, 0, 0, AMREX_SPACEDIM,
                        vel_main.nGrow());
                MultiFab vel_state;
                vel_state.define(vel_main.boxArray(), vel_main.DistributionMap(),
                        AMREX_SPACEDIM, vel_main.nGrow(), MFInfo(),
                        vel_main.Factory());
                vel_state.setVal(Real(0.), vel_main.nGrow());
                redistribute_term(vel_main, vel_tmp, vel_state, bc_vel, lev);
            }
        }
#endif
    }

    // *********************************************************************************************
    // Modified Time Stepping needs to solve diffusion equation for u* but using timestepping_alpha
    // *********************************************************************************************
    if (m_gran_rheo_modified_time_stepping) {
        const int ng_diffusion = 1;
        for (int lev = 0; lev <= finest_level; ++lev) {
            fillphysbc_velocity(lev, new_time, m_leveldata[lev]->velocity, ng_diffusion);
            fillphysbc_density (lev, new_time, m_leveldata[lev]->density , ng_diffusion);
        }

        Real dt_diff = m_dt;
        diffuse_velocity(get_velocity_new(), get_density_new(),
                         GetVecOfConstPtrs(timestepping_alpha), dt_diff);
    }
}
