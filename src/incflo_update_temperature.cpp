#include <incflo.H>

using namespace amrex;

void incflo::update_temperature (StepType step_type, Vector<MultiFab>& tem_eta, Vector<MultiFab>& scratch)
{
    BL_PROFILE("incflo::update_temperature");

    if (!m_use_temperature) { return; }

    Real const  new_time = m_cur_time + m_dt;
    Real const half_time = m_cur_time + m_dt * Real(0.5);

    const bool gran_temp = m_use_granular_temperature;
    Vector<iMultiFab> overset_mask;
    if (gran_temp) {
        for (int lev = 0; lev <= finest_level; lev++) {
            overset_mask.emplace_back(grids[lev], dmap[lev], 1, nghost_state(),
                                      MFInfo(), DefaultFabFactory<IArrayBox>());
        }
    }

    // *************************************************************************************
    // Compute the temperature forcing terms
    // *************************************************************************************
    compute_tem_forces(half_time, GetVecOfPtrs(scratch));

    // *************************************************************************************
    // Compute explicit diffusive term (if corrector)
    // *************************************************************************************
    if (step_type == StepType::Corrector)
    {
        compute_temperature_diff_coeff(new_time, GetVecOfPtrs(tem_eta));
        if (m_diff_type == DiffusionType::Explicit) {
            compute_laps_T(get_laps_new(), get_temperature_new_const(), GetVecOfConstPtrs(tem_eta));
        }
    }

    // *************************************************************************************
    // Update the temperature with time-explicit terms
    // *************************************************************************************
    if (step_type == StepType::Predictor) {
        constexpr Real m_half = Real(0.5);
        Real l_dt = m_dt;

        for (int lev = 0; lev <= finest_level; lev++)
        {
            auto& ld = *m_leveldata[lev];

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(ld.tracer,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box const& bx = mfi.tilebox();
                Array4<Real const> const& tem_o   = ld.temperature_o.const_array(mfi);
                Array4<Real      > const& tem     = ld.temperature.array(mfi);
                Array4<Real const> const& rho_h   = ld.density_nph.const_array(mfi);
                Array4<Real const> const& dtdt_o  = ld.conv_temperature_o.const_array(mfi);
                // temperature forcing term (Q) is in scratch
                Array4<Real      > const& tem_f   = scratch[lev].array(mfi);
                // First tracer used when granular temperature is true
                Array4<Real const> const& tra_n   = ld.tracer.const_array(mfi);
                const Real min_conc_scnd = m_min_conc_second;
                const Real gt_coll_dissp = m_gran_temp_collisional_dissipation;

                FArrayBox cp_fab(bx, 1, The_Async_Arena());
                compute_cp(lev, mfi, cp_fab);
                Array4<Real      > const& cp      = cp_fab.array();

                if (m_diff_type == DiffusionType::Explicit)
                {
                    Array4<Real const> const& laps_o = ld.laps_tem_o.const_array(mfi);
                    if (!gran_temp)
                    {
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            tem(i,j,k) = tem_o(i,j,k) + l_dt *
                                ( dtdt_o(i,j,k) + (tem_f(i,j,k) + laps_o(i,j,k))/(rho_h(i,j,k) * cp(i,j,k)) );
                        });
                    }
                    else
                    {
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            if (tra_n(i,j,k,0) > min_conc_scnd) {
                                tem(i,j,k) = tem_o(i,j,k) + l_dt *
                                    ( dtdt_o(i,j,k) + (tem_f(i,j,k) + laps_o(i,j,k))/cp(i,j,k) );
                                // Adding the collisional dissipation term
                                tem(i,j,k) += l_dt *(-gt_coll_dissp)*tem_o(i,j,k)/cp(i,j,k);
                            }
                        });
                    }
                }
                else if (m_diff_type == DiffusionType::Crank_Nicolson)
                {
                    Array4<Real const> const& laps_o = ld.laps_tem_o.const_array(mfi);
                    if (!gran_temp) {
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            tem(i,j,k) = tem_o(i,j,k) + l_dt *
                                ( dtdt_o(i,j,k) + (tem_f(i,j,k) + m_half*laps_o(i,j,k))/(rho_h(i,j,k) * cp(i,j,k)) );
                            // Save rhoCp for use in implicit solve.
                            // Reuse scratch space since we are done with forcing now.
                            tem_f(i,j,k) = rho_h(i,j,k) * cp(i,j,k);
                        });
                    }
                    else
                    {
                        auto const& osm = overset_mask[lev].array(mfi);
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            tem(i,j,k) = tem_o(i,j,k) + l_dt *
                                ( dtdt_o(i,j,k) + (tem_f(i,j,k) + m_half*laps_o(i,j,k))/cp(i,j,k) );
                            // Reuse scratch space since we are done with forcing now.
                            // This will be the chi used in diffusion solve
                            // Collisional dissipation is considered implicitly
                            tem_f(i,j,k) = cp(i,j,k) + gt_coll_dissp*l_dt;
                            // The below modification is due to the structure of
                            // incflo's temperature diffusion solve
                            tem(i,j,k) *= (cp(i,j,k)/(cp(i,j,k) + gt_coll_dissp*l_dt));
                            // Using overset_mask to only solve for granular region
                            if (tra_n(i,j,k,0) >= min_conc_scnd ) {
                                osm(i,j,k) = 1;
                            }
                            else {
                                osm(i,j,k) = 0;
                                // The solution should not change in the masked region
                                tem(i,j,k) = tem_o(i,j,k);
                            }
                        });
                    }
                }
                else if (m_diff_type == DiffusionType::Implicit)
                {
                    if (!gran_temp)
                    {
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            tem(i,j,k) = tem_o(i,j,k) + l_dt *
                                (dtdt_o(i,j,k) + tem_f(i,j,k) / (rho_h(i,j,k) * cp(i,j,k)));
                            // Save rhoCp for use in implicit solve.
                            // Reuse scratch space since we are done with forcing now.
                            tem_f(i,j,k) = rho_h(i,j,k) * cp(i,j,k);
                        });
                    }
                    else
                    {
                        auto const& osm = overset_mask[lev].array(mfi);
                        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            tem(i,j,k) = tem_o(i,j,k) + l_dt *
                                (dtdt_o(i,j,k) + tem_f(i,j,k) / cp(i,j,k));
                            // Reuse scratch space since we are done with forcing now.
                            // This will be the chi used in diffusion solve
                            // Collisional dissipation is considered implicitly
                            tem_f(i,j,k) = cp(i,j,k) + gt_coll_dissp*l_dt;
                            // The below modification is due to the structure of
                            // incflo's temperature diffusion solve
                            tem(i,j,k) *= (cp(i,j,k)/(cp(i,j,k) + gt_coll_dissp*l_dt));
                            // Using overset_mask to only solve for granular region
                            if (tra_n(i,j,k,0) > min_conc_scnd ) {
                                osm(i,j,k) = 1;
                            }
                            else {
                                osm(i,j,k) = 0;
                                // The solution should not change in the masked region
                                tem(i,j,k) = tem_o(i,j,k);
                            }
                        });
                    }
                }
            } // mfi
            if (gran_temp) {
                overset_mask[lev].FillBoundary(geom[lev].periodicity());
            }
        } // lev

    } else if (step_type == StepType::Corrector) {
        Abort("incflo::update_temperature does not yet work with the corrector");
    }

    // *************************************************************************************
    // Solve implicit diffusion equation for temperature
    // *************************************************************************************
    if (m_diff_type == DiffusionType::Crank_Nicolson || m_diff_type == DiffusionType::Implicit)
    {
        const int ng_diffusion = 1;
        for (int lev = 0; lev <= finest_level; ++lev) {
            fillphysbc_temperature(lev, new_time, m_leveldata[lev]->temperature, ng_diffusion);
        }
        Real dt_diff = (m_diff_type == DiffusionType::Implicit) ? m_dt : Real(0.5)*m_dt;
        auto overset_mask_ptrs = gran_temp ? GetVecOfConstPtrs(overset_mask)
                                           : Vector<iMultiFab const*>{};
        // scratch holds rhoCp if it is NOT Granular Temperature
        diffuse_temperature(get_temperature_new(), GetVecOfPtrs(scratch), GetVecOfConstPtrs(tem_eta),
                            dt_diff, gran_temp ? &overset_mask_ptrs : nullptr);
    }
    else
    {
        // Need to average down temperature since the diffusion solver didn't do it for us.
        for (int lev = finest_level-1; lev >= 0; --lev) {
#ifdef AMREX_USE_EB
            amrex::EB_average_down(m_leveldata[lev+1]->temperature, m_leveldata[lev]->temperature,
                                   0, 1, refRatio(lev));
#else
            amrex::average_down(m_leveldata[lev+1]->temperature, m_leveldata[lev]->temperature,
                                0, 1, refRatio(lev));
#endif
        }
    }
}
