#include <incflo.H>

using namespace amrex;
//
// Apply predictor:
//
//  1. Use u = vel_old to compute
//
//      if (!advect_momentum) then
//          conv_u  = - u grad u
//      else
//          conv_u  = - del dot (rho u u)
//      conv_r  = - div( u rho  )
//      if (m_iconserv_tracer) then
//          conv_t  = - div( u trac )
//      else
//          conv_t  = - u dot grad trac
//      eta_old     = visosity at m_cur_time
//      if (m_diff_type == DiffusionType::Explicit)
//         divtau _old = div( eta ( (grad u) + (grad u)^T ) ) / rho^n
//         rhs = u + dt * ( conv + divtau_old )
//      else
//         divtau_old  = 0.0
//         rhs = u + dt * conv
//
//      eta     = eta at new_time
//
//  2. Add explicit forcing term i.e. gravity + lagged pressure gradient
//
//      rhs += dt * ( g - grad(p + p0) / rho^nph )
//
//  3. A. If (m_diff_type == DiffusionType::Implicit)
//        solve implicit diffusion equation for u*
//
//     ( 1 - dt / rho^nph * div ( eta grad ) ) u* = u^n + dt * conv_u
//                                                  + dt * ( g - grad(p + p0) / rho^nph )
//
//     B. If (m_diff_type == DiffusionType::Crank-Nicolson)
//        solve semi-implicit diffusion equation for u*
//
//     ( 1 - (dt/2) / rho^nph * div ( eta_old grad ) ) u* = u^n +
//            dt * conv_u + (dt/2) / rho * div (eta_old grad) u^n
//          + dt * ( g - grad(p + p0) / rho^nph )
//
//  4. Apply projection
//
//     Add pressure gradient term back to u*:
//
//      if (advect_momentum) then
//          (rho^(n+1) u**) = (rho^(n+1) u*) + dt * grad p
//      else
//          u** = u* + dt * grad p / rho^nph
//
//     Solve Poisson equation for phi:
//
//     div( grad(phi) / rho^nph ) = div( u** )
//
//     Update pressure:
//
//     p = phi / dt
//
//     Update velocity, now divergence free
//
//     vel = u** - dt * grad p / rho^nph
//
// It is assumed that the ghost cells of the old data have been filled and
// the old and new data are the same in valid region.
//
void incflo::ApplyPredictor (bool incremental_projection)
{
    BL_PROFILE("incflo::ApplyPredictor");

    // We use the new time value for things computed on the "*" state
    Real new_time = m_cur_time + m_dt;

    // *************************************************************************************
    // Allocate space for the MAC velocities
    // *************************************************************************************
    Vector<MultiFab> u_mac(finest_level+1), v_mac(finest_level+1), w_mac(finest_level+1);
    int ngmac = nghost_mac();

    for (int lev = 0; lev <= finest_level; ++lev) {
        AMREX_D_TERM(u_mac[lev].define(amrex::convert(grids[lev],IntVect::TheDimensionVector(0)), dmap[lev],
                          1, ngmac, MFInfo(), Factory(lev));,
                     v_mac[lev].define(amrex::convert(grids[lev],IntVect::TheDimensionVector(1)), dmap[lev],
                          1, ngmac, MFInfo(), Factory(lev));,
                     w_mac[lev].define(amrex::convert(grids[lev],IntVect::TheDimensionVector(2)), dmap[lev],
                          1, ngmac, MFInfo(), Factory(lev)););
        // do we still want to do this now that we always call a FillPatch (and all ghost cells get filled)?
        if (ngmac > 0) {
            AMREX_D_TERM(u_mac[lev].setBndry(0.0);,
                         v_mac[lev].setBndry(0.0);,
                         w_mac[lev].setBndry(0.0););
        }
    }

    // *************************************************************************************
    // Allocate space for half-time density
    // *************************************************************************************
    // Forcing terms
    Vector<MultiFab> vel_forces, tra_forces;

    Vector<MultiFab> vel_eta, tra_eta;

    // *************************************************************************************
    // Allocate space for the forcing terms
    // *************************************************************************************
    for (int lev = 0; lev <= finest_level; ++lev) {
        vel_forces.emplace_back(grids[lev], dmap[lev], AMREX_SPACEDIM, nghost_force(),
                                MFInfo(), Factory(lev));

        if (m_advect_tracer) {
            tra_forces.emplace_back(grids[lev], dmap[lev], m_ntrac, nghost_force(),
                                    MFInfo(), Factory(lev));
        }
        if (m_nodal_vel_eta) {
            vel_eta.emplace_back(amrex::convert(grids[lev],
                    IndexType::TheNodeType().ixType()),
                    dmap[lev], 1, 0, MFInfo(), Factory(lev));
        } else {
            vel_eta.emplace_back(grids[lev], dmap[lev], 1, 1, MFInfo(), Factory(lev));
        }
        if (m_advect_tracer) {
            tra_eta.emplace_back(grids[lev], dmap[lev], m_ntrac, 1, MFInfo(), Factory(lev));
        }
    }

    // *************************************************************************************
    // Compute viscosity / diffusive coefficients
    // *************************************************************************************
    compute_viscosity(GetVecOfPtrs(vel_eta),
                      get_density_old(), get_velocity_old(),
                      m_cur_time, 1);

    // *************************************************************************************
    // Compute explicit viscous term
    // Note that for !advect_momentum, this actually computes divtau / rho
    // *************************************************************************************
    if (need_divtau() || use_tensor_correction )
    {
        compute_divtau(get_divtau_old(),get_velocity_old_const(),
                       get_density_old_const(),GetVecOfConstPtrs(vel_eta));
    }

    // *************************************************************************************
    // Compute explicit diffusive term -- note this is used inside compute_convective_term
    // *************************************************************************************
    if (m_advect_tracer)
    {
        compute_tracer_diff_coeff(GetVecOfPtrs(tra_eta),1);
        if (need_divtau()) {
            compute_laps(get_laps_old(), get_tracer_old_const(), GetVecOfConstPtrs(tra_eta));
        }
    }

    // **********************************************************************************************
    // Compute the forcing terms
    // *************************************************************************************
    bool include_pressure_gradient = !(m_use_mac_phi_in_godunov);
    compute_vel_forces(GetVecOfPtrs(vel_forces), get_velocity_old_const(),
                       get_density_old_const(), get_tracer_old_const(), get_tracer_old_const(),
                       include_pressure_gradient);

    // **********************************************************************************************
    // Compute the MAC-projected velocities at all levels
    // *************************************************************************************
    compute_MAC_projected_velocities(get_velocity_old_const(), get_density_old_const(),
                                     AMREX_D_DECL(GetVecOfPtrs(u_mac), GetVecOfPtrs(v_mac),
                                     GetVecOfPtrs(w_mac)), GetVecOfPtrs(vel_forces), m_cur_time);

    // *************************************************************************************
    // if (advection_type == "Godunov")
    //      Compute the explicit advective terms R_u^(n+1/2), R_s^(n+1/2) and R_t^(n+1/2)
    // if (advection_type == "MOL"                )
    //      Compute the explicit advective terms R_u^n      , R_s^n       and R_t^n
    // Note that if advection_type != "MOL" then we call compute_tra_forces inside this routine
    // *************************************************************************************
    compute_convective_term(get_conv_velocity_old(), get_conv_density_old(), get_conv_tracer_old(),
                            get_velocity_old_const(), get_density_old_const(), get_tracer_old_const(),
                            AMREX_D_DECL(GetVecOfPtrs(u_mac), GetVecOfPtrs(v_mac),
                            GetVecOfPtrs(w_mac)),
                            GetVecOfPtrs(vel_forces), GetVecOfPtrs(tra_forces),
                            m_cur_time);

    // *************************************************************************************
    // Update density
    // *************************************************************************************
    if (!m_two_fluid) update_density(StepType::Predictor);

    // **********************************************************************************************
    // Update tracer
    // **********************************************************************************************
    update_tracer(StepType::Predictor, tra_eta, tra_forces);

    // **********************************************************************************************
    // Update velocity
    // **********************************************************************************************
    update_velocity(StepType::Predictor, vel_eta, vel_forces);

    // **********************************************************************************************
    // Project velocity field, update pressure
    // **********************************************************************************************
    ApplyProjection(get_density_nph_const(),new_time,m_dt,incremental_projection);

#ifdef INCFLO_USE_PARTICLES
    // **************************************************************************************
    // Update the particle positions
    // **************************************************************************************
    if (m_advection_type != "MOL") {
        evolveTracerParticles(AMREX_D_DECL(GetVecOfConstPtrs(u_mac), GetVecOfConstPtrs(v_mac),
                                           GetVecOfConstPtrs(w_mac)));
    }
#endif

#ifdef AMREX_USE_EB
    // **********************************************************************************************
    // Over-write velocity in cells with vfrac < 1e-4
    // **********************************************************************************************
    if (m_advection_type == "MOL")
        incflo_correct_small_cells(get_velocity_new(),
                                   AMREX_D_DECL(GetVecOfConstPtrs(u_mac), GetVecOfConstPtrs(v_mac),
                                   GetVecOfConstPtrs(w_mac)));
#endif


    // Apply re-initialization of Level Set Method
    /*
    if (m_two_fluid and m_advection_type != "MOL") {
        // Copy from new to old tracer
        copy_from_new_to_old_tracer();
        int ng = nghost_state();
        for (int lev = 0; lev <= finest_level; ++lev) {
            fillpatch_tracer(lev, m_t_old[lev], m_leveldata[lev]->tracer_o, ng);
            m_leveldata[lev]->tracer_o.FillBoundary(geom[lev].periodicity());
        }
        Real lvl_set_d = m_level_set_d;
        for (int pseudo_t=0; pseudo_t < 1; ++pseudo_t) {
        const Vector<MultiFab const*> trac_old = get_tracer_old_const();

        // Evaluate the new tra_eta at each level
        for (int lev = 0; lev <= finest_level; ++lev) {
            const Real dx_lev = geom[lev].CellSize(0);
            const Real inv_dx = Real(1.0)/dx_lev;
            const Real dy_lev = geom[lev].CellSize(1);
            const Real inv_dy = Real(1.0)/dy_lev;
            Real lvl_set_eps = amrex::max(dx_lev,dy_lev);
#if (AMREX_SPACEDIM == 3)
            const Real dz_lev = geom[lev].CellSize(2);
            const Real inv_dz = Real(1.0)/dz_lev;
            lvl_set_eps = amrex::max(lvl_set_eps,dz_lev);
#endif
            lvl_set_eps = std::pow(lvl_set_eps,Real(1.0)-lvl_set_d)*Real(0.5);
            // Set to lvl_set_eps initially
            tra_eta[lev].setVal(lvl_set_eps,1);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(tra_eta[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box const& bx = mfi.tilebox();
                Array4<Real      > const& eta_tracer = tra_eta[lev].array(mfi);
                Array4<Real const> const& old_tracer = trac_old[lev]->const_array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real grad_x = Real(0.5)*inv_dx*(old_tracer(i+1,j  ,k  )-old_tracer(i-1,j  ,k  ));
                    Real grad_y = Real(0.5)*inv_dy*(old_tracer(i  ,j+1,k  )-old_tracer(i  ,j-1,k  ));
#if (AMREX_SPACEDIM == 3)
                    Real grad_z = Real(0.5)*inv_dz*(old_tracer(i  ,j  ,k+1)-old_tracer(i  ,j  ,k-1));
#endif
                    Real grad_mag = AMREX_D_TERM(grad_x*grad_x,+grad_y*grad_y,+grad_z*grad_z);
                    grad_mag = std::sqrt(grad_mag) + Real(1.0e-18);
                    eta_tracer(i,j,k) -= (old_tracer(i,j,k)*(Real(1.0)-old_tracer(i,j,k))/grad_mag);
                });
            }

            tra_eta[lev].FillBoundary(geom[lev].periodicity());
        }
        // Compute div (tra_eta grad) phi (Here phi is level set value)
        compute_laps(get_laps_old(), get_tracer_old_const(), GetVecOfConstPtrs(tra_eta));
        // Use laps_o to update tracer values
        // Single Time step based on the lev=0 is used
        Real lvl_set_dt = geom[0].CellSize(0);
        lvl_set_dt = amrex::max(lvl_set_dt,geom[0].CellSize(1));
#if (AMREX_SPACEDIM == 3)
        lvl_set_dt = amrex::max(lvl_set_dt,geom[0].CellSize(2));
#endif
        lvl_set_dt = Real(0.5)*std::pow(lvl_set_dt,Real(1.0)+lvl_set_d);
        for (int lev = 0; lev <= finest_level; lev++)
        {
            auto& ld = *m_leveldata[lev];
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(ld.tracer,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box const& bx = mfi.tilebox();
                Array4<Real const> const& tra_o   = ld.tracer_o.const_array(mfi);
                Array4<Real const> const& laps_o = ld.laps_o.const_array(mfi);
                Array4<Real> const& tra           = ld.tracer.array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    tra(i,j,k) = tra_o(i,j,k)+lvl_set_dt*laps_o(i,j,k);
                });
            }

            ld.tracer.FillBoundary(geom[lev].periodicity());
        }
        // Copy from new to old
        copy_from_new_to_old_tracer();
        for (int lev = 0; lev <= finest_level; ++lev) {
            fillpatch_tracer(lev, m_t_old[lev], m_leveldata[lev]->tracer_o, ng);
            m_leveldata[lev]->tracer_o.FillBoundary(geom[lev].periodicity());
        }
        } // pseudo-t
        // Initialize tra_eta to its original value
        compute_tracer_diff_coeff(GetVecOfPtrs(tra_eta),1);

        // Update density again at the end
        ng = 1;
        for (int lev = 0; lev <= finest_level; lev++)
        {
            auto& ld = *m_leveldata[lev];
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(ld.velocity,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box const& bx = mfi.tilebox();
                Array4<Real> const& rho_new  = ld.density.array(mfi);
                Array4<Real> const& tracer   = ld.tracer.array(mfi);
                Real rho_1 = m_ro_0;
                Real rho_2 = m_ro_0_second;
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // Clipping for tracer
                    if (tracer(i,j,k) < Real(0.0)) {
                        tracer(i,j,k) = Real(0.0);
                    }
                    if (tracer(i,j,k) > Real(1.0)) {
                        tracer(i,j,k) = Real(1.0);
                    }
                    rho_new(i,j,k) = rho_1 + (rho_2-rho_1)*tracer(i,j,k,0);
                });

            } // mfi
        } // lev

        // Average down solution
        for (int lev = finest_level-1; lev >= 0; --lev) {
#ifdef AMREX_USE_EB
            amrex::EB_average_down(m_leveldata[lev+1]->density, m_leveldata[lev]->density,
                                   0, 1, refRatio(lev));
#else
            amrex::average_down(m_leveldata[lev+1]->density, m_leveldata[lev]->density,
                                0, 1, refRatio(lev));
#endif
        }

        for (int lev = 0; lev <= finest_level; lev++)
        {
            auto& ld = *m_leveldata[lev];

            // Fill ghost cells of new-time density if needed (we assume ghost cells of old density are already filled)
            if (ng > 0) {
                fillpatch_density(lev, m_t_new[lev], ld.density, ng);
            }

            // Define half-time density after the average down
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(ld.velocity,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box const& bx = mfi.tilebox();
                Array4<Real      > const& rho_nph  = ld.density_nph.array(mfi);
                Array4<Real const> const& tracer   = ld.tracer.const_array(mfi);
                Array4<Real const> const& tracer_o   = ld.tracer_o.const_array(mfi);
                Real rho_1 = m_ro_0;
                Real rho_2 = m_ro_0_second;
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real tracer_nph = Real(0.5)*(tracer(i,j,k,0)+tracer_o(i,j,k,0));
                    rho_nph(i,j,k) = rho_1 + (rho_2-rho_1)*tracer_nph;
                });
            } // mfi
            if (ng > 0) {
                fillpatch_density(lev, m_t_new[lev], ld.density_nph, ng);
                ld.density_nph.FillBoundary(IntVectND<AMREX_SPACEDIM>::TheUnitVector(),
                                            geom[lev].periodicity());
            }
        }

    }
    */
}
