#include <incflo.H>
#include <NonlinearDiffusionTensorOp.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>

#ifdef AMREX_USE_EB
#include <AMReX_EB_Redistribution.H>
#include <memory>
#endif

using namespace amrex;

NonlinearDiffusionTensorOp::NonlinearDiffusionTensorOp (incflo* a_incflo)
    : m_incflo(a_incflo)
{
    readParameters();
    int finest_level = m_incflo->finestLevel();

    // GMRES
    m_gmres = std::make_unique<GM>();
    m_gmres->define(*this);
    m_gmres->setMaxIters(m_gmres_max_iter);
    m_gmres->setVerbose(m_gmres_verbose);

    if (m_incflo->m_nodal_vel_eta) {m_nghost_eta = 0;}

    // The below code is related to linear part of divtau
    LPInfo info_solve;
    info_solve.setMaxCoarseningLevel(m_mg_max_coarsening_level);
    LPInfo info_apply;
    info_apply.setMaxCoarseningLevel(0);
#ifdef AMREX_USE_EB
    if (!m_incflo->EBFactory(0).isAllRegular())
    {
        Vector<EBFArrayBoxFactory const*> ebfact;
        for (int lev = 0; lev <= finest_level; ++lev) {
            ebfact.push_back(&(m_incflo->EBFactory(lev)));
        }

        if (m_incflo->useTensorSolve())
        {
            m_eb_solve_op = std::make_unique<MLEBTensorOp>(m_incflo->Geom(0,finest_level),
                                                 m_incflo->boxArray(0,finest_level),
                                                 m_incflo->DistributionMap(0,finest_level),
                                                 info_solve, ebfact);
            m_eb_solve_op->setMaxOrder(m_mg_maxorder);
            m_eb_solve_op->setDomainBC(m_incflo->get_diffuse_tensor_bc(Orientation::low),
                                       m_incflo->get_diffuse_tensor_bc(Orientation::high));
        }

        if (m_incflo->need_divtau() || m_incflo->useTensorCorrection())
        {
            m_eb_apply_op = std::make_unique<MLEBTensorOp>(m_incflo->Geom(0,finest_level),
                                                 m_incflo->boxArray(0,finest_level),
                                                 m_incflo->DistributionMap(0,finest_level),
                                                 info_apply, ebfact);
            m_eb_apply_op->setMaxOrder(m_mg_maxorder);
            m_eb_apply_op->setDomainBC(m_incflo->get_diffuse_tensor_bc(Orientation::low),
                                       m_incflo->get_diffuse_tensor_bc(Orientation::high));
        }
    }
    else
#endif
    {
        if (m_incflo->useTensorSolve())
        {
            m_reg_solve_op = std::make_unique<MLTensorOp>(m_incflo->Geom(0,finest_level),
                                                m_incflo->boxArray(0,finest_level),
                                                m_incflo->DistributionMap(0,finest_level),
                                                info_solve);
            m_reg_solve_op->setGaussSeidel(m_mg_use_gauss_seidel);
            m_reg_solve_op->setMaxOrder(m_mg_maxorder);
            m_reg_solve_op->setDomainBC(m_incflo->get_diffuse_tensor_bc(Orientation::low),
                                        m_incflo->get_diffuse_tensor_bc(Orientation::high));
        }

        if (m_incflo->need_divtau() || m_incflo->useTensorCorrection())
        {
            m_reg_apply_op = std::make_unique<MLTensorOp>(m_incflo->Geom(0,finest_level),
                                                m_incflo->boxArray(0,finest_level),
                                                m_incflo->DistributionMap(0,finest_level),
                                                info_apply);
            m_reg_apply_op->setMaxOrder(m_mg_maxorder);
            m_reg_apply_op->setDomainBC(m_incflo->get_diffuse_tensor_bc(Orientation::low),
                                        m_incflo->get_diffuse_tensor_bc(Orientation::high));
        }
    }
#ifdef AMREX_USE_EB
    if ((!m_incflo->EBFactory(0).isAllRegular()) &&
        (m_incflo->m_nodal_vel_eta != 0)) {
        amrex::Abort(
          "Nodal viscosity with Embedded Boundaries does NOT exist.\n");
    }
    const bool has_granular_powerlaw_ho =
        (m_incflo->m_fluid_model_second == incflo::FluidModel::GranularPowerlaw
         && m_incflo->m_mu_powerlaw.size() > 1)
        || (m_incflo->m_fluid_model_second
                == incflo::FluidModel::GranularPowerlawTemperature
            && m_incflo->m_mu_powerlaw_temperature.size() > 1);
    if ((m_incflo->m_nodal_vel_eta != 0) &&
        has_granular_powerlaw_ho) {
        amrex::Abort(
          "High-order effects require Cell-centered Viscosity.\n");
    }
    AMREX_ALWAYS_ASSERT(!m_incflo->hasEBFlow());
#endif
    // Number of high-order coefficients
    if (m_incflo->m_mu_powerlaw_temperature.size() > 0) {
        m_ncomp_ho = std::max(m_ncomp_ho,
                (int)(m_incflo->m_mu_powerlaw_temperature.size()-1));
    }
}

void NonlinearDiffusionTensorOp::readParameters ()
{
    ParmParse pp("nonlinear_tensor_diffusion");

    pp.query("verbose", m_verbose);
    pp.query("newton_max_iter", m_newton_max_iter);
    pp.query("newton_rtol", m_newton_rtol);
    pp.query("newton_atol", m_newton_atol);
    pp.query("newton_update_alpha", m_newton_update_alpha);
    pp.query("newton_update_max_iter", m_newton_update_max_iter);
    pp.query("num_time_substeps", m_num_time_substeps);
    pp.query("adaptive_time_substeps", m_adaptive_time_substeps);
    pp.query("max_time_substeps", m_max_time_substeps);
    pp.query("time_substep_newton_iter_high",
             m_time_substep_newton_iter_high);
    pp.query("time_substep_newton_iter_low",
             m_time_substep_newton_iter_low);
    pp.query("use_eta_from_prev_time", m_use_eta_from_prev_time);
    pp.query("use_ho_coeff_from_prev_time", m_use_ho_coeff_from_prev_time);
    pp.query("use_ho_eta_precond", m_use_ho_eta_precond);
    pp.query("newton_epsilon", m_newton_epsilon);
    if (m_num_time_substeps < 1) {
        amrex::Abort("nonlinear_tensor_diffusion.num_time_substeps must be >= 1");
    }
    if (m_max_time_substeps < m_num_time_substeps) {
        m_max_time_substeps = m_num_time_substeps;
    }
    if (m_time_substep_newton_iter_high < 0) {
        m_time_substep_newton_iter_high =
            std::max(1, (8*m_newton_max_iter)/10);
    }
    if (m_time_substep_newton_iter_low < 0) {
        m_time_substep_newton_iter_low =
            std::max(1, m_newton_max_iter/4);
    }
    m_next_time_substeps = m_num_time_substeps;
    // Get the alpha_factor_list from input file
    if (pp.queryarr("alpha_factor_list",m_alpha_factor_list)) {
        m_alpha_factor_list.clear();
    }
    pp.queryarr("alpha_factor_list",m_alpha_factor_list);
    // Last value NEEDS to be UNITY
    if (m_alpha_factor_list.back() != Real(1.0)) {
        m_alpha_factor_list.emplace_back(Real(1.0));
    }
    pp.query("nonunity_alpha_tol_scale",m_nonunity_alpha_tol_scale);

    pp.query("gmres_verbose", m_gmres_verbose);
    pp.query("gmres_max_iter", m_gmres_max_iter);
    pp.query("gmres_rtol", m_gmres_rtol);
    pp.query("gmres_atol", m_gmres_atol);
    pp.query("gmres_use_precond", m_gmres_use_precond);
    // This is for linear part of divtau
    // MLMG-related
    pp.query("mg_verbose", m_mg_verbose);
    pp.query("mg_bottom_verbose", m_mg_bottom_verbose);
    pp.query("mg_max_iter", m_mg_max_iter);
    pp.query("mg_bottom_maxiter", m_mg_bottom_maxiter);
    pp.query("mg_max_fmg_iter", m_mg_max_fmg_iter);
    pp.query("mg_max_coarsening_level", m_mg_max_coarsening_level);
    pp.query("mg_maxorder", m_mg_maxorder);
    pp.query("mg_rtol", m_mg_rtol);
    pp.query("mg_atol", m_mg_atol);
    pp.query("bottom_solver", m_bottom_solver);
    pp.query("num_pre_smooth", m_num_pre_smooth);
    pp.query("num_post_smooth", m_num_post_smooth);
    pp.query("use_gauss_seidel", m_mg_use_gauss_seidel);
}


void NonlinearDiffusionTensorOp::diffuse_velocity (
                       Vector<MultiFab*> const& velocity,
                       Vector<MultiFab*> const& density,
                       Vector<MultiFab const*> const& eta,
                       Real dt)
{
    const int nlevels = velocity.size();
    Vector<MultiFab> velocity_save(nlevels);
    for (int ilev=0; ilev < nlevels; ++ilev) {
        velocity_save[ilev].define(velocity[ilev]->boxArray(),
                                   velocity[ilev]->DistributionMap(),
                                   AMREX_SPACEDIM, velocity[ilev]->nGrow(),
                                   MFInfo(), velocity[ilev]->Factory());
        MultiFab::Copy(velocity_save[ilev], *velocity[ilev],
                       0, 0, AMREX_SPACEDIM, velocity[ilev]->nGrow());
    }

    int nsub = m_adaptive_time_substeps
        ? std::min(m_next_time_substeps, m_max_time_substeps)
        : m_num_time_substeps;
    nsub = std::max(1, nsub);
    SolveStats accepted_stats;
    bool retried = false;

    for (;;)
    {
        SolveStats attempt_stats;
        attempt_stats.converged = true;
        const Real dt_sub = dt / Real(nsub);

        for (int isub=0; isub < nsub; ++isub) {
            for (int ilev=0; ilev < nlevels; ++ilev) {
                velocity[ilev]->FillBoundary(m_incflo->Geom(ilev).periodicity());
            }

            SolveStats sub_stats =
                diffuse_velocity_one_step(velocity, density, eta, dt_sub);
            attempt_stats.newton_iters += sub_stats.newton_iters;
            attempt_stats.final_alpha_newton_iters =
                std::max(attempt_stats.final_alpha_newton_iters,
                         sub_stats.final_alpha_newton_iters);
            attempt_stats.final_abs_norm = sub_stats.final_abs_norm;
            attempt_stats.final_rel_norm = sub_stats.final_rel_norm;

            if (!sub_stats.converged) {
                attempt_stats.converged = false;
                break;
            }
        }

        if (attempt_stats.converged) {
            accepted_stats = attempt_stats;
            break;
        }

        if (!m_adaptive_time_substeps || nsub >= m_max_time_substeps) {
            std::stringstream convergenceMsg;
            convergenceMsg << "Newton solver failed to converge during nonlinear "
                              "diffusion with "
                           << nsub << " time substep(s). Relative norm is "
                           << attempt_stats.final_rel_norm
                           << " and the relative tolerance is " << m_newton_rtol
                           << ". Absolute norm is "
                           << attempt_stats.final_abs_norm
                           << " and the absolute tolerance is " << m_newton_atol;
            amrex::Abort(convergenceMsg.str().c_str());
        }
        // Reaching here means solver did NOT converge
        for (int ilev=0; ilev < nlevels; ++ilev) {
            MultiFab::Copy(*velocity[ilev], velocity_save[ilev],
                           0, 0, AMREX_SPACEDIM, velocity[ilev]->nGrow());
        }
        nsub = std::min(2*nsub, m_max_time_substeps);
        retried = true;
        if (m_verbose) {
            amrex::Print() << "Nonlinear diffusion retrying with "
                           << nsub << " time substeps\n";
        }
    }

    if (m_adaptive_time_substeps) {
        if (accepted_stats.final_alpha_newton_iters
            > m_time_substep_newton_iter_high) {
            m_next_time_substeps = std::min(2*nsub, m_max_time_substeps);
        } else if (!retried && accepted_stats.final_alpha_newton_iters
                   < m_time_substep_newton_iter_low) {
            m_next_time_substeps = std::max(m_num_time_substeps, nsub/2);
        } else {
            m_next_time_substeps = nsub;
        }
    }

    if (m_incflo->m_nonlinear_diffusion_dt_control) {
        Real& dt_scale = m_incflo->m_nonlinear_diffusion_dt_scale;
        if (retried || accepted_stats.final_alpha_newton_iters
            > m_time_substep_newton_iter_high) {
            dt_scale = std::max(m_incflo->m_nonlinear_diffusion_dt_scale_min,
                                dt_scale
                                * m_incflo->m_nonlinear_diffusion_dt_scale_shrink);
        } else if (accepted_stats.final_alpha_newton_iters
                   < m_time_substep_newton_iter_low) {
            dt_scale = std::min(Real(1.0),
                                dt_scale
                                * m_incflo->m_nonlinear_diffusion_dt_scale_growth);
        }
        if (m_verbose) {
            amrex::Print() << "Nonlinear diffusion dt scale for next step = "
                           << dt_scale << "\n";
        }
    }
}

NonlinearDiffusionTensorOp::SolveStats
NonlinearDiffusionTensorOp::diffuse_velocity_one_step (
                       Vector<MultiFab*> const& velocity,
                       Vector<MultiFab*> const& density,
                       Vector<MultiFab const*> const& eta,
                       Real dt)
{
    SolveStats total_stats;
    total_stats.converged = true;

    // This function sets the internal member variables. It also initializes
    // iteration 0 velocity to the provided velocity.
    update_member_multifabs(GetVecOfConstPtrs(density),
                            GetVecOfConstPtrs(velocity),
                            GetVecOfConstPtrs(eta), dt);

    if (!m_use_eta_from_prev_time) {
        m_incflo->compute_viscosity(GetVecOfPtrs(m_eta),
                                    GetVecOfPtrs(m_density),
                                    GetVecOfPtrs(m_newton_iter_vel),
                                    m_incflo->m_cur_time, m_nghost_eta);
    }

    for (Real alpha_factor : m_alpha_factor_list) {
        m_alpha_factor = alpha_factor;
        // Update m_newton_iter_func because m_alpha_factor has changed
        compute_viscous_solve_equation(GetVecOfPtrs(m_newton_iter_func),
                                  GetVecOfConstPtrs(m_newton_iter_vel));
        SolveStats stats = diffuse_velocity_alpha_factor(velocity, density,
                                                         eta, dt);
        total_stats.newton_iters += stats.newton_iters;
        total_stats.final_abs_norm = stats.final_abs_norm;
        total_stats.final_rel_norm = stats.final_rel_norm;
        if (alpha_factor == Real(1.0)) {
            total_stats.final_alpha_newton_iters = stats.newton_iters;
        }
        if (!stats.converged && alpha_factor == Real(1.0)) {
            total_stats.converged = false;
            break;
        }
    }

    return total_stats;
}

NonlinearDiffusionTensorOp::SolveStats
NonlinearDiffusionTensorOp::diffuse_velocity_alpha_factor (
                       Vector<MultiFab*> const& velocity,
                       Vector<MultiFab*> const& density,
                       Vector<MultiFab const*> const& eta,
                       Real dt)
{
    SolveStats stats;
    int nlevels = velocity.size();
    // Create a Vector<MultiFab> for RHS of Newton Method
    // This is different from RHS of Viscous solve equation
    Vector<MultiFab> rhs_newton(nlevels);
    // Create a Vector<MultiFab> for vel_incrment from Newton Method
    Vector<MultiFab> vel_incrmt_newton(nlevels);
    for (int ilev=0; ilev < nlevels; ++ilev) {
        rhs_newton[ilev].define(velocity[ilev]->boxArray(),
                                velocity[ilev]->DistributionMap(),
                                AMREX_SPACEDIM,0,MFInfo(),
                                velocity[ilev]->Factory());

        vel_incrmt_newton[ilev].define(velocity[ilev]->boxArray(),
                                velocity[ilev]->DistributionMap(),
                                AMREX_SPACEDIM,0,MFInfo(),
                                velocity[ilev]->Factory());

        vel_incrmt_newton[ilev].setVal(Real(0.));
    }
    // Different tolerance when m_alpha_factor is NOT unity
    const Real tol_scale =
        (m_alpha_factor == Real(1.0)) ? Real(1.0) : m_nonunity_alpha_tol_scale;
    const Real newton_rtol = tol_scale * m_newton_rtol;
    const Real newton_atol = tol_scale * m_newton_atol;
    const Real gmres_rtol  = tol_scale * m_gmres_rtol;
    const Real gmres_atol  = tol_scale * m_gmres_atol;
    // Look into WarpX NewtonSolver for stopping criterion
    Real norm_abs = Real(0.);
    Real norm0    = Real(1.);
    Real norm_rel = Real(0.);
    bool converged = false;
    bool residual_grew = false;
    int inewt;
    for (inewt=0; inewt < m_newton_max_iter;) {
        // Evaluate current residual's norm
        norm_abs = get_norm_of_residual();
        if (inewt == 0) {
            if (norm_abs > Real(0.)) {
                norm0 = norm_abs;
            }
            else {
                norm0 = Real(1.0);
            }
        }
        norm_rel = norm_abs/norm0;
        // Check for convergence criteria; Copied from WarpX - NewtonSolver.H
        if (m_verbose || inewt == m_newton_max_iter) {
            amrex::Print() << "Newton: iteration = " << std::setw(3) << inewt <<  ", norm = "
                 << std::scientific << std::setprecision(5) << norm_abs << " (abs.), "
                 << std::scientific << std::setprecision(5) << norm_rel << " (rel.)" << "\n";
        }

        if (norm_abs < newton_atol) {
            converged = true;
            if (m_verbose) {
                amrex::Print() << "Newton: exiting at iteration = " << std::setw(3) << inewt
                               << ". Satisfied absolute tolerance " << newton_atol << "\n";
            }
            break;
        }

        if (norm_rel < newton_rtol) {
            converged = true;
            if (m_verbose) {
                amrex::Print() << "Newton: exiting at iteration = " << std::setw(3) << inewt
                               << ". Satisfied relative tolerance " << newton_rtol << "\n";
            }
            break;
        }

        if (norm_abs > Real(100.)*norm0) {
            residual_grew = true;
            amrex::Print() << "Newton: exiting at iteration = " << std::setw(3) << inewt
                 << ". SOLVER DIVERGED! relative tolerance = " << norm_rel << "\n";
            break;
        }
        // Update RHS of Newton Iteration
        for (int ilev=0; ilev < nlevels; ++ilev) {
            MultiFab::Copy(rhs_newton[ilev], *m_newton_iter_func[ilev],
                           0,0,AMREX_SPACEDIM,0);
            // Need to negate
            rhs_newton[ilev].mult(Real(-1.0),0);
        }
        m_gmres->solve(vel_incrmt_newton,rhs_newton,
                       gmres_rtol,gmres_atol);

        update_newton_iteration_multifabs(
                      GetVecOfConstPtrs(vel_incrmt_newton));
        inewt++;
        if (inewt >= m_newton_max_iter) {
            if (m_verbose) {
                amrex::Print() << "Newton: exiting at iter = " << std::setw(3) << inewt
                     << ". Maximum iteration reached: iter = " << m_newton_max_iter << "\n";
            }
            break;
        }
    }  // end of Newton Iteration loop

    stats.converged = converged && !residual_grew;
    stats.newton_iters = inewt;
    stats.final_alpha_newton_iters =
        (m_alpha_factor == Real(1.0)) ? inewt : 0;
    stats.final_abs_norm = norm_abs;
    stats.final_rel_norm = norm_rel;

    if (!stats.converged && m_alpha_factor == Real(1.0)) {
        return stats;
    }

    // Copy final newton iteration velocity
    for (int ilev=0; ilev < nlevels; ++ilev) {
        MultiFab::Copy(*velocity[ilev],*m_newton_iter_vel[ilev],
                       0, 0, AMREX_SPACEDIM, m_nghost_vel);
    }

    return stats;
}

void NonlinearDiffusionTensorOp::compute_divtau (
                         Vector<MultiFab*> const& divtau,
                         Vector<MultiFab const*> const& velocity,
                         Vector<MultiFab const*> const& density,
                         Vector<MultiFab const*> const& eta)
{
    // Preparation for two_fluid scenario
    Vector<std::unique_ptr<MultiFab>> conc_second, p_static;
    if (m_incflo->m_two_fluid) {
        int nlevels = velocity.size();
        conc_second.resize(nlevels);
        p_static.resize(nlevels);
        for (int ilev = 0; ilev < nlevels; ++ilev) {
            conc_second[ilev] = std::make_unique<MultiFab>(
                                            eta[ilev]->boxArray(),
                                            eta[ilev]->DistributionMap(),
                                            1, m_nghost_eta, MFInfo(),
                                            eta[ilev]->Factory());
            p_static[ilev] = std::make_unique<MultiFab>(
                                            eta[ilev]->boxArray(),
                                            eta[ilev]->DistributionMap(),
                                            1, 0, MFInfo(),
                                            eta[ilev]->Factory());
           if (m_incflo->m_nodal_vel_eta) {
              m_incflo->compute_nodal_second_fluid_conc(conc_second[ilev].get(),
                                                        density[ilev],
                                                        m_nghost_eta);
              m_incflo->compute_nodal_hydrostatic_pressure_at_level(
                            ilev, p_static[ilev].get(), density[ilev],
                            m_incflo->m_mu_p_surf_second,
                            m_incflo->Geom(ilev),0);
           }
           else
           {
              m_incflo->compute_cc_second_fluid_conc(conc_second[ilev].get(),
                                                     density[ilev],
                                                     m_nghost_eta);
              m_incflo->compute_cc_hydrostatic_pressure_at_level(
                            ilev, p_static[ilev].get(), density[ilev],
                            m_incflo->m_mu_p_surf_second,
                            m_incflo->Geom(ilev),0);
           }
        }
    }
    // Evaluation of linear and nonlinear divtau contributions
    int finest_level = velocity.size()-1;
#ifdef AMREX_USE_EB
    if (m_eb_apply_op)
    {
        Vector<MultiFab> divtau_tmp(finest_level+1);
        int tmp_comp = (m_incflo->m_redistribution_type == "StateRedist") ? 3 : 2;
        for (int lev = 0; lev <= finest_level; ++lev) {
            divtau_tmp[lev].define(divtau[lev]->boxArray(),
                                   divtau[lev]->DistributionMap(),
                                   AMREX_SPACEDIM, tmp_comp, MFInfo(),
                                   divtau[lev]->Factory());
            divtau_tmp[lev].setVal(0.0);
        }
        compute_linear_part_of_divtau(GetVecOfPtrs(divtau_tmp), velocity,
                                      density, eta);
        // Here velocity is used as old_velocity as well;
        // Reason: This is NOT used in implicit solve
        add_non_linear_part_of_divtau(GetVecOfPtrs(divtau_tmp), velocity,
                                      density,
                                      GetVecOfConstPtrs(conc_second),
                                      GetVecOfConstPtrs(p_static),
                                      velocity);
        // Redistribution
        for(int lev = 0; lev <= finest_level; lev++)
        {
           amrex::single_level_redistribute(divtau_tmp[lev],
                   *divtau[lev], 0, AMREX_SPACEDIM, m_incflo->Geom(lev));
        }
    }
    else
#endif
    {
        compute_linear_part_of_divtau(divtau, velocity, density, eta);
        // Here velocity is used as old_velocity as well;
        // Reason: This is NOT used in implicit solve
        add_non_linear_part_of_divtau(divtau, velocity, density,
                                      GetVecOfConstPtrs(conc_second),
                                      GetVecOfConstPtrs(p_static),
                                      velocity);
    }

    // This is to be consistent with incflo code
    bool advect_momentum = m_incflo->AdvectMomentum();
    if (!advect_momentum) {
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (int lev = 0; lev <= finest_level; ++lev) {
            for (MFIter mfi(*divtau[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                Box const& bx = mfi.tilebox();
                Array4<Real> const& divtau_arr = divtau[lev]->array(mfi);
                Array4<Real const> const& rho_arr = density[lev]->const_array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real rhoinv = Real(1.0)/rho_arr(i,j,k);
                    AMREX_D_TERM(divtau_arr(i,j,k,0) *= rhoinv;,
                                 divtau_arr(i,j,k,1) *= rhoinv;,
                                 divtau_arr(i,j,k,2) *= rhoinv;);
                });
            } // mfi
        } // lev
    } // not m_advect_momentum
}

void NonlinearDiffusionTensorOp::compute_viscous_solve_equation (
                       Vector<MultiFab*> const& nonlin_func,
                       Vector<MultiFab const*> const& velocity)
{
    auto const& ho_coeff_velocity = m_use_ho_coeff_from_prev_time
        ? GetVecOfConstPtrs(m_old_iter_vel)
        : GetVecOfConstPtrs(m_newton_iter_vel);
    int numcomp = nonlin_func[0]->nComp();
#ifdef AMREX_USE_EB
    if (m_eb_apply_op)
    {
        int nlevels = nonlin_func.size();
        Vector<MultiFab> divtau_tmp(nlevels);
        int ng_redist = 2;
        for (int lev = 0; lev < nlevels; ++lev) {
            divtau_tmp[lev].define(nonlin_func[lev]->boxArray(),
                                   nonlin_func[lev]->DistributionMap(),
                                   numcomp, ng_redist, MFInfo(),
                                   nonlin_func[lev]->Factory());
            divtau_tmp[lev].setVal(Real(0.));
        }

        compute_linear_part_of_divtau(GetVecOfPtrs(divtau_tmp), velocity,
                                      GetVecOfConstPtrs(m_density),
                                      GetVecOfConstPtrs(m_eta));
        add_non_linear_part_of_divtau(GetVecOfPtrs(divtau_tmp), velocity,
                                      GetVecOfConstPtrs(m_density),
                                      GetVecOfConstPtrs(m_conc_second),
                                      GetVecOfConstPtrs(m_p_static),
                                      ho_coeff_velocity);
        for (int lev = 0; lev < nlevels; ++lev) {
            divtau_tmp[lev].FillBoundary(m_incflo->Geom(lev).periodicity());
            amrex::single_level_redistribute(divtau_tmp[lev],
                                             *nonlin_func[lev], 0, numcomp,
                                             m_incflo->Geom(lev));
        }
    }
    else
#endif
    {
        compute_linear_part_of_divtau(nonlin_func, velocity,
                                      GetVecOfConstPtrs(m_density),
                                      GetVecOfConstPtrs(m_eta));
        add_non_linear_part_of_divtau(nonlin_func, velocity,
                                      GetVecOfConstPtrs(m_density),
                                      GetVecOfConstPtrs(m_conc_second),
                                      GetVecOfConstPtrs(m_p_static),
                                      ho_coeff_velocity);
    }
    // First multiply divtau with (-dt)
    scale(nonlin_func, Real(-1.0)*m_dt);
    increment(nonlin_func, GetVecOfConstPtrs(m_rhs_n), Real(-1.0));
    for (int idim=0; idim < numcomp; ++idim) {
        AddProduct(nonlin_func, GetVecOfConstPtrs(m_density),
                    0, velocity, idim, idim, 1);
    }
}

// This function computes the linear part of divtau using MLTensorOp
void NonlinearDiffusionTensorOp::compute_linear_part_of_divtau (Vector<MultiFab*> const& a_divtau,
                                        Vector<MultiFab const*> const& a_velocity,
                                        Vector<MultiFab const*> const& a_density,
                                        Vector<MultiFab const*> const& a_eta)
{
    BL_PROFILE("NonlinearDiffusionTensorOp::compute_linear_part_of_divtau");

    int finest_level = m_incflo->finestLevel();

    Vector<MultiFab> velocity(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) {
        velocity[lev].define(a_velocity[lev]->boxArray(),
                             a_velocity[lev]->DistributionMap(),
                             AMREX_SPACEDIM, 1, MFInfo(),
                             a_velocity[lev]->Factory());
        MultiFab::Copy(velocity[lev], *a_velocity[lev], 0, 0, AMREX_SPACEDIM, 1);
    }

#ifdef AMREX_USE_EB
    if (m_eb_apply_op)
    {
        // We want to return div (mu grad)) phi
        m_eb_apply_op->setScalars(0.0, -1.0);

        // For when we use the stencil for centroid values
        // m_eb_apply_op->setPhiOnCentroid();

        for (int lev = 0; lev <= finest_level; ++lev) {
            m_eb_apply_op->setACoeffs(lev, *a_density[lev]);

            Array<MultiFab,AMREX_SPACEDIM> b = m_incflo->average_velocity_eta_to_faces(lev, *a_eta[lev]);

            m_eb_apply_op->setShearViscosity(lev, GetArrOfConstPtrs(b), MLMG::Location::FaceCentroid);

            if (m_incflo->hasEBFlow()) {
               m_eb_apply_op->setEBShearViscosityWithInflow(lev, *a_eta[lev], *(m_incflo->get_velocity_eb()[lev]));
            } else {
               m_eb_apply_op->setEBShearViscosity(lev, *a_eta[lev]);
            }
            m_eb_apply_op->setLevelBC(lev, &velocity[lev]);
        }

        MLMG mlmg(*m_eb_apply_op);
        mlmg.apply(a_divtau, GetVecOfPtrs(velocity));
    }
    else
#endif
    {
        // We want to return div (mu grad)) phi
        m_reg_apply_op->setScalars(0.0, -1.0);
        for (int lev = 0; lev <= finest_level; ++lev) {
            m_reg_apply_op->setACoeffs(lev, *a_density[lev]);
            Array<MultiFab,AMREX_SPACEDIM> b;
            if (a_eta[lev]->boxArray().ixType().nodeCentered()) {
                b = incflo::average_nodal_velocity_eta_to_faces(lev, *a_eta[lev]);
            }
            else {
                b = m_incflo->average_velocity_eta_to_faces(lev, *a_eta[lev]);
            }
            m_reg_apply_op->setShearViscosity(lev, GetArrOfConstPtrs(b));
            m_reg_apply_op->setLevelBC(lev, &velocity[lev]);
        }

        MLMG mlmg(*m_reg_apply_op);
        mlmg.apply(a_divtau, GetVecOfPtrs(velocity));
    }
}

// This function adds the non-linear part of divtau
// For now, only performed if it is two-fluid
void NonlinearDiffusionTensorOp::add_non_linear_part_of_divtau (Vector<MultiFab*> const& a_divtau,
                                        Vector<MultiFab const*> const& a_velocity,
                                        Vector<MultiFab const*> const& a_density,
                                        Vector<MultiFab const*> const& a_conc_second,
                                        Vector<MultiFab const*> const& a_p_static,
                                        Vector<MultiFab const*> const& a_old_velocity)
{
    if (!(m_incflo->m_two_fluid)) {
        return;
    }
    int nlevels = a_velocity.size();
    auto loc = MLMG::Location::FaceCentroid;
    bool already_on_centroids = (loc == MLMG::Location::FaceCentroid);
    Vector<MultiFab> ho_divtau;
    ho_divtau.resize(nlevels);
    for (int ilev = 0; ilev < nlevels; ++ilev) {
        ho_divtau[ilev].define(a_divtau[ilev]->boxArray(),
                               a_divtau[ilev]->DistributionMap(),
                               AMREX_SPACEDIM, a_divtau[ilev]->nGrow(),
                               MFInfo(), a_divtau[ilev]->Factory());
        ho_divtau[ilev].setVal(Real(0.));
        // Calculate second-order rheology coefficients
        // USING OLD VELOCITY with 1 ghost cell
        MultiFab scndOrderCoeff(a_old_velocity[ilev]->boxArray(),
                                a_old_velocity[ilev]->DistributionMap(),
                                m_ncomp_ho,1, MFInfo(),
                                a_old_velocity[ilev]->Factory());
        scndOrderCoeff.setVal(0.);
        // This function takes care of ghost cells
        m_incflo->compute_second_order_coeff(ilev,
                                scndOrderCoeff, *a_old_velocity[ilev],
                                *a_density[ilev], *a_conc_second[ilev],
                                *a_p_static[ilev], m_incflo->Geom(ilev));
        MultiFab velocity_tmp(a_velocity[ilev]->boxArray(),
                              a_velocity[ilev]->DistributionMap(),
                              AMREX_SPACEDIM, a_velocity[ilev]->nGrow(),
                              MFInfo(),a_velocity[ilev]->Factory());
        MultiFab::Copy(velocity_tmp, *a_velocity[ilev], 0, 0, AMREX_SPACEDIM,
                       a_velocity[ilev]->nGrow());

        Array<MultiFab, AMREX_SPACEDIM> gradVel;
        for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
            gradVel[idim].define(amrex::convert(a_velocity[ilev]->boxArray(),
                                 IntVect::TheDimensionVector(idim)),
                                 a_velocity[ilev]->DistributionMap(),
                                 AMREX_SPACEDIM*AMREX_SPACEDIM, 0,
                                 MFInfo(),a_velocity[ilev]->Factory());
            gradVel[idim].setVal(Real(0.));
        }
#ifdef AMREX_USE_EB
        MultiFab gradVel_EB;
        if (m_eb_apply_op)
        {
            gradVel_EB.define(a_velocity[ilev]->boxArray(),
                              a_velocity[ilev]->DistributionMap(),
                              AMREX_SPACEDIM*AMREX_SPACEDIM, 0,
                              MFInfo(),a_velocity[ilev]->Factory());
        }
        compVelGrad(ilev, velocity_tmp, loc, amrex::GetArrOfPtrs(gradVel),
                    &gradVel_EB);
        m_incflo->compute_granular_high_order_divtau_on_level(ilev, ho_divtau[ilev],
                   amrex::GetArrOfConstPtrs(gradVel), &gradVel_EB,
                   scndOrderCoeff, already_on_centroids);
#else
        compVelGrad(ilev, velocity_tmp, loc, amrex::GetArrOfPtrs(gradVel));
        m_incflo->compute_granular_high_order_divtau_on_level(ilev, ho_divtau[ilev],
                   amrex::GetArrOfConstPtrs(gradVel),
                   scndOrderCoeff, already_on_centroids);
#endif
    }
    // Increment only in valid and non-covered cells
    Real alpha_factor = m_alpha_factor;
    increment(a_divtau, GetVecOfConstPtrs(ho_divtau), alpha_factor);
}

void NonlinearDiffusionTensorOp::update_member_multifabs (
         Vector<MultiFab const*> const& a_density,
         Vector<MultiFab const*> const& a_vel,
         Vector<MultiFab const*> const& a_eta,
         Real a_dt)
{
    m_dt = a_dt;
    int nlevels = a_density.size();
    // This is to initialize level 0 only the first time
    const int lev_start = m_density.size() ? 1 : 0;
    // To take into account if finer levels change
    m_rhs_n.resize(nlevels);
    m_density.resize(nlevels);
    m_eta.resize(nlevels);
    if (m_incflo->m_two_fluid) {
        m_conc_second.resize(nlevels);
        m_p_static.resize(nlevels);
    }
    m_newton_iter_vel.resize(nlevels);
    m_old_iter_vel.resize(nlevels);
    m_newton_iter_func.resize(nlevels);
    for (int ilev = lev_start; ilev < nlevels; ++ilev) {
        m_rhs_n[ilev] = std::make_unique<MultiFab>(a_vel[ilev]->boxArray(),
                                        a_vel[ilev]->DistributionMap(),
                                        AMREX_SPACEDIM, 0, MFInfo(),
                                        a_vel[ilev]->Factory());

        m_density[ilev] = std::make_unique<MultiFab>(
                                        a_density[ilev]->boxArray(),
                                        a_density[ilev]->DistributionMap(),
                                        1, m_nghost_density, MFInfo(),
                                        a_density[ilev]->Factory());

        m_eta[ilev] = std::make_unique<MultiFab>(
                                        a_eta[ilev]->boxArray(),
                                        a_eta[ilev]->DistributionMap(),
                                        1, m_nghost_eta, MFInfo(),
                                        a_eta[ilev]->Factory());
        if (m_incflo->m_two_fluid) {
            m_conc_second[ilev] = std::make_unique<MultiFab>(
                                            a_eta[ilev]->boxArray(),
                                            a_eta[ilev]->DistributionMap(),
                                            1, m_nghost_eta, MFInfo(),
                                            a_eta[ilev]->Factory());

            m_p_static[ilev] = std::make_unique<MultiFab>(
                                            a_eta[ilev]->boxArray(),
                                            a_eta[ilev]->DistributionMap(),
                                            1, 0, MFInfo(),
                                            a_eta[ilev]->Factory());
        }

        m_newton_iter_vel[ilev] = std::make_unique<MultiFab>(
                                        a_vel[ilev]->boxArray(),
                                        a_vel[ilev]->DistributionMap(),
                                        AMREX_SPACEDIM, m_nghost_vel,
                                        MFInfo(),a_vel[ilev]->Factory());

        m_old_iter_vel[ilev] = std::make_unique<MultiFab>(
                                        a_vel[ilev]->boxArray(),
                                        a_vel[ilev]->DistributionMap(),
                                        AMREX_SPACEDIM, m_nghost_vel,
                                        MFInfo(),a_vel[ilev]->Factory());

        m_newton_iter_func[ilev] = std::make_unique<MultiFab>(
                                        a_vel[ilev]->boxArray(),
                                        a_vel[ilev]->DistributionMap(),
                                        AMREX_SPACEDIM, 0, MFInfo(),
                                        a_vel[ilev]->Factory());
    }
    // Update the member variables
    for (int ilev = 0; ilev < nlevels; ++ilev) {
        MultiFab::Copy(*m_density[ilev],*a_density[ilev],
                       0,0,1,m_nghost_density);

        MultiFab::Copy(*m_eta[ilev],*a_eta[ilev],
                       0,0,1,m_nghost_eta);

        MultiFab::Copy(*m_newton_iter_vel[ilev],*a_vel[ilev],
                       0,0,AMREX_SPACEDIM,m_nghost_vel);

        // Store the initial velocity passed into diffuse_velocity.
        // This stays fixed for the full Newton solve when requested.
        MultiFab::Copy(*m_old_iter_vel[ilev],*a_vel[ilev],
                       0,0,AMREX_SPACEDIM,m_nghost_vel);

        MultiFab::Copy(*m_rhs_n[ilev],*a_vel[ilev],
                       0,0,AMREX_SPACEDIM,0);

        for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
            MultiFab::Multiply(*m_rhs_n[ilev], *a_density[ilev],
                               0,idim,1,0);
        }
    }
    if (m_incflo->m_two_fluid) {
        auto a_conc_second = GetVecOfPtrs(m_conc_second);
        auto a_p_static    = GetVecOfPtrs(m_p_static);
        for (int ilev = 0; ilev < nlevels; ++ilev) {
           if (m_incflo->m_nodal_vel_eta) {
              m_incflo->compute_nodal_second_fluid_conc(a_conc_second[ilev],
                                                        a_density[ilev],
                                                        m_nghost_eta);
              m_incflo->compute_nodal_hydrostatic_pressure_at_level(
                            ilev, a_p_static[ilev], a_density[ilev],
                            m_incflo->m_mu_p_surf_second,
                            m_incflo->Geom(ilev),0);
           }
           else
           {
              m_incflo->compute_cc_second_fluid_conc(a_conc_second[ilev],
                                                      a_density[ilev],
                                                      m_nghost_eta);
              m_incflo->compute_cc_hydrostatic_pressure_at_level(
                            ilev, a_p_static[ilev], a_density[ilev],
                            m_incflo->m_mu_p_surf_second,
                            m_incflo->Geom(ilev),0);
           }
        }
    }
    // Update m_newton_iter_func
    compute_viscous_solve_equation(GetVecOfPtrs(m_newton_iter_func),
                                   GetVecOfConstPtrs(m_newton_iter_vel));
}

// This function should NOT touch physical boundaries and covered cells
void NonlinearDiffusionTensorOp::update_newton_iteration_multifabs (
                Vector<MultiFab const*> const& a_vel_increment)
{
    int nlevels = a_vel_increment.size();
    Real norm_old, norm_new;
    norm_old = get_norm_of_residual();
    // Add the incremental velocity without ghost and covered cells
    increment(GetVecOfPtrs(m_newton_iter_vel),
              a_vel_increment, Real(1.0));
    for (int ilev=0; ilev < nlevels; ++ilev) {
        // Use FillBoundary to update ghost cells
        m_newton_iter_vel[ilev]->FillBoundary(
                          m_incflo->Geom(ilev).periodicity());
    }
    // Update m_newton_iter_func
    compute_viscous_solve_equation(GetVecOfPtrs(m_newton_iter_func),
                                   GetVecOfConstPtrs(m_newton_iter_vel));
    norm_new = get_norm_of_residual();

    if (norm_new >= norm_old) {
        // Update using a factor of lambda
        Real lambda_1 = Real(1.0); Real lambda_2;
        Real alpha  = m_newton_update_alpha;
        Real beta;
        for (int iter=1; iter <= m_newton_update_max_iter; ++iter) {
            lambda_2 = (Real(1.0) - alpha)*lambda_1;
            beta = lambda_2 - lambda_1;
            lambda_1 = lambda_2;
            // Remove portion of the incremental velocity without ghost and covered cells
            increment(GetVecOfPtrs(m_newton_iter_vel),
                      a_vel_increment, beta);
            for (int ilev=0; ilev < nlevels; ++ilev) {
                // Use FillBoundary to update ghost cells
                m_newton_iter_vel[ilev]->FillBoundary(
                                  m_incflo->Geom(ilev).periodicity());
            }
            // Update m_newton_iter_func
            compute_viscous_solve_equation(GetVecOfPtrs(m_newton_iter_func),
                                           GetVecOfConstPtrs(m_newton_iter_vel));
            norm_new = get_norm_of_residual();
            if (norm_new < norm_old) {
                break;
            }
        }
    }

    if (!m_use_eta_from_prev_time) {
        // Update m_eta based on m_newton_iter_vel
        m_incflo->compute_viscosity(GetVecOfPtrs(m_eta),
                                    GetVecOfPtrs(m_density),
                                    GetVecOfPtrs(m_newton_iter_vel),
                                    m_incflo->m_cur_time, m_nghost_eta);
    }
}

// This function calculates norm2 for the non-linear function
Real NonlinearDiffusionTensorOp::get_norm_of_residual ()
{
    return norm2(GetVecOfConstPtrs(m_newton_iter_func));
}

void NonlinearDiffusionTensorOp::compute_preconditioner_eta (
                            Vector<std::unique_ptr<MultiFab>>& eta_precond,
                            Vector<MultiFab const*> const& eta)
{
    const int finest_level = m_incflo->finestLevel();
    auto const& ho_coeff_velocity = m_use_ho_coeff_from_prev_time
        ? GetVecOfConstPtrs(m_old_iter_vel)
        : GetVecOfConstPtrs(m_newton_iter_vel);

    eta_precond.resize(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
        const int ng = eta[lev]->nGrow();
        eta_precond[lev] = std::make_unique<MultiFab>(
            eta[lev]->boxArray(), eta[lev]->DistributionMap(),
            1, ng, MFInfo(), eta[lev]->Factory());
        MultiFab::Copy(*eta_precond[lev], *eta[lev], 0, 0, 1, ng);

        if (!m_incflo->m_two_fluid || eta[lev]->boxArray().ixType().nodeCentered()) {
            continue;
        }

        MultiFab scndOrderCoeff(eta[lev]->boxArray(), eta[lev]->DistributionMap(),
                                m_ncomp_ho, 0, MFInfo(), eta[lev]->Factory());
        m_incflo->compute_second_order_coeff(lev, scndOrderCoeff,
                                *ho_coeff_velocity[lev],
                                *m_density[lev], *m_conc_second[lev],
                                *m_p_static[lev], m_incflo->Geom(lev));

        MultiFab strainrate(eta[lev]->boxArray(), eta[lev]->DistributionMap(),
                            1, 0, MFInfo(), eta[lev]->Factory());
        m_incflo->compute_strainrate_at_level(lev, &strainrate,
                                ho_coeff_velocity[lev],
                                m_incflo->Geom(lev), Real(0.0), 0);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*eta_precond[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Box const& bx = mfi.tilebox();
            Array4<Real      > const& eta_arr = eta_precond[lev]->array(mfi);
            Array4<Real const> const& c2_arr  = scndOrderCoeff.const_array(mfi);
            Array4<Real const> const& sr_arr  = strainrate.const_array(mfi);
            const Real alpha_factor = m_alpha_factor;
            const Real eps = m_incflo->m_mu_sr_eps_second;
            const int ncomp_ho = m_ncomp_ho;

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Note: sr_mf contains TWO TIMES strain rate
                Real Dmag = Real(0.5)*sr_arr(i,j,k) + eps;
                for (int n=0; n < ncomp_ho; n++) {
                    eta_arr(i,j,k) += alpha_factor*c2_arr(i,j,k,n)*Dmag;
                }
            });
        }

        eta_precond[lev]->FillBoundary(m_incflo->Geom(lev).periodicity());
    }
}

// This function is used in the precond of GMRES
void NonlinearDiffusionTensorOp::diffuse_velocity_mlmg (
                            Vector<MultiFab*> const& velocity,
                            Vector<MultiFab*> const& density,
                            Vector<MultiFab const*> const& eta,
                            Vector<MultiFab const*> const& rhs,
                            Real dt)
{
    //
    //      alpha a - beta div ( b grad )   <--->   rho - dt div ( mu grad )
    //
    // So the constants and variable coefficients are:
    //
    //      alpha: 1
    //      beta: dt
    //      a: rho
    //      b: mu
    const int finest_level = m_incflo->finestLevel();
    Vector<std::unique_ptr<MultiFab>> eta_precond;
    Vector<MultiFab const*> eta_mlmg = eta;
    if (m_use_ho_eta_precond && m_incflo->m_two_fluid) {
        compute_preconditioner_eta(eta_precond, eta);
        eta_mlmg = GetVecOfConstPtrs(eta_precond);
    }
#ifdef AMREX_USE_EB
    if (m_eb_solve_op)
    {
        // For when we use the stencil for centroid values
        // m_eb_solve_op->setPhiOnCentroid();

        m_eb_solve_op->setScalars(1.0, dt);
        for (int lev = 0; lev <= finest_level; ++lev) {
            m_eb_solve_op->setACoeffs(lev, *density[lev]);

            Array<MultiFab,AMREX_SPACEDIM> b = m_incflo->average_velocity_eta_to_faces(lev, *eta_mlmg[lev]);

            m_eb_solve_op->setShearViscosity(lev, GetArrOfConstPtrs(b), MLMG::Location::FaceCentroid);

            //if (m_incflo->hasEBFlow()) {
            //   m_eb_solve_op->setEBShearViscosityWithInflow(lev, *eta_mlmg[lev], *(m_incflo->get_velocity_eb()[lev]));
            //} else {
            //   m_eb_solve_op->setEBShearViscosity(lev, *eta_mlmg[lev]);
            //}
            // Preconditioner is solving for delta u; so DIRICHLET IS ALWAYS HOMOGENEOUS
            m_eb_solve_op->setEBShearViscosity(lev, *eta_mlmg[lev]);
        }
    }
    else
#endif
    {
        m_reg_solve_op->setScalars(1.0, dt);
        for (int lev = 0; lev <= finest_level; ++lev) {
            m_reg_solve_op->setACoeffs(lev, *density[lev]);
            Array<MultiFab,AMREX_SPACEDIM> b;
            if (eta_mlmg[lev]->boxArray().ixType().nodeCentered()) {
                b = incflo::average_nodal_velocity_eta_to_faces(lev, *eta_mlmg[lev]);
            }
            else {
                b = m_incflo->average_velocity_eta_to_faces(lev, *eta_mlmg[lev]);
            }
            m_reg_solve_op->setShearViscosity(lev, GetArrOfConstPtrs(b));
        }
    }

    for (int lev = 0; lev <= finest_level; ++lev) {
#ifdef AMREX_USE_EB
        if (m_eb_solve_op) {
            m_eb_solve_op->setLevelBC(lev, velocity[lev]);
        } else
#endif
        {
            m_reg_solve_op->setLevelBC(lev, velocity[lev]);
        }
    }

#ifdef AMREX_USE_EB
    MLMG mlmg(m_eb_solve_op ? static_cast<MLLinOp&>(*m_eb_solve_op)
              :               static_cast<MLLinOp&>(*m_reg_solve_op));
#else
    MLMG mlmg(*m_reg_solve_op);
#endif

    // The default bottom solver is BiCG
    if (m_bottom_solver == "smoother")
    {
        mlmg.setBottomSolver(MLMG::BottomSolver::smoother);
    }
    else if (m_bottom_solver == "hypre")
    {
        mlmg.setBottomSolver(MLMG::BottomSolver::hypre);
    }
    // Maximum iterations for MultiGrid / ConjugateGradients
    mlmg.setMaxIter(m_mg_max_iter);
    mlmg.setFixedIter(m_mg_max_iter);
    mlmg.setMaxFmgIter(m_mg_max_fmg_iter);
    mlmg.setBottomMaxIter(m_mg_bottom_maxiter);

    // Verbosity for MultiGrid / ConjugateGradients
    mlmg.setVerbose(m_mg_verbose);
    mlmg.setBottomVerbose(m_mg_bottom_verbose);

    mlmg.setPreSmooth(m_num_pre_smooth);
    mlmg.setPostSmooth(m_num_post_smooth);

    mlmg.solve(velocity, rhs, m_mg_rtol, m_mg_atol);
}

// Putting everything needed by GMRES below
// All these need to be public member functions
// Jv corresponds to matrix vector product of Jacobian and increment
// NOTE: norm2 calculations might need to be changed when it becomes multi-level
void NonlinearDiffusionTensorOp::apply (VMF& Jv, VMF& v)
{
    int numcomp = v[0].nComp();
    int nlevels = v.size();
    Real eps_newton, old_vel_norm2, vel_incrmt_norm2;
    old_vel_norm2 = norm2(GetVecOfConstPtrs(m_newton_iter_vel));
    vel_incrmt_norm2 = norm2(v);

    eps_newton = m_newton_epsilon;
    eps_newton *= (Real(1.0) + old_vel_norm2);
    eps_newton = std::sqrt(eps_newton);
    eps_newton /= (vel_incrmt_norm2 + Real(1.0e-18));

    Vector<MultiFab> vel_jacobian(nlevels);
    for (int ilev=0; ilev < nlevels; ++ilev) {
        vel_jacobian[ilev].define(v[ilev].boxArray(),
                                  v[ilev].DistributionMap(),
                                  numcomp,m_nghost_vel, MFInfo(),
                                  v[ilev].Factory());
        // Old iteration velocity has correct physical boundary
        // information so copy from its ghost cells
        MultiFab::Copy(vel_jacobian[ilev],*m_newton_iter_vel[ilev],
                       0,0,numcomp,m_nghost_vel);
    }
    // Add increment only to valid and non-covered cells
    increment(GetVecOfPtrs(vel_jacobian),GetVecOfConstPtrs(v),eps_newton);
    // FillBoundary call for interior/periodic ghost cells
    for (int ilev=0; ilev < nlevels; ++ilev) {
        vel_jacobian[ilev].FillBoundary(m_incflo->Geom(ilev).periodicity());
    }
    compute_viscous_solve_equation(GetVecOfPtrs(Jv),
                                   GetVecOfConstPtrs(vel_jacobian));

    increment(GetVecOfPtrs(Jv),
              GetVecOfConstPtrs(m_newton_iter_func), Real(-1.0));
    scale(Jv, Real(1.0)/(eps_newton+Real(1.0e-18)));
}

// Does NOT touch ghost and covered cells
void NonlinearDiffusionTensorOp::assign (VMF& lhs,
                                         VMF const& rhs)
{
    assign(GetVecOfPtrs(lhs), GetVecOfConstPtrs(rhs));
}

// Do NOT touch ghost and covered cells
void NonlinearDiffusionTensorOp::assign (VMFPtr const& lhs,
                                         VCMFPtr const& rhs)
{
    int numcomp = rhs[0]->nComp();
    int nlevels = rhs.size();
    for (int ilev=0; ilev < nlevels; ++ilev) {
#ifdef AMREX_USE_EB
        const auto& factory =
          dynamic_cast<EBFArrayBoxFactory const&>(rhs[ilev]->Factory());
        auto const& flags = factory.getMultiEBCellFlagFab();
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*rhs[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            auto const& flag_fab = flags[mfi];
            auto const& flag_arr = flag_fab.const_array();
            Array4<Real      > const& lhs_arr = lhs[ilev]->array(mfi);
            Array4<Real const> const& rhs_arr = rhs[ilev]->const_array(mfi);
            ParallelFor(bx, numcomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                if (!flag_arr(i,j,k).isCovered()) {
                    lhs_arr(i,j,k,n) = rhs_arr(i,j,k,n);
                }
            });
        }
#else
        MultiFab::Copy(*lhs[ilev],*rhs[ilev],0,0,numcomp,0);
#endif
    }
}

Real NonlinearDiffusionTensorOp::dotProduct (VMF const& v1,
                                             VMF const& v2)
{
    return dotProduct(GetVecOfConstPtrs(v1),
                      GetVecOfConstPtrs(v2));
}

// Do NOT touch ghost and covered cells
Real NonlinearDiffusionTensorOp::dotProduct (VCMFPtr const& v1,
                                             VCMFPtr const& v2)
{
    Real dot_all_lev = Real(0.);
    Real dot_lev;
    int numcomp = v1[0]->nComp();
    int nlevels = v1.size();
    for (int ilev=0; ilev < nlevels; ++ilev)
    {
#ifdef AMREX_USE_EB
        const auto& factory =
          dynamic_cast<EBFArrayBoxFactory const&>(v1[ilev]->Factory());
        auto const& v1_arrays    = v1[ilev]->const_arrays();
        auto const& v2_arrays    = v2[ilev]->const_arrays();
        auto const& flag_arrays = factory.getMultiEBCellFlagFab().const_arrays();
        dot_lev = amrex::ParReduce(TypeList<ReduceOpSum>{}, TypeList<Real>{},
                                     *v1[ilev],
                    [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
                    noexcept -> GpuTuple<Real>
                    {
                        if (!flag_arrays[box_no](i,j,k).isCovered()) {
                            Real dot_cell = Real(0.);
                            for (int idim=0; idim < numcomp; ++idim) {
                                 dot_cell +=
                                   v1_arrays[box_no](i,j,k,idim)*v2_arrays[box_no](i,j,k,idim);
                            }
                            return { dot_cell };
                        }
                        else {
                            return { Real(0.)};
                        }
                    });
        // ParReduce is ONLY local operation;
        // Sum across MPI ranks
        amrex::ParallelDescriptor::ReduceRealSum(&dot_lev, 1);
        dot_all_lev += dot_lev;
#else
        dot_all_lev += MultiFab::Dot(*v1[ilev],0,*v2[ilev],0,numcomp,0);
#endif
    }
    return dot_all_lev;
}

void NonlinearDiffusionTensorOp::increment (VMF& lhs,
                                            VMF const& rhs, Real a)
{
    increment(GetVecOfPtrs(lhs), GetVecOfConstPtrs(rhs), a);
}

// Do NOT touch ghost and covered cells
void NonlinearDiffusionTensorOp::increment (VMFPtr const& lhs,
                                            VCMFPtr const& rhs, Real a)
{
    int numcomp = rhs[0]->nComp();
    int nlevels = rhs.size();
    for (int ilev=0; ilev < nlevels; ++ilev)
    {
#ifdef AMREX_USE_EB
        const auto& factory =
          dynamic_cast<EBFArrayBoxFactory const&>(rhs[ilev]->Factory());
        auto const& flags = factory.getMultiEBCellFlagFab();
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*rhs[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            auto const& flag_fab = flags[mfi];
            auto const& flag_arr = flag_fab.const_array();
            Array4<Real      > const& lhs_arr = lhs[ilev]->array(mfi);
            Array4<Real const> const& rhs_arr = rhs[ilev]->const_array(mfi);
            ParallelFor(bx, numcomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                if (!flag_arr(i,j,k).isCovered()) {
                    lhs_arr(i,j,k,n) += a*rhs_arr(i,j,k,n);
                }
            });
        }
#else
        MultiFab::Saxpy(*lhs[ilev],a,*rhs[ilev],0,0,numcomp,0);
#endif
    }
}

void NonlinearDiffusionTensorOp::linComb (VMF& lhs,
                                          Real a, VMF const& rhs_a,
                                          Real b, VMF const& rhs_b)
{
    linComb(GetVecOfPtrs(lhs), a, GetVecOfConstPtrs(rhs_a),
            b, GetVecOfConstPtrs(rhs_b));
}

// Do NOT touch ghost and covered cells
void NonlinearDiffusionTensorOp::linComb (VMFPtr const& lhs,
                                          Real a, VCMFPtr const& rhs_a,
                                          Real b, VCMFPtr const& rhs_b)
{
    int numcomp = rhs_a[0]->nComp();
    int nlevels = rhs_a.size();
    for (int ilev=0; ilev < nlevels; ++ilev)
    {
#ifdef AMREX_USE_EB
        const auto& factory =
          dynamic_cast<EBFArrayBoxFactory const&>(rhs_a[ilev]->Factory());
        auto const& flags = factory.getMultiEBCellFlagFab();
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*rhs_a[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            auto const& flag_fab = flags[mfi];
            auto const& flag_arr = flag_fab.const_array();
            Array4<Real      > const& lhs_arr   = lhs[ilev]->array(mfi);
            Array4<Real const> const& rhs_a_arr = rhs_a[ilev]->const_array(mfi);
            Array4<Real const> const& rhs_b_arr = rhs_b[ilev]->const_array(mfi);
            ParallelFor(bx, numcomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                if (!flag_arr(i,j,k).isCovered()) {
                    lhs_arr(i,j,k,n) = a*rhs_a_arr(i,j,k,n) + b*rhs_b_arr(i,j,k,n);
                }
            });
        }
#else
        MultiFab::LinComb(*lhs[ilev],a,*rhs_a[ilev],0,
                          b,*rhs_b[ilev],0,0,numcomp,0);
#endif
    }
}

Vector<MultiFab> NonlinearDiffusionTensorOp::makeVecRHS ()
{
    int nlevels = m_incflo->finestLevel()+1;
    Vector<MultiFab> rhs;
    rhs.resize(nlevels);
    for (int ilev = 0; ilev < nlevels; ++ilev) {
        rhs[ilev].define(m_newton_iter_vel[ilev]->boxArray(),
                         m_newton_iter_vel[ilev]->DistributionMap(),
                         AMREX_SPACEDIM,0, MFInfo(),
                         m_newton_iter_vel[ilev]->Factory());
    }
    return rhs;
}

Vector<MultiFab> NonlinearDiffusionTensorOp::makeVecLHS ()
{
    int nlevels = m_incflo->finestLevel()+1;
    Vector<MultiFab> lhs;
    lhs.resize(nlevels);
    for (int ilev = 0; ilev < nlevels; ++ilev) {
        lhs[ilev].define(m_newton_iter_vel[ilev]->boxArray(),
                         m_newton_iter_vel[ilev]->DistributionMap(),
                         AMREX_SPACEDIM,m_nghost_vel, MFInfo(),
                         m_newton_iter_vel[ilev]->Factory());
    }
    return lhs;
}

Real NonlinearDiffusionTensorOp::norm2 (VMF const& v)
{
    return norm2(GetVecOfConstPtrs(v));
}

Real NonlinearDiffusionTensorOp::norm2 (VCMFPtr const& v)
{
    return std::sqrt(dotProduct(v, v));
}

// Do NOT touch ghost and covered cells
void NonlinearDiffusionTensorOp::precond (VMF& lhs, VMF const& rhs)
{
    int nlevels = rhs.size();
    int numcomp = rhs[0].nComp();
    if (m_gmres_use_precond) {
        Vector<MultiFab> vel_mlmg(nlevels);
        for (int ilev=0; ilev < nlevels; ++ilev) {
            vel_mlmg[ilev].define(rhs[ilev].boxArray(),
                                  rhs[ilev].DistributionMap(),
                                  numcomp, 1, MFInfo(),
                                  rhs[ilev].Factory());
            // Setting physical boundaries to zero
            // assumes that all of them are Dirichlet
            vel_mlmg[ilev].setVal(Real(0.));
        }
        // Perform MLMG solve
        diffuse_velocity_mlmg(GetVecOfPtrs(vel_mlmg), GetVecOfPtrs(m_density),
                              GetVecOfConstPtrs(m_eta),
                              GetVecOfConstPtrs(rhs), m_dt);
        assign(lhs, vel_mlmg);
    }
    else{
        for (int ilev=0; ilev < nlevels; ++ilev)
        {
#ifdef AMREX_USE_EB
            const auto& factory =
              dynamic_cast<EBFArrayBoxFactory const&>(rhs[ilev].Factory());
            auto const& flags = factory.getMultiEBCellFlagFab();
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(rhs[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box const& bx = mfi.tilebox();
                auto const& flag_fab = flags[mfi];
                auto const& flag_arr = flag_fab.const_array();
                Array4<Real      > const& lhs_arr = lhs[ilev].array(mfi);
                Array4<Real const> const& rhs_arr = rhs[ilev].const_array(mfi);
                ParallelFor(bx, numcomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    if (!flag_arr(i,j,k).isCovered()) {
                        lhs_arr(i,j,k,n) = rhs_arr(i,j,k,n);
                    }
                });
            }
#else
            MultiFab::Copy(lhs[ilev],rhs[ilev],0,0,numcomp,0);
#endif
        }
    }
}

void NonlinearDiffusionTensorOp::scale (VMF& v, Real fac)
{
    scale(GetVecOfPtrs(v),fac);
}

// Do NOT touch ghost and covered cells
void NonlinearDiffusionTensorOp::scale (VMFPtr const& v, Real fac)
{
    int nlevels = v.size();
    int numcomp = v[0]->nComp();
    for (int ilev=0; ilev < nlevels; ++ilev)
    {
#ifdef AMREX_USE_EB
        const auto& factory =
          dynamic_cast<EBFArrayBoxFactory const&>(v[ilev]->Factory());
        auto const& flags = factory.getMultiEBCellFlagFab();
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*v[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            auto const& flag_fab = flags[mfi];
            auto const& flag_arr = flag_fab.const_array();
            Array4<Real      > const& v_arr = v[ilev]->array(mfi);
            ParallelFor(bx, numcomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                if (!flag_arr(i,j,k).isCovered()) {
                    v_arr(i,j,k,n) *= fac;
                }
            });
        }
#else
        v[ilev]->mult(fac, 0);
#endif
    }
}

// Do NOT touch ghost and covered cells
void NonlinearDiffusionTensorOp::setToZero (VMF& v)
{
    int nlevels = v.size();
    int numcomp = v[0].nComp();
    for (int ilev=0; ilev < nlevels; ++ilev)
    {
#ifdef AMREX_USE_EB
        const auto& factory =
          dynamic_cast<EBFArrayBoxFactory const&>(v[ilev].Factory());
        auto const& flags = factory.getMultiEBCellFlagFab();
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(v[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            auto const& flag_fab = flags[mfi];
            auto const& flag_arr = flag_fab.const_array();
            Array4<Real      > const& v_arr = v[ilev].array(mfi);
            ParallelFor(bx, numcomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                if (!flag_arr(i,j,k).isCovered()) {
                    v_arr(i,j,k,n) = Real(0.);
                }
            });
        }
#else
        v[ilev].setVal(Real(0.), 0);
#endif
    }
}

// Do NOT touch ghost and covered cells
void NonlinearDiffusionTensorOp::AddProduct (VMFPtr const& dst,
        VCMFPtr const& src1, int comp1, VCMFPtr const& src2, int comp2,
        int dstcomp, int numcomp)
{
    int nlevels = src1.size();
    for (int ilev=0; ilev < nlevels; ++ilev)
    {
#ifdef AMREX_USE_EB
        const auto& factory =
          dynamic_cast<EBFArrayBoxFactory const&>(src1[ilev]->Factory());
        auto const& flags = factory.getMultiEBCellFlagFab();
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*src1[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            auto const& flag_fab = flags[mfi];
            auto const& flag_arr = flag_fab.const_array();
            Array4<Real      > const& dst_arr  = dst[ilev]->array(mfi);
            Array4<Real const> const& src1_arr = src1[ilev]->const_array(mfi);
            Array4<Real const> const& src2_arr = src2[ilev]->const_array(mfi);
            ParallelFor(bx, numcomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                if (!flag_arr(i,j,k).isCovered()) {
                    dst_arr(i,j,k,dstcomp+n) += src1_arr(i,j,k,comp1+n)*src2_arr(i,j,k,comp2+n);
                }
            });
        }
#else
        MultiFab::AddProduct(*dst[ilev], *src1[ilev], comp1,
                             *src2[ilev], comp2, dstcomp, numcomp, 0);
#endif
    }
}


void NonlinearDiffusionTensorOp::compVelGrad (int amrlev,
            MultiFab & sol, // temporary velocity
            MLMG::Location loc,
            Array<MultiFab*, AMREX_SPACEDIM> const& gradVel // regular faces
#ifdef AMREX_USE_EB
            , MultiFab* gradVel_EB
#endif
            )
{
#ifdef AMREX_USE_EB
    if (m_eb_apply_op)
    {
        AMREX_D_TERM(Real lev_dx =m_incflo->Geom(amrlev).CellSize(0);,
                     Real lev_dy =m_incflo->Geom(amrlev).CellSize(1);,
                     Real lev_dz =m_incflo->Geom(amrlev).CellSize(2););
        AMREX_D_TERM(Real lev_mn_dx = lev_dx,+lev_dy,+lev_dz);
        lev_mn_dx /= Real(AMREX_SPACEDIM);

        m_eb_apply_op->compVelGrad(amrlev, gradVel, sol, loc);
        // Population of gradVel_EB
        gradVel_EB->setVal(Real(0.));
        // Create a null vel_eb; as it needs to be zero
        Array4<Real const> vel_eb_arr;
        const auto& factory =
          dynamic_cast<EBFArrayBoxFactory const&>(sol.Factory());
        auto const& flags        = factory.getMultiEBCellFlagFab();
        MultiCutFab const& bcent = factory.getBndryCent();
        MultiCutFab const& ccent = factory.getCentroid();
        MultiCutFab const& bnorm = factory.getBndryNormal();
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(sol,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            auto const& flag_fab = flags[mfi];
            auto typ = flag_fab.getType(bx);
            if (typ == FabType::singlevalued) {
                auto const& flag_arr = flag_fab.const_array();
                Array4<Real const> const& bcfab      = bcent.const_array(mfi);
                Array4<Real const> const& ccfab      = ccent.const_array(mfi);
                Array4<Real const> const& bnrmfab    = bnorm.const_array(mfi);
                Array4<Real const> const& vel_arr    = sol.const_array(mfi);
                Array4<Real      > const& gradVel_arr = gradVel_EB->array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (flag_arr(i,j,k).isSingleValued()) {
                        // Normal points outward
                        AMREX_D_TERM(Real anrmx = bnrmfab(i,j,k,0);,
                                     Real anrmy = bnrmfab(i,j,k,1);,
                                     Real anrmz = bnrmfab(i,j,k,2););

                        AMREX_D_TERM(
                          Real un = amrex::grad_eb_of_phi_on_centroids(i,j,k,0,
                                           vel_arr, vel_eb_arr, flag_arr,
                                           ccfab, bcfab,
                                           AMREX_D_DECL(anrmx, anrmy, anrmz),
                                           false);,
                          Real vn = amrex::grad_eb_of_phi_on_centroids(i,j,k,1,
                                           vel_arr, vel_eb_arr, flag_arr,
                                           ccfab, bcfab,
                                           AMREX_D_DECL(anrmx, anrmy, anrmz),
                                           false);,
                          Real wn = amrex::grad_eb_of_phi_on_centroids(i,j,k,2,
                                           vel_arr, vel_eb_arr, flag_arr,
                                           ccfab, bcfab,
                                           AMREX_D_DECL(anrmx, anrmy, anrmz),
                                           false););
                        // Note that the values returned by grad_eb_of_phi_on_centroids
                        // are NOT scaled by dx = dy = dz
                        AMREX_D_TERM(un /= lev_mn_dx;,
                                     vn /= lev_mn_dx;,
                                     wn /= lev_mn_dx;);
// The derivatives are put in the array with the following order:
// component: 0    ,  1    ,  2    ,  3    ,  4    , 5    ,  6    ,  7    ,  8
// in 2D:     dU/dx,  dV/dx,  dU/dy,  dV/dy
// in 3D:     dU/dx,  dV/dx,  dW/dx,  dU/dy,  dV/dy, dW/dy,  dU/dz,  dV/dz,  dW/dz
// THIS RECONSTRUCTION IS ONLY VALID FOR NO-SLIP, NO-PENETRATION, TRANSLATING RIGID EB SURFACE
#if (AMREX_SPACEDIM == 2)
                        gradVel_arr(i,j,k,0) = un*anrmx; gradVel_arr(i,j,k,1) = vn*anrmx;
                        gradVel_arr(i,j,k,2) = un*anrmy; gradVel_arr(i,j,k,3) = vn*anrmy;
#else
                        gradVel_arr(i,j,k,0) = un*anrmx; gradVel_arr(i,j,k,1) = vn*anrmx;
                        gradVel_arr(i,j,k,2) = wn*anrmx;
                        gradVel_arr(i,j,k,3) = un*anrmy; gradVel_arr(i,j,k,4) = vn*anrmy;
                        gradVel_arr(i,j,k,5) = wn*anrmy;
                        gradVel_arr(i,j,k,6) = un*anrmz; gradVel_arr(i,j,k,7) = vn*anrmz;
                        gradVel_arr(i,j,k,8) = wn*anrmz;
#endif
                    }
                });
            }
        }
    }
    else
#endif
    {
        m_reg_apply_op->compVelGrad(amrlev, gradVel, sol, loc);
    }
}
