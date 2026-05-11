#include <incflo.H>
#include <incflo_derive_K.H>
#include <cmath>

using namespace amrex;

namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real expterm (amrex::Real nu) noexcept
{
    return (nu < Real(1.e-9)) ? (Real(1.0)-Real(0.5)*nu+nu*nu*Real(1.0/6.0)-(nu*nu*nu)*Real(1./24.))
                        : -std::expm1(-nu)/nu;
}

struct NonNewtonianViscosity
{
    incflo::FluidModel fluid_model;
    amrex::Real mu, n_flow, tau_0, eta_0, papa_reg;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator() (amrex::Real sr) const noexcept {
        switch (fluid_model)
        {
        case incflo::FluidModel::powerlaw:
        {
            return mu * std::pow(sr,n_flow-Real(1.0));
        }
        case incflo::FluidModel::Bingham:
        {
            return mu + tau_0 * expterm(sr/papa_reg) / papa_reg;
        }
        case incflo::FluidModel::HerschelBulkley:
        {
            return (mu*std::pow(sr,n_flow)+tau_0)*expterm(sr/papa_reg)/papa_reg;
        }
        case incflo::FluidModel::deSouzaMendesDutra:
        {
            return (mu*std::pow(sr,n_flow)+tau_0)*expterm(sr*(eta_0/tau_0))*(eta_0/tau_0);
        }
        default:
        {
            return mu;
        }
        };
    }
};

struct GranularViscosity
{
    incflo::FluidModel fluid_model;
    amrex::Real mu_1, mu_2, I_0;
    amrex::Real mu_const, mu_A, mu_alpha;
    amrex::Real I_1_N, I_1_N_A_minus;
    amrex::Real I_1_N_alpha = amrex::Real(1.9);
    // The below are for granular temperature dependent mu_1
    amrex::Real I_c1,I_c2, I_c3, I_c4, I_e1, I_e2, I_e3, I_e4;

    void set_rauter_parameters (amrex::Real a_mu_1, amrex::Real a_mu_2,
                                amrex::Real a_I_0) {
        mu_1 = a_mu_1;
        mu_2 = a_mu_2;
        I_0 = a_I_0;
    }

    void set_granularpowerlaw_parameters (amrex::Real a_mu_const,
                                          amrex::Real a_mu_A,
                                          amrex::Real a_mu_alpha,
                                          amrex::Real a_I_1_N) {
        mu_const = a_mu_const;
        mu_A = a_mu_A;
        mu_alpha = a_mu_alpha;
        I_1_N = a_I_1_N;
        if (I_1_N > amrex::Real(0.0)) {
           amrex::Real I_1_N_mu = mu_const + mu_A * std::pow(I_1_N, mu_alpha);
           I_1_N_A_minus = I_1_N*std::exp(I_1_N_alpha/(I_1_N_mu*I_1_N_mu));
        }
    }

    void set_granularpowerlaw_temperature_parameters (
                                          amrex::Real a_I_c1, amrex::Real a_I_e1,
                                          amrex::Real a_I_c2, amrex::Real a_I_e2,
                                          amrex::Real a_I_c3, amrex::Real a_I_e3,
                                          amrex::Real a_I_c4, amrex::Real a_I_e4) {
        I_c1 = a_I_c1; I_e1 = a_I_e1;
        I_c2 = a_I_c2; I_e2 = a_I_e2;
        I_c3 = a_I_c3; I_e3 = a_I_e3;
        I_c4 = a_I_c4; I_e4 = a_I_e4;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator() (amrex::Real inrt_num) const noexcept {
        switch (fluid_model)
        {
        case incflo::FluidModel::Rauter:
        {
            return mu_1 + (mu_2-mu_1)*(inrt_num/(I_0 + inrt_num));
        }
        case incflo::FluidModel::GranularPowerlaw:
        {
            if (I_1_N > amrex::Real(0.0)) {
              if (inrt_num > I_1_N) {
                 return mu_const + mu_A * std::pow(inrt_num, mu_alpha);
              }
              else {
                 return std::sqrt(I_1_N_alpha/std::log(I_1_N_A_minus/(inrt_num+amrex::Real(1.0e-18))));
              }
            }
            else {
               return mu_const + mu_A * std::pow(inrt_num, mu_alpha);
            }
        }
        default:
        {
            return Real(0.);
        }
        };
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator() (amrex::Real inrt_num, amrex::Real temperature) const noexcept {
        if (fluid_model == incflo::FluidModel::GranularPowerlawTemperature) {
            // Functional form from Kim and Kamrin, Frontiers in Physics (2023)
            amrex::Real aa = I_c1*std::pow(inrt_num, I_e1) + I_c2*std::pow(inrt_num, I_e2)
                            + I_c3*std::pow(inrt_num, I_e3) + I_c4*std::pow(inrt_num, I_e4);
            aa /= std::pow(temperature+amrex::Real(1.0e-18), amrex::Real(1.0/6.0));
            return aa;
        }
        else {
            return amrex::Real(0.);
        }
    }
};

}

void incflo::compute_viscosity (Vector<MultiFab*> const& vel_eta,
                                Vector<MultiFab*> const& rho,
                                Vector<MultiFab*> const& vel,
                                Real time, int nghost)
{
#ifdef USE_AMREX_MPMD
    // Call to indicate this is from incflo::compute_viscosity
    if (ParallelDescriptor::MyProc() == 0) {
        Vector<int> last_call;
        last_call.push_back(0);
        MPI_Send(last_call.data(), last_call.size(), MPI_INT,m_mpmd_other_root,94,MPI_COMM_WORLD);
    }
#endif
    int nghost_cc_nd = m_nodal_vel_eta ? 0 : nghost;
    for (int lev = 0; lev <= finest_level; ++lev)
    {
            compute_viscosity_at_level(lev, vel_eta[lev], rho[lev],
                    vel[lev], geom[lev], time, nghost_cc_nd);
    }
}

void incflo::compute_viscosity_at_level (int lev,
                                         MultiFab* vel_eta,
                                         MultiFab* rho,
                                         MultiFab* vel,
                                         Geometry& lev_geom,
                                         Real time, int nghost)
{
    if (m_fluid_model == FluidModel::Newtonian)
    {
        vel_eta->setVal(m_mu, 0, 1, nghost);
    }
#ifdef USE_AMREX_MPMD
    else if (m_fluid_model == FluidModel::DataDrivenMPMD)
    {
        // Copier send of *vel_eta and Copier recv of *vel_eta
        if (m_nodal_vel_eta) {
            compute_nodal_strainrate_at_level(lev,vel_eta,vel,lev_geom,
                                              time,nghost);
        }
        else
        {
            compute_strainrate_at_level(lev, vel_eta, vel, lev_geom,
                                        time, nghost);
        }
        mpmd_copiers_send_lev(*vel_eta,0,1,lev);
        mpmd_copiers_recv_lev(*vel_eta,0,1,lev);
    }
#endif
    else
    {
        if (m_nodal_vel_eta) {
           compute_nodal_non_newtonian_viscosity(lev, vel_eta, rho, vel,
                                                 lev_geom, time, nghost,
                                                 0);
        }
        else
        {
           compute_cc_non_newtonian_viscosity(lev, vel_eta, vel, lev_geom,
                                              nghost, 0);
        }
    }
    // Clamp vel_eta if it is NOT-NEWTONIAN
    if (m_fluid_model != FluidModel::Newtonian)
    {
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*vel_eta,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.growntilebox(nghost);
            Array4<Real> const& eta_arr = vel_eta->array(mfi);
            const Real eta_min = m_eta_min;
            const Real eta_max = m_eta_max;
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                eta_arr(i,j,k) = amrex::Clamp(eta_arr(i,j,k),eta_min,eta_max);
            });
        }
    }

    if (m_two_fluid) {
       // Create a viscosity MultiFab for the second fluid
       MultiFab vel_eta_second(vel_eta->boxArray(),vel_eta->DistributionMap(),1,nghost);
       vel_eta_second.setVal(Real(0.0), 0, 1, nghost);
       // second fluid concentration MultiFab
       MultiFab conc_second(vel_eta->boxArray(),vel_eta->DistributionMap(),1,nghost);
       if (m_nodal_vel_eta) {
           compute_nodal_second_fluid_conc(&conc_second,rho,nghost);
       }
       else
       {
           compute_cc_second_fluid_conc(&conc_second,rho,nghost);
       }
       // Calculate second fluid viscosity
       compute_second_fluid_viscosity_at_level(lev, rho, vel, lev_geom,
                                               time, nghost, vel_eta_second,
                                               conc_second);
       // Calculate weighted viscosity
       if (!(m_mu-m_mu_second == Real(0.) and m_fluid_model == m_fluid_model_second)) {
         // Models that rely on hydrostatic pressure should only work on valid cells/nodes
         int nghost_mix = nghost;
         if (m_fluid_model_second == FluidModel::DataDrivenMPMD
             || m_fluid_model_second == FluidModel::Rauter
             || m_fluid_model_second == FluidModel::GranularPowerlaw
             || m_fluid_model_second == FluidModel::GranularPowerlawTemperature) {
             nghost_mix = 0;
         }
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
         for (MFIter mfi(conc_second,TilingIfNotGPU()); mfi.isValid(); ++mfi)
         {
             Box const& bx = mfi.growntilebox(nghost_mix);
             Array4<Real const> const& conc_second_arr = conc_second.array(mfi);
             Array4<Real const> const& eta_arr_second = vel_eta_second.const_array(mfi);
             Array4<Real> const& eta_arr = vel_eta->array(mfi);
             const Real min_conc_scnd = m_min_conc_second;
             const bool eta_harmonic = m_two_fluid_eta_harmonic;
             amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
             {
                if (conc_second_arr(i,j,k) > min_conc_scnd) {
                  // Using weighted harmonic mean for vel_eta
                  if (eta_harmonic) {
                      eta_arr(i,j,k) = ((Real(1.0)-conc_second_arr(i,j,k))/eta_arr(i,j,k))
                                    + (conc_second_arr(i,j,k)/eta_arr_second(i,j,k));
                      eta_arr(i,j,k) = Real(1.0)/eta_arr(i,j,k);
                  }
                  else {
                      eta_arr(i,j,k) = (Real(1.0)-conc_second_arr(i,j,k))*eta_arr(i,j,k) +
                                       conc_second_arr(i,j,k)*eta_arr_second(i,j,k);
                  }
                }
             });
         }
         if (nghost - nghost_mix) {
            vel_eta->FillBoundary(lev_geom.periodicity());
         }
       }
    }
#ifdef AMREX_USE_EB
    smooth_eb_cell_centered_coeff(lev, *vel_eta, lev_geom);
#endif
}

void incflo::compute_second_fluid_viscosity_at_level (int lev,
                                                      MultiFab* rho,
                                                      MultiFab* vel,
                                                      Geometry& lev_geom,
                                                      Real time, int nghost,
                                                      MultiFab& vel_eta_second,
                                                      MultiFab& conc_second
                                                     )
{
   if (m_fluid_model_second == FluidModel::Newtonian)
   {
       vel_eta_second.setVal(m_mu_second, 0, 1, nghost);
   }
   else if (m_fluid_model_second != FluidModel::DataDrivenMPMD
            && m_fluid_model_second != FluidModel::Rauter
            && m_fluid_model_second != FluidModel::GranularPowerlaw
            && m_fluid_model_second != FluidModel::GranularPowerlawTemperature)
   {
       // Non-Newtonian
       if (m_nodal_vel_eta) {
          compute_nodal_non_newtonian_viscosity(lev, &vel_eta_second, rho,
                                                vel, lev_geom, time,
                                                nghost,1);
       }
       else
       {
          compute_cc_non_newtonian_viscosity(lev, &vel_eta_second, vel,
                                             lev_geom, nghost, 1);
       }
   }
   else
   {
       // Hydrostatic pressure is only calculated in valid cells/nodes
       int nghost_hydrostatic = 0;
       // Create a strain-rate MultiFab
       MultiFab sr_mf(vel_eta_second.boxArray(),
                      vel_eta_second.DistributionMap(),1,nghost_hydrostatic);
       // MultiFab for hydrostatic pressure
       MultiFab p_static(vel_eta_second.boxArray(),
                         vel_eta_second.DistributionMap(),1,nghost_hydrostatic);
       if (m_nodal_vel_eta) {
          compute_nodal_strainrate_at_level(lev,&sr_mf,vel,lev_geom,time,nghost_hydrostatic);
          compute_nodal_hydrostatic_pressure_at_level(lev,&p_static,rho,
                                                   m_mu_p_surf_second,
                                                   lev_geom,nghost_hydrostatic);
       }
       else
       {
          compute_strainrate_at_level(lev,&sr_mf,vel,lev_geom,time,nghost_hydrostatic);
          compute_cc_hydrostatic_pressure_at_level(lev,&p_static,rho,
                                                   m_mu_p_surf_second,
                                                   lev_geom,nghost_hydrostatic);
       }
#ifdef USE_AMREX_MPMD
       if (m_fluid_model_second == FluidModel::DataDrivenMPMD) {
           MultiFab inertial_num_mpmd(vel_eta_second.boxArray(),
                                      vel_eta_second.DistributionMap(),
                                      2,nghost_hydrostatic);
           // Inertial Number = diameter*strainrate*sqrt(rho_grain/p)
           // NOTE: Strain-rate calculated is TWO TIMES the actual value
           // The second component will carry concentration
           MultiFab inertial_num(inertial_num_mpmd,amrex::make_alias,0,1);
           compute_inertial_num_at_level(lev,&inertial_num,
                                         &sr_mf,&p_static,m_mu_p_eps_second,
                                         m_ro_grain_second,m_diam_second,
                                         nghost_hydrostatic);
           // Copy concentration
           MultiFab::Copy(inertial_num_mpmd,conc_second,0,1,1,nghost_hydrostatic);
           // Copier send inertial_num_mpmd
           mpmd_copiers_send_lev(inertial_num_mpmd,0,2,lev);
           // NOTE: Actual received quantity is stress ratio
           mpmd_copiers_recv_lev(vel_eta_second,0,1,lev);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
           for (MFIter mfi(sr_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
           {
               Box const& bx = mfi.growntilebox(nghost_hydrostatic);
               Array4<Real const> const& sr_arr = sr_mf.const_array(mfi);
               Array4<Real const> const& p_static_arr = p_static.const_array(mfi);
               Array4<Real> const& vel_eta_snd_arr = vel_eta_second.array(mfi);
               const Real eps = m_mu_sr_eps_second;
               // Note: sr_mf contains TWO TIMES strain rate
               amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
               {
                    // Regularized strain rate
                    Real sr_reg = Real(0.5)*sr_arr(i,j,k) + eps;
                    vel_eta_snd_arr(i,j,k) *= p_static_arr(i,j,k);
                    vel_eta_snd_arr(i,j,k) /= (Real(2.0)*sr_reg);
               });
           }

       } else
#endif
       if (m_fluid_model_second == FluidModel::Rauter ||
           m_fluid_model_second == FluidModel::GranularPowerlaw ||
           m_fluid_model_second == FluidModel::GranularPowerlawTemperature) {
          GranularViscosity granvisc;
          granvisc.fluid_model = m_fluid_model_second;
          if (m_fluid_model_second == FluidModel::Rauter)
          {
             granvisc.set_rauter_parameters(m_mu_1_second, m_mu_2_second,
                                            m_I_0_second);
          }
          else if (m_fluid_model_second == FluidModel::GranularPowerlaw)
          {
             granvisc.set_granularpowerlaw_parameters(m_mu_powerlaw[0][0],
                                                      m_mu_powerlaw[0][1],
                                                      m_mu_powerlaw[0][2],
                                                      m_I_1_N_powerlaw);
          }
          else
          {
             granvisc.set_granularpowerlaw_temperature_parameters(
                m_mu_powerlaw_temperature[0][0], m_mu_powerlaw_temperature[0][1],
                m_mu_powerlaw_temperature[0][2], m_mu_powerlaw_temperature[0][3],
                m_mu_powerlaw_temperature[0][4], m_mu_powerlaw_temperature[0][5],
                m_mu_powerlaw_temperature[0][6], m_mu_powerlaw_temperature[0][7]);
          }
          // Inertial Number = diameter*strainrate*sqrt(rho_grain/p)
          // NOTE: Strain-rate calculated is TWO TIMES the actual value
          // The second component will carry concentration
          MultiFab inertial_num(vel_eta_second.boxArray(),
                                vel_eta_second.DistributionMap(),
                                1,nghost_hydrostatic);
          compute_inertial_num_at_level(lev,&inertial_num,
                                        &sr_mf,&p_static,m_mu_p_eps_second,
                                        m_ro_grain_second,m_diam_second,
                                        nghost_hydrostatic);
          if (m_fluid_model_second == FluidModel::GranularPowerlawTemperature) {
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
              for (MFIter mfi(sr_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
              {
                  Box const& bx = mfi.growntilebox(nghost_hydrostatic);
                  Array4<Real const> const& sr_arr = sr_mf.const_array(mfi);
                  Array4<Real const> const& p_static_arr = p_static.const_array(mfi);
                  Array4<Real const> const& inrt_num_arr = inertial_num.const_array(mfi);
                  Array4<Real const> const& temperature_arr =
                      m_leveldata[lev]->temperature.const_array(mfi);
                  Array4<Real> const& vel_eta_snd_arr = vel_eta_second.array(mfi);
                  const Real eps = m_mu_sr_eps_second;
                  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                  {
                       vel_eta_snd_arr(i,j,k) =
                           granvisc(inrt_num_arr(i,j,k,0), temperature_arr(i,j,k));
                       Real sr_reg = Real(0.5)*sr_arr(i,j,k) + eps;
                       vel_eta_snd_arr(i,j,k) *= p_static_arr(i,j,k);
                       vel_eta_snd_arr(i,j,k) /= (Real(2.0)*sr_reg);
                  });
              }
          } else {
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
              for (MFIter mfi(sr_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
              {
                  Box const& bx = mfi.growntilebox(nghost_hydrostatic);
                  Array4<Real const> const& sr_arr = sr_mf.const_array(mfi);
                  Array4<Real const> const& p_static_arr = p_static.const_array(mfi);
                  Array4<Real const> const& inrt_num_arr = inertial_num.const_array(mfi);
                  Array4<Real> const& vel_eta_snd_arr = vel_eta_second.array(mfi);
                  const Real eps = m_mu_sr_eps_second;
                  // Note: sr_mf contains TWO TIMES strain rate
                  // Note: Inertial number in Rauter 2021 (Eq. 2.29)
                  // has an extra factor of 2
                  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                  {
                       vel_eta_snd_arr(i,j,k) = granvisc(inrt_num_arr(i,j,k,0));
                       // The above value is stress ratio
                       // Regularized strain rate
                       Real sr_reg = Real(0.5)*sr_arr(i,j,k) + eps;
                       vel_eta_snd_arr(i,j,k) *= p_static_arr(i,j,k);
                       vel_eta_snd_arr(i,j,k) /= (Real(2.0)*sr_reg);
                  });
              }
          }
      }
      // As only valid cells/nodes were populated
      if (nghost) {
         vel_eta_second.FillBoundary(lev_geom.periodicity());
      }
   }

   // Clamp vel_eta_second if it is NOT-NEWTONIAN
   if (m_fluid_model_second != FluidModel::Newtonian)
   {
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(vel_eta_second,TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
          Box const& bx = mfi.growntilebox(nghost);
          Array4<Real> const& eta_arr = vel_eta_second.array(mfi);
          const Real eta_min_scnd = m_eta_min_second;
          const Real eta_max_scnd = m_eta_max_second;
          amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              eta_arr(i,j,k) = amrex::Clamp(eta_arr(i,j,k),eta_min_scnd,
                                            eta_max_scnd);
          });
      }
    }
}

// This is cell-centered non-newtonian viscosity calculation
// EBs can be handled
void incflo::compute_cc_non_newtonian_viscosity (int lev,
                                         MultiFab* vel_eta,
                                         MultiFab* vel,
                                         Geometry& lev_geom,
                                         int nghost, int comp_id)
{
    NonNewtonianViscosity non_newtonian_viscosity;
    if (comp_id == 0) {
       non_newtonian_viscosity.fluid_model = m_fluid_model;
       non_newtonian_viscosity.mu = m_mu;
       non_newtonian_viscosity.n_flow = m_n_0;
       non_newtonian_viscosity.tau_0 = m_tau_0;
       non_newtonian_viscosity.eta_0 = m_eta_0;
       non_newtonian_viscosity.papa_reg = m_papa_reg;
    }
    else {
       non_newtonian_viscosity.fluid_model = m_fluid_model_second;
       non_newtonian_viscosity.mu = m_mu_second;
       non_newtonian_viscosity.n_flow = m_n_0_second;
       non_newtonian_viscosity.tau_0 = m_tau_0_second;
       non_newtonian_viscosity.eta_0 = m_eta_0_second;
       non_newtonian_viscosity.papa_reg = m_papa_reg_second;
    }
#ifdef AMREX_USE_EB
    auto const& fact = EBFactory(lev);
    auto const& flags = fact.getMultiEBCellFlagFab();
    MultiCutFab const& bcent = fact.getBndryCent();
    MultiCutFab const& ccent = fact.getCentroid();
    MultiCutFab const& bnorm = fact.getBndryNormal();
#endif

    Real idx = Real(1.0) / lev_geom.CellSize(0);
    Real idy = Real(1.0) / lev_geom.CellSize(1);
#if (AMREX_SPACEDIM == 3)
    Real idz = Real(1.0) / lev_geom.CellSize(2);
#endif
    const Dim3 dlo = amrex::lbound(lev_geom.Domain());
    const Dim3 dhi = amrex::ubound(lev_geom.Domain());
    GpuArray<bool, AMREX_SPACEDIM> is_periodic;
    AMREX_D_TERM(is_periodic[0] = lev_geom.isPeriodic(0);,
                 is_periodic[1] = lev_geom.isPeriodic(1);,
                 is_periodic[2] = lev_geom.isPeriodic(2););
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*vel_eta,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& bx = mfi.growntilebox(nghost);
        Array4<Real> const& eta_arr = vel_eta->array(mfi);
        Array4<Real const> const& vel_arr = vel->const_array(mfi);
#ifdef AMREX_USE_EB
        auto const& flag_fab = flags[mfi];
        auto typ = flag_fab.getType(bx);
        Array4<Real const> const& bcfab      = bcent.const_array(mfi);
        Array4<Real const> const& ccfab      = ccent.const_array(mfi);
        Array4<Real const> const& bnrmfab    = bnorm.const_array(mfi);
        if (typ == FabType::covered)
        {
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                eta_arr(i,j,k) = Real(0.0);
            });
        }
        else if (typ == FabType::singlevalued)
        {
            auto const& flag_arr = flag_fab.const_array();
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real sr = incflo_strainrate_eb(i,j,k,AMREX_D_DECL(idx,idy,idz),
                                               vel_arr, flag_arr, dlo, dhi,
                                               is_periodic, true, ccfab, bcfab,
                                               AMREX_D_DECL(bnrmfab(i,j,k,0),
                                                            bnrmfab(i,j,k,1),
                                                            bnrmfab(i,j,k,2)));
                eta_arr(i,j,k) = non_newtonian_viscosity(sr);
            });
        }
        else
#endif
        {
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real sr = incflo_strainrate(i,j,k,AMREX_D_DECL(idx,idy,idz),
                                            vel_arr, dlo, dhi, is_periodic);
                eta_arr(i,j,k) = non_newtonian_viscosity(sr);
            });
        }
    }
}

// This is nodal non-newtonian viscosity
// EBs CANNOT be handled
void incflo::compute_nodal_non_newtonian_viscosity (int lev,
                                                    MultiFab* vel_eta,
                                                    MultiFab* rho,
                                                    MultiFab* vel,
                                                    Geometry& lev_geom,
                                                    Real time, int nghost,
                                                    int comp_id)
{

    NonNewtonianViscosity non_newtonian_viscosity;
    if (comp_id == 0) {
       non_newtonian_viscosity.fluid_model = m_fluid_model;
       non_newtonian_viscosity.mu = m_mu;
       non_newtonian_viscosity.n_flow = m_n_0;
       non_newtonian_viscosity.tau_0 = m_tau_0;
       non_newtonian_viscosity.eta_0 = m_eta_0;
       non_newtonian_viscosity.papa_reg = m_papa_reg;
    }
    else {
       non_newtonian_viscosity.fluid_model = m_fluid_model_second;
       non_newtonian_viscosity.mu = m_mu_second;
       non_newtonian_viscosity.n_flow = m_n_0_second;
       non_newtonian_viscosity.tau_0 = m_tau_0_second;
       non_newtonian_viscosity.eta_0 = m_eta_0_second;
       non_newtonian_viscosity.papa_reg = m_papa_reg_second;
    }

    compute_nodal_strainrate_at_level(lev,vel_eta,vel,lev_geom,time,nghost);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*vel_eta,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& bx = mfi.growntilebox(nghost);
        Array4<Real> const& eta_arr = vel_eta->array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eta_arr(i,j,k) = non_newtonian_viscosity(eta_arr(i,j,k));
        });
    }
}

// This function in-place converts inertial number to mu(I)
void incflo::compute_mu_I_at_level (int lev, MultiFab* inertial_num,
                                    MultiFab const* temperature,
                                    int nghost)
{
  GranularViscosity granvisc;
  granvisc.fluid_model = m_fluid_model_second;
  if (m_fluid_model_second == FluidModel::Rauter)
  {
     granvisc.set_rauter_parameters(m_mu_1_second, m_mu_2_second,
                                    m_I_0_second);
  }
  else if (m_fluid_model_second == FluidModel::GranularPowerlaw)
  {
     granvisc.set_granularpowerlaw_parameters(m_mu_powerlaw[0][0],
                                              m_mu_powerlaw[0][1],
                                              m_mu_powerlaw[0][2],
                                              m_I_1_N_powerlaw);
  }
  else
  {
     granvisc.set_granularpowerlaw_temperature_parameters(
        m_mu_powerlaw_temperature[0][0], m_mu_powerlaw_temperature[0][1],
        m_mu_powerlaw_temperature[0][2], m_mu_powerlaw_temperature[0][3],
        m_mu_powerlaw_temperature[0][4], m_mu_powerlaw_temperature[0][5],
        m_mu_powerlaw_temperature[0][6], m_mu_powerlaw_temperature[0][7]);
  }
  if (m_fluid_model_second == FluidModel::GranularPowerlawTemperature) {
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(*inertial_num,TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
          Box const& bx = mfi.growntilebox(nghost);
          Array4<Real> const& inrt_num_arr = inertial_num->array(mfi);
          Array4<Real const> const& temperature_arr = temperature->const_array(mfi);
          amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
               inrt_num_arr(i,j,k,0) =
                   granvisc(inrt_num_arr(i,j,k,0), temperature_arr(i,j,k));
          });
      }
  } else {
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(*inertial_num,TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
          Box const& bx = mfi.growntilebox(nghost);
          Array4<Real> const& inrt_num_arr = inertial_num->array(mfi);
          amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
               inrt_num_arr(i,j,k,0) = granvisc(inrt_num_arr(i,j,k,0));
          });
      }
  }
}

#ifdef AMREX_USE_EB
void incflo::smooth_eb_cell_centered_coeff (int lev,
                                            MultiFab& mf,
                                            Geometry& lev_geom)
{
    if (!m_eb_smooth_cutcell_viscosity || EBFactory(lev).isAllRegular()) {
        return;
    }

    if (mf.nGrow() > 0) {
        mf.FillBoundary(lev_geom.periodicity());
    }

    MultiFab mf_smooth(mf.boxArray(), mf.DistributionMap(), mf.nComp(), mf.nGrow(),
                       MFInfo(), mf.Factory());
    MultiFab::Copy(mf_smooth, mf, 0, 0, mf.nComp(), mf.nGrow());

    const auto& fact = EBFactory(lev);
    auto const& flags = fact.getMultiEBCellFlagFab();
    const Dim3 dlo = amrex::lbound(lev_geom.Domain());
    const Dim3 dhi = amrex::ubound(lev_geom.Domain());
    GpuArray<bool, AMREX_SPACEDIM> is_periodic;
    AMREX_D_TERM(is_periodic[0] = lev_geom.isPeriodic(0);,
                 is_periodic[1] = lev_geom.isPeriodic(1);,
                 is_periodic[2] = lev_geom.isPeriodic(2););
    const Real blend = m_eb_smooth_cutcell_viscosity_blend;

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& bx = mfi.tilebox();
        auto const& flag_fab = flags[mfi];
        auto typ = flag_fab.getType(bx);
        if (typ != FabType::singlevalued) {
            continue;
        }

        auto const& flag_arr = flag_fab.const_array();
        const Dim3 flo = amrex::lbound(flag_fab.box());
        const Dim3 fhi = amrex::ubound(flag_fab.box());
        Array4<Real const> const& src_arr = mf.const_array(mfi);
        Array4<Real> const& dst_arr = mf_smooth.array(mfi);
        const int ncomp = mf.nComp();
        const Real eta_min = m_eta_min;
        const Real eta_max = m_eta_max;

        ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            if (!flag_arr(i,j,k).isSingleValued()) {
                return;
            }

            Real regular_sum = Real(0.0);
            int regular_count = 0;
            const Real src_val = src_arr(i,j,k,n);
            const bool src_ok = std::isfinite(src_val);

#if (AMREX_SPACEDIM == 2)
            for (int jj = -1; jj <= 1; ++jj) {
                for (int ii = -1; ii <= 1; ++ii) {
                    if (ii == 0 && jj == 0) { continue; }
                    int ni = i + ii;
                    int nj = j + jj;
                    int nk = k;
                    if ((!is_periodic[0] && (ni < dlo.x || ni > dhi.x)) ||
                        (!is_periodic[1] && (nj < dlo.y || nj > dhi.y)) ||
                        ni < flo.x || ni > fhi.x ||
                        nj < flo.y || nj > fhi.y) {
                        continue;
                    }
                    auto nflag = flag_arr(ni,nj,nk);
                    if (nflag.isCovered()) {
                        continue;
                    }
                    Real nval = src_arr(ni,nj,nk,n);
                    if (nflag.isRegular() &&
                        std::isfinite(nval) &&
                        nval >= eta_min && nval <= eta_max) {
                        regular_sum += nval;
                        ++regular_count;
                    }
                }
            }
#else
            for (int kk = -1; kk <= 1; ++kk) {
                for (int jj = -1; jj <= 1; ++jj) {
                    for (int ii = -1; ii <= 1; ++ii) {
                        if (ii == 0 && jj == 0 && kk == 0) { continue; }
                        int ni = i + ii;
                        int nj = j + jj;
                        int nk = k + kk;
                        if ((!is_periodic[0] && (ni < dlo.x || ni > dhi.x)) ||
                            (!is_periodic[1] && (nj < dlo.y || nj > dhi.y)) ||
                            (!is_periodic[2] && (nk < dlo.z || nk > dhi.z)) ||
                            ni < flo.x || ni > fhi.x ||
                            nj < flo.y || nj > fhi.y ||
                            nk < flo.z || nk > fhi.z) {
                            continue;
                        }
                        auto nflag = flag_arr(ni,nj,nk);
                        if (nflag.isCovered()) {
                            continue;
                        }
                        Real nval = src_arr(ni,nj,nk,n);
                        if (nflag.isRegular() &&
                            std::isfinite(nval) &&
                            nval >= eta_min && nval <= eta_max) {
                            regular_sum += nval;
                            ++regular_count;
                        }
                    }
                }
            }
#endif

            if (src_ok && regular_count > 0) {
                Real avg = regular_sum / Real(regular_count);
                Real blended = (Real(1.0)-blend)*src_val + blend*avg;
                dst_arr(i,j,k,n) = amrex::Clamp(blended, eta_min, eta_max);
            }
        });
    }

    MultiFab::Copy(mf, mf_smooth, 0, 0, mf.nComp(), mf.nGrow());
    if (mf.nGrow() > 0) {
        mf.FillBoundary(lev_geom.periodicity());
    }
}
#endif

void
incflo::compute_granular_high_order_divtau_on_level (int ilev,
            MultiFab & a_divtau,
            Array<const MultiFab*, AMREX_SPACEDIM> const& gradVel,
#ifdef AMREX_USE_EB
            const MultiFab* gradVel_EB,
#endif
            MultiFab& scndOrderCoeff, bool already_on_centroids)
{
        // Face-averaged scndOrderCoeff; This handles boundary faces
    Array<MultiFab,AMREX_SPACEDIM> fc_scndOrdr =
                 average_velocity_eta_to_faces(ilev,scndOrderCoeff);
    a_divtau.setVal(Real(0.));
    // Fluxes for faces that align with (x,y,z)
    Array<MultiFab, AMREX_SPACEDIM> fluxes;
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
        fluxes[idim].define(gradVel[idim]->boxArray(),
                            gradVel[idim]->DistributionMap(),
                            AMREX_SPACEDIM, 0, MFInfo(),
                            gradVel[idim]->Factory());
        fluxes[idim].setVal(Real(0.));
    }
    // Get fluxes
    compute_granular_high_order_fluxes_on_level(amrex::GetArrOfPtrs(fluxes),
                    gradVel, amrex::GetArrOfConstPtrs(fc_scndOrdr));

    auto & lev_geom = Geom(ilev);
    // Get divergence of fluxes
#ifdef AMREX_USE_EB
    if (!EBFactory(0).isAllRegular())
    {
        MultiFab divtau_regfaces(a_divtau.boxArray(), a_divtau.DistributionMap(),
                                 AMREX_SPACEDIM, a_divtau.nGrow(), MFInfo(),
                                 a_divtau.Factory());
        divtau_regfaces.setVal(Real(0.));
        // This sums over faces that align with (x,y,z)
        amrex::EB_computeDivergence(divtau_regfaces,
                    amrex::GetArrOfConstPtrs(fluxes),
                    lev_geom, already_on_centroids);
        // Include EB-Flux into divergence
        MultiFab flux_eb(gradVel_EB->boxArray(),gradVel_EB->DistributionMap(),
                         AMREX_SPACEDIM, 0);
        MultiFab divtau_ebfaces(a_divtau.boxArray(), a_divtau.DistributionMap(),
                                AMREX_SPACEDIM, a_divtau.nGrow(), MFInfo(),
                                a_divtau.Factory());
        divtau_ebfaces.setVal(Real(0.));
        compute_granular_high_order_fluxes_on_level(&flux_eb, gradVel_EB,
                                                    &scndOrderCoeff);
        AMREX_D_TERM(Real dx = lev_geom.CellSize(0);,
                     Real dy = lev_geom.CellSize(1);,
                     Real dz = lev_geom.CellSize(2););
        const auto& factory =
          dynamic_cast<EBFArrayBoxFactory const&>(a_divtau.Factory());
        auto const& flags        = factory.getMultiEBCellFlagFab();
        MultiFab    const& vfrac = factory.getVolFrac();
        MultiCutFab const& bnorm = factory.getBndryNormal();
        MultiCutFab const& barea = factory.getBndryArea();
        const Real ho_vfrac_threshold = m_eb_ho_vfrac_threshold;
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(a_divtau,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            auto const& flag_fab = flags[mfi];
            auto typ = flag_fab.getType(bx);
            if (typ == FabType::singlevalued) {
                auto const& flag_arr = flag_fab.const_array();
                Array4<Real const> const& bnrm_arr   = bnorm.const_array(mfi);
                Array4<Real const> const& barea_arr  = barea.const_array(mfi);
                Array4<Real const> const& vfrac_arr  = vfrac.const_array(mfi);
                Array4<Real const> const& fluxeb_arr = flux_eb.const_array(mfi);
                Array4<Real      > const& divtau_reg_arr = divtau_regfaces.array(mfi);
                Array4<Real      > const& divtau_eb_arr  = divtau_ebfaces.array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (flag_arr(i,j,k).isSingleValued()) {
                        if (vfrac_arr(i,j,k) <= ho_vfrac_threshold) {
                            AMREX_D_TERM(divtau_reg_arr(i,j,k,0) = Real(0.0);,
                                         divtau_reg_arr(i,j,k,1) = Real(0.0);,
                                         divtau_reg_arr(i,j,k,2) = Real(0.0););
                            return;
                        }
                        // Normal points outward
                        AMREX_D_TERM(Real anrmx = bnrm_arr(i,j,k,0);,
                                     Real anrmy = bnrm_arr(i,j,k,1);,
                                     Real anrmz = bnrm_arr(i,j,k,2););
                        Real eb_farea = anrmx*anrmx*dy*dy + anrmy*anrmy*dx*dx;
                        Real inv_eb_vol = Real(1.0)/(dx*dy*vfrac_arr(i,j,k));
#if (AMREX_SPACEDIM == 3)
                        eb_farea *= dz*dz;
                        eb_farea += anrmz*anrmz*dx*dx*dy*dy;
                        inv_eb_vol /= dz;
#endif
                        eb_farea = std::sqrt(eb_farea)*barea_arr(i,j,k);
                        AMREX_D_TERM(
                          divtau_eb_arr(i,j,k,0) += inv_eb_vol*eb_farea*fluxeb_arr(i,j,k,0);,
                          divtau_eb_arr(i,j,k,1) += inv_eb_vol*eb_farea*fluxeb_arr(i,j,k,1);,
                          divtau_eb_arr(i,j,k,2) += inv_eb_vol*eb_farea*fluxeb_arr(i,j,k,2););
                    }
                });
            }
        }
        MultiFab::Copy(a_divtau, divtau_regfaces, 0, 0, AMREX_SPACEDIM, a_divtau.nGrow());
        MultiFab::Add(a_divtau, divtau_ebfaces, 0, 0, AMREX_SPACEDIM, a_divtau.nGrow());
    }
    else
#endif
    {
        amrex::computeDivergence(a_divtau, amrex::GetArrOfConstPtrs(fluxes), lev_geom);
    }
}

// This function is to consider high-order terms in Granular Rheology
// Need to think of a better way to write this function
// Different elements of fluxes (MFIter loops) can be performed asynchronously
void
incflo::compute_granular_high_order_fluxes_on_level (
                     const Array<      MultiFab*,AMREX_SPACEDIM>& fluxes,
                     const Array<const MultiFab*,AMREX_SPACEDIM>& gradVel,
                     const Array<const MultiFab*,AMREX_SPACEDIM>& scndOrderCoeff)
{
    // X-flux
    int idim = 0;
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
   for (MFIter mfi(*fluxes[idim],TilingIfNotGPU()); mfi.isValid(); ++mfi)
   {
       Box const& bx = mfi.tilebox();
       Array4<Real const> const& gradVel_arr = gradVel[idim]->const_array(mfi);
       Array4<Real const> const& scndCoeff_arr = scndOrderCoeff[idim]->const_array(mfi);
       Array4<Real      > const& flux_arr = fluxes[idim]->array(mfi);
       amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
          Real ux = gradVel_arr(i,j,k,0);
          Real vx = gradVel_arr(i,j,k,1);
#if (AMREX_SPACEDIM == 2)
          Real uy = gradVel_arr(i,j,k,2);
          Real vy = gradVel_arr(i,j,k,3);
#else
          Real wx = gradVel_arr(i,j,k,2);
          Real uy = gradVel_arr(i,j,k,3);
          Real vy = gradVel_arr(i,j,k,4);
          Real wy = gradVel_arr(i,j,k,5);
          Real uz = gradVel_arr(i,j,k,6);
          Real vz = gradVel_arr(i,j,k,7);
          Real wz = gradVel_arr(i,j,k,8);
#endif
          // A is a symmetric tensor
#if (AMREX_SPACEDIM == 2)
          Real A_11 =  Real(0.5)*(ux*ux-vy*vy);

          Real A_12 =  Real(0.5)*(uy+vx)*(ux+vy);
#else
          Real A_11 =   (uy+vx)*(uy+vx)/Real(12.0)
                        + (uz+wx)*(uz+wx)/Real(12.0)
                        - (vz+wy)*(vz+wy)/Real(6.0)
                        + Real(2.0)*ux*ux/Real(3.0)
                        - vy*vy/Real(3.0)
                        - wz*wz/Real(3.0);
          Real A_12 =   Real(0.5)*(uy+vx)*(ux+vy)
                        + Real(0.25)*(uz+wx)*(vz+wy);
          Real A_13 =   Real(0.25)*(uy+vx)*(vz+wy)
                        + Real(0.5)*(uz+wx)*(ux+wz);
          // Multiplying the rheological coefficient
          A_13 *= scndCoeff_arr(i,j,k,0);
#endif
          // Multiplying the rheological coefficient
          A_11 *= scndCoeff_arr(i,j,k,0);
          A_12 *= scndCoeff_arr(i,j,k,0);
          // THIS IS A COMPRESSIVE FORCE SO THERE NEEDS TO
          // BE A MINUS IN FRONT OF THE TERMS
          A_11 *= Real(-1.0); A_12 *= Real(-1.0);
          flux_arr(i,j,k,0) = A_11; flux_arr(i,j,k,1) = A_12;
#if (AMREX_SPACEDIM == 3)
          A_13 *= Real(-1.0);
          flux_arr(i,j,k,2) = A_13;
#endif
       });
   }

    // Y-flux
    idim = 1;
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
   for (MFIter mfi(*fluxes[idim],TilingIfNotGPU()); mfi.isValid(); ++mfi)
   {
       Box const& bx = mfi.tilebox();
       Array4<Real const> const& gradVel_arr = gradVel[idim]->const_array(mfi);
       Array4<Real const> const& scndCoeff_arr = scndOrderCoeff[idim]->const_array(mfi);
       Array4<Real      > const& flux_arr = fluxes[idim]->array(mfi);

       amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
          Real ux = gradVel_arr(i,j,k,0);
          Real vx = gradVel_arr(i,j,k,1);
#if (AMREX_SPACEDIM == 2)
          Real uy = gradVel_arr(i,j,k,2);
          Real vy = gradVel_arr(i,j,k,3);
#else
          Real wx = gradVel_arr(i,j,k,2);
          Real uy = gradVel_arr(i,j,k,3);
          Real vy = gradVel_arr(i,j,k,4);
          Real wy = gradVel_arr(i,j,k,5);
          Real uz = gradVel_arr(i,j,k,6);
          Real vz = gradVel_arr(i,j,k,7);
          Real wz = gradVel_arr(i,j,k,8);
#endif
          // A is a symmetric tensor
#if (AMREX_SPACEDIM == 2)
          Real A_12 =  Real(0.5)*(uy+vx)*(ux+vy);
          Real A_22 =  Real(0.5)*(vy*vy-ux*ux);
#else
          Real A_12 =   Real(0.5)*(uy+vx)*(ux+vy)
                        + Real(0.25)*(uz+wx)*(vz+wy);

          Real A_22 =   (uy+vx)*(uy+vx)/Real(12.0)
                        - (uz+wx)*(uz+wx)/Real(6.0)
                        + (vz+wy)*(vz+wy)/Real(12.0)
                        - ux*ux/Real(3.0)
                        + Real(2.0)*vy*vy/Real(3.0)
                        - wz*wz/Real(3.0);

          Real A_23 =   Real(0.25)*(uy+vx)*(uz+wx)
                        + Real(0.5)*(vz+wy)*(vy+wz);
          // Multiplying the rheological coefficient
          A_23 *= scndCoeff_arr(i,j,k,0);
#endif
          // Multiplying the rheological coefficient
          A_12 *= scndCoeff_arr(i,j,k,0);
          A_22 *= scndCoeff_arr(i,j,k,0);
          // THIS IS A COMPRESSIVE FORCE SO THERE NEEDS TO
          // BE A MINUS IN FRONT OF THE TERMS
          A_12 *= Real(-1.0);  A_22 *= Real(-1.0);
          flux_arr(i,j,k,0) = A_12; flux_arr(i,j,k,1) = A_22;
#if (AMREX_SPACEDIM == 3)
          A_23 *= Real(-1.0); flux_arr(i,j,k,2) = A_23;
#endif
       });
   }

#if (AMREX_SPACEDIM == 3)
    // Z-flux
    idim = 2;
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
   for (MFIter mfi(*fluxes[idim],TilingIfNotGPU()); mfi.isValid(); ++mfi)
   {
       Box const& bx = mfi.tilebox();
       Array4<Real const> const& gradVel_arr = gradVel[idim]->const_array(mfi);
       Array4<Real const> const& scndCoeff_arr = scndOrderCoeff[idim]->const_array(mfi);
       Array4<Real      > const& flux_arr = fluxes[idim]->array(mfi);

       amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
          Real ux = gradVel_arr(i,j,k,0);
          Real vx = gradVel_arr(i,j,k,1);
          Real wx = gradVel_arr(i,j,k,2);
          Real uy = gradVel_arr(i,j,k,3);
          Real vy = gradVel_arr(i,j,k,4);
          Real wy = gradVel_arr(i,j,k,5);
          Real uz = gradVel_arr(i,j,k,6);
          Real vz = gradVel_arr(i,j,k,7);
          Real wz = gradVel_arr(i,j,k,8);
          // A is a symmetric tensor
          Real A_13 =   Real(0.25)*(uy+vx)*(vz+wy)
                        + Real(0.5)*(uz+wx)*(ux+wz);

          Real A_23 =   Real(0.25)*(uy+vx)*(uz+wx)
                        + Real(0.5)*(vz+wy)*(vy+wz);

          Real A_33 =   -(uy+vx)*(uy+vx)/Real(6.0)
                        + (uz+wx)*(uz+wx)/Real(12.0)
                        + (vz+wy)*(vz+wy)/Real(12.0)
                        - ux*ux/Real(3.0)
                        - vy*vy/Real(3.0)
                        + Real(2.0)*wz*wz;
          // Multiplying the rheological coefficient
          A_13 *= scndCoeff_arr(i,j,k,0);
          A_23 *= scndCoeff_arr(i,j,k,0);
          A_33 *= scndCoeff_arr(i,j,k,0);
          // THIS IS A COMPRESSIVE FORCE SO THERE NEEDS TO
          // BE A MINUS IN FRONT OF THE TERMS
          A_13 *= Real(-1.0); A_23 *= Real(-1.0); A_33 *= Real(-1.0);
          flux_arr(i,j,k,0) = A_13; flux_arr(i,j,k,1) = A_23;
          flux_arr(i,j,k,2) = A_33;
       });
   }
#endif
}

#ifdef AMREX_USE_EB
void
incflo::compute_granular_high_order_fluxes_on_level (MultiFab* flux_eb,
                                          const MultiFab* gradVel_EB,
                                          const MultiFab* scndOrderCoeff)
{
    flux_eb->setVal(Real(0.));
    const auto& factory =
      dynamic_cast<EBFArrayBoxFactory const&>(scndOrderCoeff->Factory());
    auto const& flags        = factory.getMultiEBCellFlagFab();
    MultiCutFab const& bnorm = factory.getBndryNormal();
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*scndOrderCoeff,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& bx = mfi.tilebox();
        auto const& flag_fab = flags[mfi];
        auto typ = flag_fab.getType(bx);
        if (typ == FabType::singlevalued) {
            auto const& flag_arr                    = flag_fab.const_array();
            Array4<Real const> const& bnrmfab       = bnorm.const_array(mfi);
            Array4<Real const> const& scndCoeff_arr = scndOrderCoeff->const_array(mfi);
            Array4<Real const> const& gradVel_arr   = gradVel_EB->const_array(mfi);
            Array4<Real      > const& flux_arr      = flux_eb->array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (flag_arr(i,j,k).isSingleValued()) {
                    const Real eta_2 = scndCoeff_arr(i,j,k);
                    const Real nx    = bnrmfab(i,j,k,0);
                    const Real ny    = bnrmfab(i,j,k,1);
                    Real ux = gradVel_arr(i,j,k,0);
                    Real vx = gradVel_arr(i,j,k,1);
#if (AMREX_SPACEDIM == 2)
                    Real uy = gradVel_arr(i,j,k,2);
                    Real vy = gradVel_arr(i,j,k,3);
#else
                    Real wx = gradVel_arr(i,j,k,2);
                    Real uy = gradVel_arr(i,j,k,3);
                    Real vy = gradVel_arr(i,j,k,4);
                    Real wy = gradVel_arr(i,j,k,5);
                    Real uz = gradVel_arr(i,j,k,6);
                    Real vz = gradVel_arr(i,j,k,7);
                    Real wz = gradVel_arr(i,j,k,8);
                    const Real nz    = bnrmfab(i,j,k,2);
#endif
#if (AMREX_SPACEDIM == 2)
                    Real A_11 =  Real(0.5)*(ux*ux-vy*vy);
                    Real A_12 =  Real(0.5)*(uy+vx)*(ux+vy);
                    Real A_22 =  Real(0.5)*(vy*vy-ux*ux);
                    A_11 *= Real(-1.0)*eta_2; A_12 *= Real(-1.0)*eta_2; A_22 *= Real(-1.0)*eta_2;
                    flux_arr(i,j,k,0) = A_11*nx + A_12*ny;
                    flux_arr(i,j,k,1) = A_12*nx + A_22*ny;
#else
                    Real A_11 =   (uy+vx)*(uy+vx)/Real(12.0) + (uz+wx)*(uz+wx)/Real(12.0)
                                  - (vz+wy)*(vz+wy)/Real(6.0) + Real(2.0)*ux*ux/Real(3.0)
                                  - vy*vy/Real(3.0) - wz*wz/Real(3.0);

                    Real A_12 =   Real(0.5)*(uy+vx)*(ux+vy) + Real(0.25)*(uz+wx)*(vz+wy);

                    Real A_13 =   Real(0.25)*(uy+vx)*(vz+wy) + Real(0.5)*(uz+wx)*(ux+wz);

                    Real A_22 =   (uy+vx)*(uy+vx)/Real(12.0) - (uz+wx)*(uz+wx)/Real(6.0)
                                  + (vz+wy)*(vz+wy)/Real(12.0) - ux*ux/Real(3.0)
                                  + Real(2.0)*vy*vy/Real(3.0) - wz*wz/Real(3.0);

                    Real A_23 =   Real(0.25)*(uy+vx)*(uz+wx) + Real(0.5)*(vz+wy)*(vy+wz);

                    Real A_33 =   -(uy+vx)*(uy+vx)/Real(6.0) + (uz+wx)*(uz+wx)/Real(12.0)
                                  + (vz+wy)*(vz+wy)/Real(12.0) - ux*ux/Real(3.0)
                                  - vy*vy/Real(3.0) + Real(2.0)*wz*wz;

                    A_11 *= Real(-1.0)*eta_2; A_12 *= Real(-1.0)*eta_2; A_13 *= Real(-1.0)*eta_2;
                    A_22 *= Real(-1.0)*eta_2; A_23 *= Real(-1.0)*eta_2; A_33 *= Real(-1.0)*eta_2;
                    flux_arr(i,j,k,0) = A_11*nx + A_12*ny + A_13*nz;
                    flux_arr(i,j,k,1) = A_12*nx + A_22*ny + A_23*nz;
                    flux_arr(i,j,k,2) = A_13*nx + A_23*ny + A_33*nz;
#endif
                }
            });
        }
    }
}
#endif

void incflo::compute_second_order_coeff (int lev, MultiFab& scnd_coeff,
                                          const MultiFab& velocity,
                                          const MultiFab& density,
                                          const MultiFab& conc_second,
                                          const MultiFab& p_static,
                                          Geometry& lev_geom)
{
    if (m_fluid_model_second == FluidModel::GranularPowerlaw
        && m_mu_powerlaw.size() > 1) {
        compute_granular_powerlaw_second_order_coeff(lev, scnd_coeff,
                velocity, density, conc_second, p_static, lev_geom);
    }
    else if (m_fluid_model_second == FluidModel::GranularPowerlawTemperature
             && m_mu_powerlaw_temperature.size() > 1) {
        compute_granular_powerlaw_temperature_second_order_coeff(
                lev, scnd_coeff, velocity, density, conc_second, p_static, lev_geom);
    }
    else if (m_probtype == 538) {
        scnd_coeff.setVal(Real(1.));
    }
    else {
        scnd_coeff.setVal(Real(0.));
    }
    // Take care of ghost cells
    if (scnd_coeff.nGrow() > 0) {
        scnd_coeff.FillBoundary(lev_geom.periodicity());
    }
#ifdef AMREX_USE_EB
    smooth_eb_cell_centered_coeff(lev, scnd_coeff, lev_geom);
#endif
    // Clamp the high-order granular coefficient with its own limiter range.
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(scnd_coeff,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& bx = mfi.growntilebox(scnd_coeff.nGrow());
        Array4<Real> const& scnd_coeff_arr = scnd_coeff.array(mfi);
        const Real eta_ho_min_scnd = m_eta_ho_min_second;
        const Real eta_ho_max_scnd = m_eta_ho_max_second;
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            scnd_coeff_arr(i,j,k) = amrex::Clamp(scnd_coeff_arr(i,j,k),
                                                 eta_ho_min_scnd,
                                                 eta_ho_max_scnd);
        });
    }
}

void incflo::compute_granular_powerlaw_temperature_second_order_coeff (
                                                           int lev, MultiFab& scnd_coeff,
                                                           const MultiFab& velocity,
                                                           const MultiFab& density,
                                                           const MultiFab& conc_second,
                                                           const MultiFab& p_static,
                                                           Geometry& lev_geom)
{
   amrex::ignore_unused(density);
   MultiFab sr_mf(velocity.boxArray(), velocity.DistributionMap(),1,0);
   compute_strainrate_at_level(lev,&sr_mf,&velocity,lev_geom,Real(0.0),0);

   MultiFab inertial_num(velocity.boxArray(),velocity.DistributionMap(),1,0);
   compute_inertial_num_at_level(lev,&inertial_num,
                                 &sr_mf,&p_static,m_mu_p_eps_second,
                                 m_ro_grain_second,m_diam_second,
                                 0);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
   for (MFIter mfi(sr_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
   {
       Box const& bx = mfi.tilebox();
       Array4<Real const> const& sr_arr         = sr_mf.const_array(mfi);
       Array4<Real const> const& p_static_arr   = p_static.const_array(mfi);
       Array4<Real const> const& inrt_num_arr   = inertial_num.const_array(mfi);
       Array4<Real const> const& conc_scnd_arr  = conc_second.const_array(mfi);
       Array4<Real const> const& temperature_arr =
           m_leveldata[lev]->temperature.const_array(mfi);
       Array4<Real      > const& scnd_coeff_arr = scnd_coeff.array(mfi);
       const Real temp_expnt  = Real(1.0/6.0);
       const Real a_I_c1      = m_mu_powerlaw_temperature[1][0];
       const Real a_I_e1      = m_mu_powerlaw_temperature[1][1];
       const Real a_I_c2      = m_mu_powerlaw_temperature[1][2];
       const Real a_I_e2      = m_mu_powerlaw_temperature[1][3];
       const Real a_I_c3      = m_mu_powerlaw_temperature[1][4];
       const Real a_I_e3      = m_mu_powerlaw_temperature[1][5];
       const Real eps           = m_mu_sr_eps_second;

       amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
            Real temperature_val = temperature_arr(i,j,k);
            amrex::ignore_unused(temperature_val);
            Real conc_val = conc_scnd_arr(i,j,k,0);
            Real inrt_num_val = inrt_num_arr(i,j,k);
            // Functional form from Kim and Kamrin, Frontiers in Physics (2023)
            scnd_coeff_arr(i,j,k) = a_I_c1*std::pow(inrt_num_val, a_I_e1)
                                    + a_I_c2*std::pow(inrt_num_val, a_I_e2)
                                    + a_I_c3*std::pow(inrt_num_val, a_I_e3);
            scnd_coeff_arr(i,j,k) /= std::pow(temperature_arr(i,j,k), temp_expnt);

            scnd_coeff_arr(i,j,k) *= p_static_arr(i,j,k);
            scnd_coeff_arr(i,j,k) /= ((Real(0.5)*sr_arr(i,j,k) + eps)
                                     * (Real(0.5)*sr_arr(i,j,k) + eps));
            scnd_coeff_arr(i,j,k) *= conc_val;
       });
   }
}
// Adding these high-order effects only in regions
// with Inertial number greater than Neutral Inertial Number
void incflo::compute_granular_powerlaw_second_order_coeff (int lev, MultiFab& scnd_coeff,
                                                           const MultiFab& velocity,
                                                           const MultiFab& density,
                                                           const MultiFab& conc_second,
                                                           const MultiFab& p_static,
                                                           Geometry& lev_geom)
{
   // Create a strain-rate MultiFab
   MultiFab sr_mf(velocity.boxArray(), velocity.DistributionMap(),1,0);
   // For now, NOT passing actual time
   compute_strainrate_at_level(lev,&sr_mf,&velocity,lev_geom,Real(0.0),0);
   // Inertial Number = diameter*strainrate*sqrt(rho_grain/p)
   // NOTE: Strain-rate calculated is TWO TIMES the actual value
   // The second component will carry concentration
   MultiFab inertial_num(velocity.boxArray(),velocity.DistributionMap(),1,0);
   compute_inertial_num_at_level(lev,&inertial_num,
                                 &sr_mf,&p_static,m_mu_p_eps_second,
                                 m_ro_grain_second,m_diam_second,
                                 0);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
   for (MFIter mfi(sr_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
   {
       Box const& bx = mfi.tilebox();
       Array4<Real const> const& sr_arr         = sr_mf.const_array(mfi);
       Array4<Real const> const& p_static_arr   = p_static.const_array(mfi);
       Array4<Real const> const& inrt_num_arr   = inertial_num.const_array(mfi);
       Array4<Real const> const& conc_scnd_arr  = conc_second.const_array(mfi);
       Array4<Real      > const& scnd_coeff_arr = scnd_coeff.array(mfi);
       const Real mu_const      = m_mu_powerlaw[1][0];
       const Real mu_A          = m_mu_powerlaw[1][1];
       const Real mu_alpha      = m_mu_powerlaw[1][2];
       const Real min_conc_scnd = m_min_conc_second;
       const Real eps           = m_mu_sr_eps_second;
       // Note: sr_mf contains TWO TIMES strain rate
       // Note: Inertial number in Rauter 2021 (Eq. 2.29)
       // has an extra factor of 2
       amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
            Real conc_val = conc_scnd_arr(i,j,k,0);
            Real inrt_num_val = inrt_num_arr(i,j,k);
            scnd_coeff_arr(i,j,k) = mu_const +
                                    mu_A * std::pow(inrt_num_val,
                                                    Real(2.0)*mu_alpha);
            scnd_coeff_arr(i,j,k) *= p_static_arr(i,j,k);
            scnd_coeff_arr(i,j,k) /= ((Real(0.5)*sr_arr(i,j,k) + eps)
                                     * (Real(0.5)*sr_arr(i,j,k) + eps));
            scnd_coeff_arr(i,j,k) *= conc_val;
       });
   }
}

void incflo::compute_tracer_diff_coeff (Vector<MultiFab*> const& tra_eta, int nghost) const
{
    for (auto *mf : tra_eta) {
        for (int n = 0; n < m_ntrac; ++n) {
            mf->setVal(m_mu_s[n], n, 1, nghost);
        }
    }
}

void incflo::compute_temperature_diff_coeff (Real /*time*/, Vector<MultiFab*> const& tem_eta) const
{
    for (auto *mf : tem_eta) { // loop over levels
        mf->setVal(m_mu_T);
    }
}
