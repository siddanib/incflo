#include <incflo.H>
#include <incflo_derive_K.H>

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
             || m_fluid_model_second == FluidModel::GranularPowerlaw) {
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
            && m_fluid_model_second != FluidModel::GranularPowerlaw)
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
           m_fluid_model_second == FluidModel::GranularPowerlaw) {
          GranularViscosity granvisc;
          granvisc.fluid_model = m_fluid_model_second;
          if (m_fluid_model_second == FluidModel::Rauter)
          {
             granvisc.set_rauter_parameters(m_mu_1_second, m_mu_2_second,
                                            m_I_0_second);
          }
          else
          {
             granvisc.set_granularpowerlaw_parameters(m_mu_powerlaw[0][0],
                                                      m_mu_powerlaw[0][1],
                                                      m_mu_powerlaw[0][2],
                                                      m_I_1_N_powerlaw);
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
#endif

    Real idx = Real(1.0) / lev_geom.CellSize(0);
    Real idy = Real(1.0) / lev_geom.CellSize(1);
#if (AMREX_SPACEDIM == 3)
    Real idz = Real(1.0) / lev_geom.CellSize(2);
#endif

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
                    Real sr = incflo_strainrate_eb(i,j,k,AMREX_D_DECL(idx,idy,idz),vel_arr,flag_arr(i,j,k));
                    eta_arr(i,j,k) = non_newtonian_viscosity(sr);
                });
            }
            else
#endif
            {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real sr = incflo_strainrate(i,j,k,AMREX_D_DECL(idx,idy,idz),vel_arr);
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
                                    int nghost)
{
  GranularViscosity granvisc;
  granvisc.fluid_model = m_fluid_model_second;
  if (m_fluid_model_second == FluidModel::Rauter)
  {
     granvisc.set_rauter_parameters(m_mu_1_second, m_mu_2_second,
                                    m_I_0_second);
  }
  else
  {
     granvisc.set_granularpowerlaw_parameters(m_mu_powerlaw[0][0],
                                              m_mu_powerlaw[0][1],
                                              m_mu_powerlaw[0][2],
                                              m_I_1_N_powerlaw);
  }
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

// This function is to consider
// high-order terms in Granular Rheology
void incflo::add_granular_high_order_divtau_on_level (int lev, MultiFab& divtau,
                                                      const MultiFab& velocity,
                                                      const MultiFab& conc_second,
                                                      const MultiFab& scndOrderCoeff)
{
   // Creating velocity gradient MultiFab
    MultiFab gradVel(velocity.boxArray(), velocity.DistributionMap(),
                    AMREX_SPACEDIM*AMREX_SPACEDIM,0);
    gradVel.setVal(Real(0.0),0,AMREX_SPACEDIM*AMREX_SPACEDIM,0);
    Geometry& lev_geom = geom[lev];
    compute_gradientOfVelocity_on_level(lev,gradVel,velocity,lev_geom);
    // The following vectors are needed for divergence
    MultiFab vecX(velocity.boxArray(), velocity.DistributionMap(),
                  AMREX_SPACEDIM,1);
    MultiFab vecY(velocity.boxArray(), velocity.DistributionMap(),
                  AMREX_SPACEDIM,1);
#if (AMREX_SPACEDIM == 3)
    MultiFab vecZ(velocity.boxArray(), velocity.DistributionMap(),
                  AMREX_SPACEDIM,1);
#endif

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
   for (MFIter mfi(gradVel,TilingIfNotGPU()); mfi.isValid(); ++mfi)
   {
       Box const& bx = mfi.tilebox();
       Array4<Real const> const& gradVel_arr = gradVel.const_array(mfi);
       Array4<Real const> const& scndCoeff_arr = scndOrderCoeff.const_array(mfi);
       Array4<Real      > const& vecX_arr  = vecX.array(mfi);
       Array4<Real      > const& vecY_arr  = vecY.array(mfi);
#if (AMREX_SPACEDIM == 3)
       Array4<Real      > const& vecZ_arr  = vecZ.array(mfi);
#endif
       amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
          Real ux = gradVel_arr(i,j,k,0);
          Real uy = gradVel_arr(i,j,k,1);
#if (AMREX_SPACEDIM == 2)
          Real vx = gradVel_arr(i,j,k,2);
          Real vy = gradVel_arr(i,j,k,3);
#else
          Real uz = gradVel_arr(i,j,k,2);
          Real vx = gradVel_arr(i,j,k,3);
          Real vy = gradVel_arr(i,j,k,4);
          Real vz = gradVel_arr(i,j,k,5);
          Real wx = gradVel_arr(i,j,k,6);
          Real wy = gradVel_arr(i,j,k,7);
          Real wz = gradVel_arr(i,j,k,8);
#endif

#if (AMREX_SPACEDIM == 2)
          Real A_11 =  Real(0.5)*(ux*ux-vy*vy);

          Real A_12 =  Real(0.5)*(uy+vx)*(ux+vy);

          Real A_22 =  Real(0.5)*(vy*vy-ux*ux);
          // Multiplying the rheological coefficient
          A_11 *= scndCoeff_arr(i,j,k,0);
          A_12 *= scndCoeff_arr(i,j,k,0);
          A_22 *= scndCoeff_arr(i,j,k,0);
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

          Real A_22 =   (uy+vx)*(uy+vx)/Real(12.0)
                        - (uz+wx)*(uz+wx)/Real(6.0)
                        + (vz+wy)*(vz+wy)/Real(12.0)
                        - ux*ux/Real(3.0)
                        + Real(2.0)*vy*vy/Real(3.0)
                        - wz*wz/Real(3.0);

          Real A_23 =   Real(0.25)*(uy+vx)*(uz+wx)
                        + Real(0.5)*(vz+wy)*(vy+wz);

          Real A_33 =   -(uy+vx)*(uy+vx)/Real(6.0)
                        + (uz+wx)*(uz+wx)/Real(12.0)
                        + (vz+wy)*(vz+wy)/Real(12.0)
                        - ux*ux/Real(3.0)
                        - vy*vy/Real(3.0)
                        + Real(2.0)*wz*wz;
          // Multiplying the rheological coefficient
          A_11 *= scndCoeff_arr(i,j,k,0);
          A_12 *= scndCoeff_arr(i,j,k,0);
          A_13 *= scndCoeff_arr(i,j,k,0);
          A_22 *= scndCoeff_arr(i,j,k,0);
          A_23 *= scndCoeff_arr(i,j,k,0);
          A_33 *= scndCoeff_arr(i,j,k,0);
#endif
          // THIS IS A COMPRESSIVE FORCE SO THERE NEEDS TO
          // BE A MINUS IN FRONT OF THE TERMS
          A_11 *= Real(-1.0); A_12 *= Real(-1.0);  A_22 *= Real(-1.0);
#if (AMREX_SPACEDIM == 3)
          A_13 *= Real(-1.0); A_23 *= Real(-1.0); A_33 *= Real(-1.0);
#endif
          // Populating the vectors
          vecX_arr(i,j,k,0) = A_11;
          vecX_arr(i,j,k,1) = A_12;
          vecY_arr(i,j,k,0) = A_12;
          vecY_arr(i,j,k,1) = A_22;
#if (AMREX_SPACEDIM == 3)
          vecX_arr(i,j,k,2) = A_13;
          vecY_arr(i,j,k,2) = A_23;
          vecZ_arr(i,j,k,0) = A_13;
          vecZ_arr(i,j,k,1) = A_23;
          vecZ_arr(i,j,k,2) = A_33;
#endif
       });
   }
   vecX.FillBoundary(lev_geom.periodicity());
   vecY.FillBoundary(lev_geom.periodicity());
#if (AMREX_SPACEDIM == 3)
   vecZ.FillBoundary(lev_geom.periodicity());
#endif

   // Divergence of each individual vector, i.e., vecX, vecY, vecZ
#ifdef AMREX_USE_EB
    auto const& fact = EBFactory(lev);
    auto const& flags = fact.getMultiEBCellFlagFab();
#endif

    Real idx = Real(1.0) / lev_geom.CellSize(0);
    Real idy = Real(1.0) / lev_geom.CellSize(1);
#if (AMREX_SPACEDIM == 3)
    Real idz = Real(1.0) / lev_geom.CellSize(2);
#endif

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(divtau,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
            Box const& bx = mfi.tilebox();
            Array4<Real const> const& vecX_arr  = vecX.const_array(mfi);
            Array4<Real const> const& vecY_arr  = vecY.const_array(mfi);
#if (AMREX_SPACEDIM == 3)
            Array4<Real const> const& vecZ_arr  = vecZ.const_array(mfi);
#endif
            Array4<Real const> const& conc_scnd_arr  = conc_second.const_array(mfi);
            Array4<Real      > const& divtau_arr   = divtau.array(mfi);
            const Real min_conc_scnd = m_min_conc_second;
#ifdef AMREX_USE_EB
            auto const& flag_fab = flags[mfi];
            auto typ = flag_fab.getType(bx);
            if (typ == FabType::covered)
            {
                // Do nothing
            }
            else if (typ == FabType::singlevalued)
            {
                auto const& flag_arr = flag_fab.const_array();
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                   Real conc_val = conc_scnd_arr(i,j,k,0);
                   if (conc_val > min_conc_scnd) {
                      divtau_arr(i,j,k,0) += incflo_divergenceOfVector_eb(i,j,k,
                                                              AMREX_D_DECL(idx,idy,idz),
                                                              vecX_arr,flag_arr(i,j,k));
                      divtau_arr(i,j,k,1) += incflo_divergenceOfVector_eb(i,j,k,
                                                              AMREX_D_DECL(idx,idy,idz),
                                                              vecY_arr,flag_arr(i,j,k));
#if (AMREX_SPACEDIM == 3)
                      divtau_arr(i,j,k,2) += incflo_divergenceOfVector_eb(i,j,k,
                                                              AMREX_D_DECL(idx,idy,idz),
                                                              vecZ_arr,flag_arr(i,j,k));
#endif
                   }
                });
            }
            else
#endif
            {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                   Real conc_val = conc_scnd_arr(i,j,k,0);
                   if (conc_val > min_conc_scnd) {
                      divtau_arr(i,j,k,0) += incflo_divergenceOfVector(i,j,k,
                                                              AMREX_D_DECL(idx,idy,idz),
                                                              vecX_arr);
                      divtau_arr(i,j,k,1) += incflo_divergenceOfVector(i,j,k,
                                                              AMREX_D_DECL(idx,idy,idz),
                                                              vecY_arr);
#if (AMREX_SPACEDIM == 3)
                      divtau_arr(i,j,k,2) += incflo_divergenceOfVector(i,j,k,
                                                              AMREX_D_DECL(idx,idy,idz),
                                                              vecZ_arr);
#endif
                   }
                });
            }
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
       const Real I_neutral     = m_I_1_N_powerlaw;
       // Note: sr_mf contains TWO TIMES strain rate
       // Note: Inertial number in Rauter 2021 (Eq. 2.29)
       // has an extra factor of 2
       amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
            Real conc_val = conc_scnd_arr(i,j,k,0);
            Real inrt_num_val = inrt_num_arr(i,j,k);
            if (conc_val > min_conc_scnd &&
                inrt_num_val > I_neutral) {
              scnd_coeff_arr(i,j,k) = mu_const +
                                      mu_A * std::pow(inrt_num_val,
                                                      Real(2.0)*mu_alpha);
              scnd_coeff_arr(i,j,k) *= p_static_arr(i,j,k);
              scnd_coeff_arr(i,j,k) /= ((Real(0.5)*sr_arr(i,j,k) + eps)
                                       * (Real(0.5)*sr_arr(i,j,k) + eps));
            }
            else {
               scnd_coeff_arr(i,j,k) = Real(0.);
            }
       });
   }
}

void incflo::compute_tracer_diff_coeff (Vector<MultiFab*> const& tra_eta, int nghost)
{
    for (auto *mf : tra_eta) {
        for (int n = 0; n < m_ntrac; ++n) {
            mf->setVal(m_mu_s[n], n, 1, nghost);
        }
    }
}
