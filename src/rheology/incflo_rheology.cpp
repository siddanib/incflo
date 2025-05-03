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

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (m_nodal_vel_eta) {
            compute_nodal_viscosity_at_level(lev, vel_eta[lev], rho[lev],
                    vel[lev], geom[lev], time, 0);
        } else {
            compute_viscosity_at_level(lev, vel_eta[lev], rho[lev],
                    vel[lev], geom[lev], time, nghost);
        }
    }
}

void incflo::compute_viscosity_at_level (int lev,
                                         MultiFab* vel_eta,
                                         MultiFab* /*rho*/,
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
        compute_strainrate_at_level(lev, vel_eta, vel, lev_geom,
                                    time, nghost);
        mpmd_copiers_send_lev(*vel_eta,0,1,lev);
        mpmd_copiers_recv_lev(*vel_eta,0,1,lev);
    }
#endif
    else
    {
        NonNewtonianViscosity non_newtonian_viscosity;
        non_newtonian_viscosity.fluid_model = m_fluid_model;
        non_newtonian_viscosity.mu = m_mu;
        non_newtonian_viscosity.n_flow = m_n_0;
        non_newtonian_viscosity.tau_0 = m_tau_0;
        non_newtonian_viscosity.eta_0 = m_eta_0;
        non_newtonian_viscosity.papa_reg = m_papa_reg;

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
}

void incflo::compute_nodal_viscosity_at_level (int lev,
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
    else {
        // Create a nodal strain-rate MultiFab, nghost is already set to 0
        MultiFab sr_mf(vel_eta->boxArray(),vel_eta->DistributionMap(),1,nghost);
        compute_nodal_strainrate_at_level(lev,&sr_mf,vel,lev_geom,time,nghost);
#ifdef USE_AMREX_MPMD
        if (m_fluid_model == FluidModel::DataDrivenMPMD) {
            // Copier send of sr_mf and Copier recv of *vel_eta
            mpmd_copiers_send_lev(sr_mf,0,1,lev);
            mpmd_copiers_recv_lev(*vel_eta,0,1,lev);
        } else
#endif
        {
            NonNewtonianViscosity non_newtonian_viscosity;
            non_newtonian_viscosity.fluid_model = m_fluid_model;
            non_newtonian_viscosity.mu = m_mu;
            non_newtonian_viscosity.n_flow = m_n_0;
            non_newtonian_viscosity.tau_0 = m_tau_0;
            non_newtonian_viscosity.eta_0 = m_eta_0;
            non_newtonian_viscosity.papa_reg = m_papa_reg;
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(sr_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box const& bx = mfi.growntilebox(nghost);
                Array4<Real const> const& sr_arr = sr_mf.const_array(mfi);
                Array4<Real> const& eta_arr = vel_eta->array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    eta_arr(i,j,k) = non_newtonian_viscosity(sr_arr(i,j,k));
                });
            }
        }
        // Clamp vel_eta if it is NOT-NEWTONIAN
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
       // Create a nodal viscosity MultiFab for the second fluid, nghost is already set to 0
       MultiFab vel_eta_second(vel_eta->boxArray(),vel_eta->DistributionMap(),1,nghost);
       // Nodal second fluid concentration MultiFab
       MultiFab conc_second_nd(vel_eta->boxArray(),vel_eta->DistributionMap(),1,nghost);
       compute_nodal_second_fluid_conc(&conc_second_nd,rho,nghost);
       // Calculate second fluid viscosity
       compute_nodal_second_fluid_viscosity_at_level(lev, vel_eta, rho, vel, lev_geom,
                                                     time, nghost, vel_eta_second,
                                                     conc_second_nd);
       // Calculate weighted viscosity
       if (!(m_mu-m_mu_second == Real(0.) and m_fluid_model == m_fluid_model_second)) {
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
         for (MFIter mfi(conc_second_nd,TilingIfNotGPU()); mfi.isValid(); ++mfi)
         {
             Box const& bx = mfi.growntilebox(nghost);
             Array4<Real const> const& conc_second_arr = conc_second_nd.array(mfi);
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
       }
    }
}

void incflo::compute_nodal_second_fluid_viscosity_at_level (int lev,
                                         MultiFab* vel_eta,
                                         MultiFab* rho,
                                         MultiFab* vel,
                                         Geometry& lev_geom,
                                         Real time, int nghost,
                                         MultiFab& vel_eta_second,
                                         MultiFab& conc_second_nd
                                         )
{
   if (m_fluid_model_second == FluidModel::Newtonian)
   {
       vel_eta_second.setVal(m_mu_second, 0, 1, nghost);
   }
   else
   {
       // Create a nodal strain-rate MultiFab, nghost is already set to 0
       MultiFab sr_mf(vel_eta->boxArray(),vel_eta->DistributionMap(),1,nghost);
       compute_nodal_strainrate_at_level(lev,&sr_mf,vel,lev_geom,time,nghost);
       // nodal MultiFab for hydrostatic pressure
       MultiFab p_static(vel_eta->boxArray(),vel_eta->DistributionMap(),1,nghost);
       compute_nodal_hydrostatic_pressure_at_level(lev,&p_static,rho,
                                                   m_mu_p_surf_second,
                                                   lev_geom,nghost);

#ifdef USE_AMREX_MPMD
       if (m_fluid_model_second == FluidModel::DataDrivenMPMD) {
           MultiFab inertial_num_mpmd(vel_eta->boxArray(),vel_eta->DistributionMap(),
                                      2,nghost);
           // Inertial Number = diameter*strainrate*sqrt(rho_grain/p)
           // NOTE: Strain-rate calculated is TWO TIMES the actual value
           // The second component will carry concentration
           MultiFab inertial_num(inertial_num_mpmd,amrex::make_alias,0,1);
           compute_nodal_inertial_num_at_level(lev,&inertial_num,
                                               &sr_mf,&p_static,m_mu_p_eps_second,
                                               m_ro_grain_second,m_diam_second,
                                               nghost);
           // Copy concentration
           MultiFab::Copy(inertial_num_mpmd,conc_second_nd,0,1,1,nghost);
           // Copier send inertial_num_mpmd
           mpmd_copiers_send_lev(inertial_num_mpmd,0,2,lev);
           // NOTE: Actual received quantity is stress ratio
           mpmd_copiers_recv_lev(vel_eta_second,0,1,lev);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
           for (MFIter mfi(sr_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
           {
               Box const& bx = mfi.growntilebox(nghost);
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
       if (m_fluid_model_second == FluidModel::Rauter) {
          // Inertial Number = diameter*strainrate*sqrt(rho_grain/p)
          // NOTE: Strain-rate calculated is TWO TIMES the actual value
          // The second component will carry concentration
          MultiFab inertial_num(vel_eta->boxArray(),vel_eta->DistributionMap(),
                                1,nghost);
          compute_nodal_inertial_num_at_level(lev,&inertial_num,
                                              &sr_mf,&p_static,m_mu_p_eps_second,
                                              m_ro_grain_second,m_diam_second,
                                              nghost);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
          for (MFIter mfi(sr_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
          {
              Box const& bx = mfi.growntilebox(nghost);
              Array4<Real const> const& sr_arr = sr_mf.const_array(mfi);
              Array4<Real const> const& p_static_arr = p_static.const_array(mfi);
              Array4<Real const> const& inrt_num_arr = inertial_num.const_array(mfi);
              Array4<Real> const& vel_eta_snd_arr = vel_eta_second.array(mfi);
              const Real mu_1_scnd = m_mu_1_second;
              const Real mu_2_scnd = m_mu_2_second;
              const Real I_0_scnd = m_I_0_second;
              const Real eps = m_mu_sr_eps_second;
              // Note: sr_mf contains TWO TIMES strain rate
              // Note: Inertial number in Rauter 2021 (Eq. 2.29)
              // has an extra factor of 2
              amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
              {
                   vel_eta_snd_arr(i,j,k) = inrt_num_arr(i,j,k,0);
                   vel_eta_snd_arr(i,j,k) /= (I_0_scnd + inrt_num_arr(i,j,k,0));
                   vel_eta_snd_arr(i,j,k) *= (mu_2_scnd-mu_1_scnd);
                   vel_eta_snd_arr(i,j,k) += mu_1_scnd;
                   // The above value is stress ratio
                   // Regularized strain rate
                   Real sr_reg = Real(0.5)*sr_arr(i,j,k) + eps;
                   vel_eta_snd_arr(i,j,k) *= p_static_arr(i,j,k);
                   vel_eta_snd_arr(i,j,k) /= (Real(2.0)*sr_reg);
              });
          }

      } else
      {
          NonNewtonianViscosity non_newtonian_viscosity;
          non_newtonian_viscosity.fluid_model = m_fluid_model_second;
          non_newtonian_viscosity.mu = m_mu_second;
          non_newtonian_viscosity.n_flow = m_n_0_second;
          non_newtonian_viscosity.tau_0 = m_tau_0_second;
          non_newtonian_viscosity.eta_0 = m_eta_0_second;
          non_newtonian_viscosity.papa_reg = m_papa_reg_second;
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
          for (MFIter mfi(sr_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
          {
              Box const& bx = mfi.growntilebox(nghost);
              Array4<Real const> const& sr_arr = sr_mf.const_array(mfi);
              Array4<Real> const& eta_arr = vel_eta_second.array(mfi);
              amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
              {
                  eta_arr(i,j,k) = non_newtonian_viscosity(sr_arr(i,j,k));
              });
          }
      }
      // Clamp vel_eta_second if it is NOT-NEWTONIAN
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

void incflo::compute_tracer_diff_coeff (Vector<MultiFab*> const& tra_eta, int nghost)
{
    for (auto *mf : tra_eta) {
        for (int n = 0; n < m_ntrac; ++n) {
            mf->setVal(m_mu_s[n], n, 1, nghost);
        }
    }
}
