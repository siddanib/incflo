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

#ifdef AMREX_USE_EB
void incflo::compute_viscosity_at_level (int lev,
#else
void incflo::compute_viscosity_at_level (int /*lev*/,
#endif
                                         MultiFab* vel_eta,
                                         MultiFab* /*rho*/,
                                         MultiFab* vel,
                                         Geometry& lev_geom,
                                         Real /*time*/, int nghost)
{
    if (m_fluid_model == FluidModel::Newtonian)
    {
        vel_eta->setVal(m_mu, 0, 1, nghost);
    }
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

#if defined(USE_AMREX_MPMD) || defined(AMREX_USE_EB)
void incflo::compute_nodal_viscosity_at_level (int lev,
#else
void incflo::compute_nodal_viscosity_at_level (int /*lev*/,
#endif
                                         MultiFab* vel_eta,
                                         MultiFab* rho,
                                         MultiFab* vel,
                                         Geometry& lev_geom,
                                         Real /*time*/, int nghost)
{
    if (m_fluid_model == FluidModel::Newtonian)
    {
        vel_eta->setVal(m_mu, 0, 1, nghost);
    }
    else {
        // Create a nodal strain-rate MultiFab, nghost is already set to 0
        MultiFab sr_mf(vel_eta->boxArray(),vel_eta->DistributionMap(),1,nghost);
#ifdef AMREX_USE_EB
        auto const& fact = EBFactory(lev);
        auto const& flags = fact.getMultiEBCellFlagFab();
#endif
        Real idx = Real(1.0) / lev_geom.CellSize(0);
        Real idy = Real(1.0) / lev_geom.CellSize(1);
#if (AMREX_SPACEDIM == 3)
        Real idz = Real(1.0) / lev_geom.CellSize(2);
#endif
        const Dim3 dlo = amrex::lbound(lev_geom.Domain());
        const Dim3 dhi = amrex::ubound(lev_geom.Domain());
        GpuArray<GpuArray<int,2>,AMREX_SPACEDIM> bc_type;
        for (OrientationIter oit; oit; ++oit) {
            Orientation ori = oit();
            int dir = ori.coordDir();
            Orientation::Side side = ori.faceDir();
            auto const bct = m_bc_type[ori];
            if (bct == BC::no_slip_wall) {
                if (side == Orientation::low) {
                    bc_type[dir][0] = 2;
                }
                if (side == Orientation::high) {
                    bc_type[dir][1] = 2;
                }
            }
            else if (bct == BC::slip_wall) {
                if (side == Orientation::low) {
                    bc_type[dir][0] = 1;
                }
                if (side == Orientation::high) {
                    bc_type[dir][1] = 1;
                }
            }
            else {
                if (side == Orientation::low) {
                    bc_type[dir][0] = 0;
                }
                if (side == Orientation::high) {
                    bc_type[dir][1] = 0;
                }
            }
        }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(sr_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.growntilebox(nghost);
            Array4<Real> const& sr_arr = sr_mf.array(mfi);
            Array4<Real const> const& vel_arr = vel->const_array(mfi);
#ifdef AMREX_USE_EB
            auto const& flag_fab = flags[mfi];
            auto typ = flag_fab.getType(bx);
            if (typ == FabType::covered)
            {
                amrex::Abort("Node-based vel_eta is not implemented for EB\n");
            }
            else if (typ == FabType::singlevalued)
            {
                amrex::Abort("Node-based vel_eta is not implemented for EB\n");
            }
            else
#endif
            {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    sr_arr(i,j,k) = incflo_strainrate_nodal(i,j,k,AMREX_D_DECL(idx,idy,idz),
                                                   vel_arr,dlo,dhi,bc_type);
                });
            }
        }
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
       const Dim3 dlo = amrex::lbound(lev_geom.Domain());
       const Dim3 dhi = amrex::ubound(lev_geom.Domain());
       GpuArray<GpuArray<int,2>,AMREX_SPACEDIM> bc_type;
       for (OrientationIter oit; oit; ++oit) {
           Orientation ori = oit();
           int dir = ori.coordDir();
           Orientation::Side side = ori.faceDir();
           auto const bct = m_bc_type[ori];
           if (bct == BC::no_slip_wall) {
               if (side == Orientation::low) {
                   bc_type[dir][0] = 2;
               }
               if (side == Orientation::high) {
                   bc_type[dir][1] = 2;
               }
           }
           else if (bct == BC::slip_wall) {
               if (side == Orientation::low) {
                   bc_type[dir][0] = 1;
               }
               if (side == Orientation::high) {
                   bc_type[dir][1] = 1;
               }
           }
           else {
               if (side == Orientation::low) {
                   bc_type[dir][0] = 0;
               }
               if (side == Orientation::high) {
                   bc_type[dir][1] = 0;
               }
           }
       }
       // A cell-centered MultiFab for concentration of second fluid,
       // needs to have ghost cells
       MultiFab conc_second_cc(rho->boxArray(),rho->DistributionMap(),1,1);
       conc_second_cc.setVal(-1.0);
       if (m_two_fluid_cc_rho_conc) {
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
          for (MFIter mfi(conc_second_cc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
          {
              Box const& bx = mfi.tilebox();
              Array4<Real const> const& rho_arr = rho->const_array(mfi);
              Array4<Real> const& conc_second_arr = conc_second_cc.array(mfi);
              const Real rho_first = m_ro_0;
              const Real rho_second = m_ro_0_second;
              amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
              {
                 // Based on weighted harmonic mean for cell-centered density
                 Real conc_scnd =
                   ((rho_first*rho_second)/rho_arr(i,j,k)) - rho_second;
                 conc_scnd /= (rho_first-rho_second);
                 // Put guards
                 conc_second_arr(i,j,k) =
                   amrex::min(Real(1.0),amrex::max(Real(0.0),conc_scnd));
              });
          }
          conc_second_cc.FillBoundary(lev_geom.periodicity());
       }

       // Obtain concentration of the second fluid, based on nodal density
       MultiFab rho_nodal(vel_eta->boxArray(),vel_eta->DistributionMap(),1,nghost);
       // Nodal second fluid concentration MultiFab
       MultiFab conc_second_nd(vel_eta->boxArray(),vel_eta->DistributionMap(),1,nghost);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
       for (MFIter mfi(conc_second_nd,TilingIfNotGPU()); mfi.isValid(); ++mfi)
       {
           Box const& bx = mfi.growntilebox(nghost);
           Array4<Real const> const& rho_arr = rho->const_array(mfi);
           Array4<Real const> const& conc_second_cc_arr = conc_second_cc.const_array(mfi);
           Array4<Real> const& rho_nodal_arr = rho_nodal.array(mfi);
           Array4<Real> const& conc_second_nd_arr = conc_second_nd.array(mfi);
           const Real rho_first = m_ro_0;
           const Real rho_second = m_ro_0_second;
           // This boolean represents if concentration is calculated based on
           // nodal or cell-centered density
           const bool cc_rho_conc = m_two_fluid_cc_rho_conc;
           amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
           {
              rho_nodal_arr(i,j,k) = incflo_nodal_density(i,j,k,
                                         rho_first,rho_second,rho_arr,
                                         dlo,dhi,bc_type);
              if (cc_rho_conc) {
                conc_second_nd_arr(i,j,k) = incflo_nodal_second_conc(i,j,k,
                                                    conc_second_cc_arr,
                                                    dlo,dhi,bc_type);
              }
              else {
                // Based on weighted harmonic mean for nodal density
                Real conc_scnd =
                  ((rho_first*rho_second)/rho_nodal_arr(i,j,k)) - rho_second;
                conc_scnd /= (rho_first-rho_second);
                // Put guards
                conc_second_nd_arr(i,j,k) =
                  amrex::min(Real(1.0),amrex::max(Real(0.0),conc_scnd));
                }
           });
       }

       if (m_fluid_model_second == FluidModel::Newtonian)
       {
           vel_eta_second.setVal(m_mu_second, 0, 1, nghost);
       }
       else {
           // Create a nodal strain-rate MultiFab, nghost is already set to 0
           MultiFab sr_mf(vel_eta->boxArray(),vel_eta->DistributionMap(),1,nghost);
           // nodal MultiFab for hydrostatic pressure
           MultiFab p_static(vel_eta->boxArray(),vel_eta->DistributionMap(),1,nghost);
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
           for (MFIter mfi(sr_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
           {
               Box const& bx = mfi.growntilebox(nghost);
               Array4<Real> const& sr_arr = sr_mf.array(mfi);
               Array4<Real const> const& vel_arr = vel->const_array(mfi);

               // Ensure that static pressure is calculated based on validbox
               Box const& v_bx = mfi.validbox();
               const Dim3 v_bxlo = amrex::lbound(v_bx);
               const Dim3 v_bxhi = amrex::ubound(v_bx);
               Array4<Real> const& p_static_arr = p_static.array(mfi);
               Array4<Real const> const& rho_nodal_arr = rho_nodal.const_array(mfi);
               const Real gravity = std::abs(m_gravity[AMREX_SPACEDIM-1]);
#ifdef AMREX_USE_EB
               auto const& flag_fab = flags[mfi];
               auto typ = flag_fab.getType(bx);
               if (typ == FabType::covered)
               {
                   amrex::Abort("Node-based vel_eta_second is not implemented for EB\n");
               }
               else if (typ == FabType::singlevalued)
               {
                   amrex::Abort("Node-based vel_eta_second is not implemented for EB\n");
               }
               else
#endif
               {
                   amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                   {
                       sr_arr(i,j,k) = incflo_strainrate_nodal(i,j,k,AMREX_D_DECL(idx,idy,idz),
                                                      vel_arr,dlo,dhi,bc_type);

                       p_static_arr(i,j,k) = incflo_local_hydrostatic_pressure_nodal(
                                                      i,j,k,AMREX_D_DECL(idx,idy,idz),
                                                      gravity,rho_nodal_arr,v_bxlo,v_bxhi);
                   });
               }
           }
           // NOTE: HYDROSTATIC PRESSURE IS INCORRECT IF THERE ARE
           // MULTIPLE BOXES/GRIDS ALONG GRAVITY DIRECTION

           // Inertial Number = diameter*strainrate*sqrt(rho_grain/p)
           // NOTE: Strain-rate calculated is TWO TIMES the actual value
           // The second component will carry concentration
           MultiFab inertial_num(vel_eta->boxArray(),vel_eta->DistributionMap(),2,nghost);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
           for (MFIter mfi(sr_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
           {
               Box const& bx = mfi.growntilebox(nghost);
               Array4<Real const> const& sr_arr = sr_mf.const_array(mfi);
               Array4<Real const> const& p_static_arr = p_static.const_array(mfi);
               Array4<Real> const& inrt_num_arr = inertial_num.array(mfi);
               const Real eps = Real(1.0e-20);
               const Real diam_scnd = m_diam_second;
               const Real ro_scnd = m_ro_grain_second;
               amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
               {
                    inrt_num_arr(i,j,k,0) =
                       std::sqrt(ro_scnd/(p_static_arr(i,j,k)+eps))*
                       diam_scnd*Real(0.5)*sr_arr(i,j,k);
               });
           }
           // Copy concentration
           MultiFab::Copy(inertial_num,conc_second_nd,0,1,1,nghost);

#ifdef USE_AMREX_MPMD
           if (m_fluid_model_second == FluidModel::DataDrivenMPMD) {
               // Copier send inertial_num
               mpmd_copiers_send_lev(inertial_num,0,2,lev);
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
                   const Real eps = Real(1.0e-20);
                   // Note: sr_mf contains TWO TIMES strain rate
                   amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                   {
                        vel_eta_snd_arr(i,j,k) *= p_static_arr(i,j,k);
                        vel_eta_snd_arr(i,j,k) /= (sr_arr(i,j,k)+eps);
                   });
               }

           } else
#endif
           if (m_fluid_model_second == FluidModel::Rauter) {
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
                   const Real eps = Real(1.0e-20);
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
                        vel_eta_snd_arr(i,j,k) *= p_static_arr(i,j,k);
                        vel_eta_snd_arr(i,j,k) /= (sr_arr(i,j,k)+eps);
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
       // Calculate weighted viscosity
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

void incflo::compute_tracer_diff_coeff (Vector<MultiFab*> const& tra_eta, int nghost)
{
    for (auto *mf : tra_eta) {
        for (int n = 0; n < m_ntrac; ++n) {
            mf->setVal(m_mu_s[n], n, 1, nghost);
        }
    }
}
