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
                                         MultiFab* /*rho*/,
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
