#include <incflo.H>
#include <incflo_derive_K.H>
#ifdef USE_AMREX_MPMD
#include <AMReX_MPMD.H>
#endif

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
        compute_viscosity_at_level(lev, vel_eta[lev], rho[lev], vel[lev], geom[lev], time, nghost);
    }
}

#ifdef AMREX_USE_EB
void incflo::compute_viscosity_at_level (int lev,
#else
void incflo::compute_viscosity_at_level (int /*lev*/,
#endif
                                         MultiFab* vel_eta,
#ifdef USE_AMREX_MPMD
                                         MultiFab* rho,
#else
                                         MultiFab* /*rho*/,
#endif
                                         MultiFab* vel,
                                         Geometry& lev_geom,
#ifdef USE_AMREX_MPMD
                                         Real time,
#else
                                         Real /*time*/,
#endif
                                         int nghost)
{
    if (m_fluid_model == FluidModel::Newtonian)
    {
        vel_eta->setVal(m_mu, 0, 1, nghost);
    }
#ifdef USE_AMREX_MPMD
    else if (m_fluid_model == FluidModel::DataDrivenMPMD)
    {
        // MPMD based vel_eta is nodal, nghost = 0
        compute_viscosity_at_level_mpmd(lev, vel_eta,
                rho, vel, lev_geom, time, 0);
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
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        eta_arr(i,j,k) = Real(0.0);
                    });
                }
                else if (typ == FabType::singlevalued)
                {
                    auto const& flag_arr = flag_fab.const_array();
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        Real sr = incflo_strainrate_eb(i,j,k,AMREX_D_DECL(idx,idy,idz),vel_arr,flag_arr(i,j,k));
                        eta_arr(i,j,k) = non_newtonian_viscosity(sr);
                    });
                }
                else
#endif
                {
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        Real sr = incflo_strainrate(i,j,k,AMREX_D_DECL(idx,idy,idz),vel_arr);
                        eta_arr(i,j,k) = non_newtonian_viscosity(sr);
                    });
                }
        }
    }
}

#ifdef USE_AMREX_MPMD
void incflo::compute_viscosity_at_level_mpmd (int lev,
                                         MultiFab* vel_eta,
                                         MultiFab* /*rho*/,
                                         MultiFab* vel,
                                         Geometry& lev_geom,
                                         Real /*time*/, int nghost)
{
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
    GpuArray<GpuArray<Real,2*AMREX_SPACEDIM>,AMREX_SPACEDIM> bc_velocity;
    for (OrientationIter oit; oit; ++oit) {
        Orientation ori = oit();
        int dir = ori.coordDir();
        Orientation::Side side = ori.faceDir();
        auto const bct = m_bc_type[ori];
        if (bct == BC::no_slip_wall) {
            if (side == Orientation::low) {
                bc_type[dir][0] = 2;
                bc_velocity[dir][0] = m_bc_velocity[ori][0];
                bc_velocity[dir][1] = m_bc_velocity[ori][1];
#if (AMREX_SPACEDIM == 3)
                bc_velocity[dir][2] = m_bc_velocity[ori][2];
#endif
            }
            if (side == Orientation::high) {
                bc_type[dir][1] = 2;
                bc_velocity[dir][AMREX_SPACEDIM] = m_bc_velocity[ori][0];
                bc_velocity[dir][AMREX_SPACEDIM + 1] = m_bc_velocity[ori][1];
#if (AMREX_SPACEDIM == 3)
                bc_velocity[dir][AMREX_SPACEDIM + 2] = m_bc_velocity[ori][2];
#endif
            }
        }
        else if (bct == BC::slip_wall) {
            if (side == Orientation::low) {
                bc_type[dir][0] = 1;
                bc_velocity[dir][0] = m_bc_velocity[ori][0];
                bc_velocity[dir][1] = m_bc_velocity[ori][1];
#if (AMREX_SPACEDIM == 3)
                bc_velocity[dir][2] = m_bc_velocity[ori][2];
#endif
            }
            if (side == Orientation::high) {
                bc_type[dir][1] = 1;
                bc_velocity[dir][AMREX_SPACEDIM] = m_bc_velocity[ori][0];
                bc_velocity[dir][AMREX_SPACEDIM + 1] = m_bc_velocity[ori][1];
#if (AMREX_SPACEDIM == 3)
                bc_velocity[dir][AMREX_SPACEDIM + 2] = m_bc_velocity[ori][2];
#endif
            }
        }
        else {
            if (side == Orientation::low) {
                bc_type[dir][0] = 0;
                bc_velocity[dir][0] = m_bc_velocity[ori][0];
                bc_velocity[dir][1] = m_bc_velocity[ori][1];
#if (AMREX_SPACEDIM == 3)
                bc_velocity[dir][2] = m_bc_velocity[ori][2];
#endif
            }
            if (side == Orientation::high) {
                bc_type[dir][1] = 0;
                bc_velocity[dir][AMREX_SPACEDIM] = m_bc_velocity[ori][0];
                bc_velocity[dir][AMREX_SPACEDIM + 1] = m_bc_velocity[ori][1];
#if (AMREX_SPACEDIM == 3)
                bc_velocity[dir][AMREX_SPACEDIM + 2] = m_bc_velocity[ori][2];
#endif
            }
        }
    }
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(sr_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& bx = mfi.tilebox();
        Array4<Real> const& sr_arr = sr_mf.array(mfi);
        Array4<Real const> const& vel_arr = vel->const_array(mfi);
#ifdef AMREX_USE_EB
        auto const& flag_fab = flags[mfi];
        auto typ = flag_fab.getType(bx);
        if (typ == FabType::covered)
        {
            amrex::Abort("MPMD-based vel_eta is not implemented for EB\n");
        }
        else if (typ == FabType::singlevalued)
        {
            amrex::Abort("MPMD-based vel_eta is not implemented for EB\n");
        }
        else
#endif
        {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real sr = incflo_strainrate_nd(i,j,k,AMREX_D_DECL(idx,idy,idz),
                                               vel_arr,dlo,dhi,bc_type,bc_velocity);
                sr_arr(i,j,k) = sr;
            });
        }
    }

    MultiFab vel_copy(*vel,amrex::make_alias,0,3);
    // Print velocity to check
    std::string plotfilename = "vel_mf_" +
                std::to_string(amrex::ParallelDescriptor::MyProc());

    std::ofstream ofs(plotfilename, std::ofstream::out);
    for (MFIter mfi(vel_copy); mfi.isValid(); ++mfi) {
        ofs<<std::setprecision(16)<< (vel_copy[mfi])<<std::endl;
    }
    ofs.close();

    plotfilename = "sr_mf_" +
                std::to_string(amrex::ParallelDescriptor::MyProc());

    ofs = std::ofstream(plotfilename, std::ofstream::out);
    for (MFIter mfi(sr_mf); mfi.isValid(); ++mfi) {
        ofs<<std::setprecision(16)<< (sr_mf[mfi])<<std::endl;
    }
    ofs.close();

    //auto mask = amrex::OwnerMask(*vel_eta, lev_geom.periodicity());
    //sr_mf.OverrideSync(*mask,lev_geom.periodicity());
    // Copier send of sr_mf and Copier recv of *vel_eta
    amrex::Print() << "Preparing to send strainrate to python\n";
    mpmd_copiers_send_lev(sr_mf,0,1,lev);
    amrex::Print() << "Successfuly sent the strainrate\n";
    mpmd_copiers_recv_lev(*vel_eta,0,1,lev);
    amrex::Print() << "Successfully received the viscosity\n";
    //vel_eta->OverrideSync(*mask,lev_geom.periodicity());
}
#endif

void incflo::compute_tracer_diff_coeff (Vector<MultiFab*> const& tra_eta, int nghost)
{
    for (auto *mf : tra_eta) {
        for (int n = 0; n < m_ntrac; ++n) {
            mf->setVal(m_mu_s[n], n, 1, nghost);
        }
    }
}
