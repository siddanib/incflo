#include <incflo.H>
#include <prob_bc.H>

using namespace amrex;

void incflo::fillphysbc_velocity (int lev, Real time, MultiFab& vel, int ng)
{
    PhysBCFunct<GpuBndryFuncFab<IncfloVelFill> > physbc(geom[lev], get_velocity_bcrec(),
                                                        IncfloVelFill{m_probtype, m_bc_velocity});
    physbc.FillBoundary(vel, 0, AMREX_SPACEDIM, IntVect(ng), time, 0);
}

void incflo::fillphysbc_density (int lev, Real time, MultiFab& density, int ng)
{
    PhysBCFunct<GpuBndryFuncFab<IncfloDenFill> > physbc(geom[lev],
                                                        get_density_bcrec(),
                                                        IncfloDenFill{m_probtype, m_bc_density, m_bc_velocity});
    physbc.FillBoundary(density, 0, 1, IntVect(ng), time, 0);
}

void incflo::fillphysbc_tracer (int lev, Real time, MultiFab& tracer, int ng)
{
    if (m_ntrac > 0) {
        PhysBCFunct<GpuBndryFuncFab<IncfloTracFill> > physbc
            (geom[lev], get_tracer_bcrec(), IncfloTracFill{m_probtype, m_ntrac, m_bc_tracer_d, m_bc_velocity});
        physbc.FillBoundary(tracer, 0, m_ntrac, IntVect(ng), time, 0);
    }
}

void incflo::fillphysbc_temperature (int lev, Real time, MultiFab& temperature, int ng)
{
    if (m_use_temperature)
    {
        PhysBCFunct<GpuBndryFuncFab<IncfloTempFill> > physbc
            (geom[lev], get_temperature_bcrec(),
             IncfloTempFill{m_probtype, m_bc_temperature, m_bc_velocity});
        physbc.FillBoundary(temperature, 0, 1, IntVect(ng), time, 0);
    }
}

// Fill ghost cells of vel with Coulomb friction flux values (∂u_T/∂n) for
// inhomogNeumann diffusion BC. Called immediately before setLevelBC for the
// diffusion tensor solve when coulomb_wall faces are present.
//
// Ghost cell convention (matches AMReX mllinop_apply_innu kernels):
//   lo face: bcval = +mu_f * max(p_h+p_cc,0) / eta_wall * (u_T / |u_T|)
//   hi face: bcval = -mu_f * max(p_h+p_cc,0) / eta_wall * (u_T / |u_T|)
// Normal direction ghost holds Dirichlet value: ghost = -interior (gives 0 at face).
void incflo::fill_coulomb_flux_ghost_cells (int lev,
                                             MultiFab& vel,
                                             MultiFab const& cc_eta)
{
    bool has_coulomb = false;
    for (int dir = 0; dir < AMREX_SPACEDIM && !has_coulomb; ++dir) {
        if (m_bc_type[Orientation(dir,Orientation::low )] == BC::coulomb_wall ||
            m_bc_type[Orientation(dir,Orientation::high)] == BC::coulomb_wall) {
            has_coulomb = true;
        }
    }
    if (!has_coulomb) return;

    MultiFab p_hydro(vel.boxArray(), vel.DistributionMap(), 1, 0,
                     MFInfo(), vel.Factory());
    compute_cc_hydrostatic_pressure_at_level(lev, &p_hydro,
                                             &m_leveldata[lev]->density,
                                             0.0, geom[lev], 0);

    Array<MultiFab,AMREX_SPACEDIM> eta_face = average_velocity_eta_to_faces(lev, cc_eta);

    // Resolve dynamic pressure to CC regardless of projection type.
    // p_cc is only allocated when m_use_cc_proj = true; the default nodal
    // projection path allocates p_nd instead, so we average it here.
    MultiFab p_dyn_local;
    MultiFab const* p_dyn = nullptr;
    if (m_use_cc_proj) {
        p_dyn = &m_leveldata[lev]->p_cc;
    } else {
        p_dyn_local.define(vel.boxArray(), vel.DistributionMap(), 1, 0,
                           MFInfo(), vel.Factory());
        amrex::average_node_to_cellcenter(p_dyn_local, 0,
                                          m_leveldata[lev]->p_nd, 0, 1);
        p_dyn = &p_dyn_local;
    }

    const Box& domain = Geom(lev).Domain();
    const Real eps2   = m_coulomb_eps * m_coulomb_eps;

    GpuArray<BC,   AMREX_SPACEDIM*2> bc_type = m_bc_type;
    GpuArray<Real, AMREX_SPACEDIM*2> bc_mu   = m_bc_mu_coulomb;

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(vel, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& vbx = mfi.validbox();
        Array4<Real>       const& vel_a  = vel.array(mfi);
        Array4<Real const> const& pcc_a  = p_dyn->const_array(mfi);
        Array4<Real const> const& phyd_a = p_hydro.const_array(mfi);

        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            Array4<Real const> const& eta_a = eta_face[dir].const_array(mfi);

            // --- lo face ---
            {
                Orientation ori(dir, Orientation::low);
                if (bc_type[ori] == BC::coulomb_wall &&
                    vbx.smallEnd(dir) == domain.smallEnd(dir))
                {
                    Real mu_f = bc_mu[ori];
                    Box gbx = amrex::adjCellLo(vbx, dir, 1);
                    ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        int ii = i + (dir==0 ? 1 : 0);
                        int jj = j + (dir==1 ? 1 : 0);
                        int kk = k + (dir==2 ? 1 : 0);

                        // Considering only hydrostatic pressure
                        // Real p_wall = amrex::max(phyd_a(ii,jj,kk) + pcc_a(ii,jj,kk), Real(0.));
                        Real p_wall = amrex::max(phyd_a(ii,jj,kk), Real(0.));

                        Real uT2 = Real(0.);
                        AMREX_D_TERM(if (dir!=0) uT2 += vel_a(ii,jj,kk,0)*vel_a(ii,jj,kk,0);,
                                     if (dir!=1) uT2 += vel_a(ii,jj,kk,1)*vel_a(ii,jj,kk,1);,
                                     if (dir!=2) uT2 += vel_a(ii,jj,kk,2)*vel_a(ii,jj,kk,2););

                        // lo wall: face index == interior cell index
                        Real eta_w = eta_a(ii, jj, kk);
                        Real coeff = (eta_w > Real(0.))
                                     ? mu_f * p_wall / eta_w / std::sqrt(uT2 + eps2)
                                     : Real(0.);

                        AMREX_D_TERM(
                            vel_a(i,j,k,0) = (dir!=0) ?  coeff*vel_a(ii,jj,kk,0) : -vel_a(ii,jj,kk,0);,
                            vel_a(i,j,k,1) = (dir!=1) ?  coeff*vel_a(ii,jj,kk,1) : -vel_a(ii,jj,kk,1);,
                            vel_a(i,j,k,2) = (dir!=2) ?  coeff*vel_a(ii,jj,kk,2) : -vel_a(ii,jj,kk,2););
                    });
                }
            }

            // --- hi face ---
            {
                Orientation ori(dir, Orientation::high);
                if (bc_type[ori] == BC::coulomb_wall &&
                    vbx.bigEnd(dir) == domain.bigEnd(dir))
                {
                    Real mu_f = bc_mu[ori];
                    Box gbx = amrex::adjCellHi(vbx, dir, 1);
                    ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        int ii = i - (dir==0 ? 1 : 0);
                        int jj = j - (dir==1 ? 1 : 0);
                        int kk = k - (dir==2 ? 1 : 0);

                        // Considering only hydrostatic pressure
                        // Real p_wall = amrex::max(phyd_a(ii,jj,kk) + pcc_a(ii,jj,kk), Real(0.));
                        Real p_wall = amrex::max(phyd_a(ii,jj,kk), Real(0.));

                        Real uT2 = Real(0.);
                        AMREX_D_TERM(if (dir!=0) uT2 += vel_a(ii,jj,kk,0)*vel_a(ii,jj,kk,0);,
                                     if (dir!=1) uT2 += vel_a(ii,jj,kk,1)*vel_a(ii,jj,kk,1);,
                                     if (dir!=2) uT2 += vel_a(ii,jj,kk,2)*vel_a(ii,jj,kk,2););

                        // hi wall: face index == ghost cell index
                        Real eta_w = eta_a(i, j, k);
                        Real coeff = (eta_w > Real(0.))
                                     ? -mu_f * p_wall / eta_w / std::sqrt(uT2 + eps2)
                                     : Real(0.);

                        AMREX_D_TERM(
                            vel_a(i,j,k,0) = (dir!=0) ? coeff*vel_a(ii,jj,kk,0) : -vel_a(ii,jj,kk,0);,
                            vel_a(i,j,k,1) = (dir!=1) ? coeff*vel_a(ii,jj,kk,1) : -vel_a(ii,jj,kk,1);,
                            vel_a(i,j,k,2) = (dir!=2) ? coeff*vel_a(ii,jj,kk,2) : -vel_a(ii,jj,kk,2););
                    });
                }
            }
        }
    }
}
