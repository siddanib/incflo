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

bool incflo::has_coulomb_wall () const noexcept
{
    for (OrientationIter oit; oit.isValid(); ++oit) {
        if (m_bc_type[oit()] == BC::coulomb_wall) { return true; }
    }
    return false;
}

// Cell-centered hydrostatic pressure used by the Coulomb friction law. The
// caller passes the density of the time level it is working on.
MultiFab incflo::make_coulomb_p_hydro (int lev, MultiFab const& rho)
{
    MultiFab p_hydro;
    if (!has_coulomb_wall()) { return p_hydro; }
    p_hydro.define(rho.boxArray(), rho.DistributionMap(), 1, 0, MFInfo(), rho.Factory());
    compute_cc_hydrostatic_pressure_at_level(lev, &p_hydro, &rho,
                                             m_mu_p_surf_second, geom[lev], 0);
    return p_hydro;
}

// Fill ghost cells of vel with Coulomb friction flux values (∂u_T/∂n) for
// inhomogNeumann diffusion BC. Called immediately before setLevelBC for the
// diffusion tensor solve when coulomb_wall faces are present.
//
// Ghost cell convention (matches AMReX mllinop_apply_innu kernels):
//   lo face: bcval = +mu_f * max(p_h,0) / eta_wall * (u_T / |u_T|)
//   hi face: bcval = -mu_f * max(p_h,0) / eta_wall * (u_T / |u_T|)
// Normal direction ghost holds the Dirichlet value u.n = 0. AMReX setLevelBC
// interprets Dirichlet ghost data as living on the domain face (not as a
// reflected cell value), so the ghost must be 0, not -interior.
//
// p_hydro is the cell-centered hydrostatic pressure of the time level the
// caller is working on (see make_coulomb_p_hydro); only the hydrostatic part
// is used, the dynamic pressure is deliberately excluded.
void incflo::fill_coulomb_flux_ghost_cells (int lev,
                                             MultiFab& vel,
                                             MultiFab const& cc_eta,
                                             MultiFab const* p_hydro)
{
    if (!has_coulomb_wall()) return;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(p_hydro != nullptr,
        "fill_coulomb_flux_ghost_cells: hydrostatic pressure is required at Coulomb walls");

    Array<MultiFab,AMREX_SPACEDIM> eta_face = average_velocity_eta_to_faces(lev, cc_eta);

    const Box& domain = Geom(lev).Domain();
    const Real eps2   = m_coulomb_eps * m_coulomb_eps;

    GpuArray<BC,   AMREX_SPACEDIM*2> bc_type = m_bc_type;
    GpuArray<Real, AMREX_SPACEDIM*2> bc_mu   = m_bc_mu_coulomb;

    // No tiling: the kernels below work on the ghost layer of the whole valid box
    for (MFIter mfi(vel); mfi.isValid(); ++mfi)
    {
        Box const& vbx = mfi.validbox();
        Array4<Real>       const& vel_a  = vel.array(mfi);
        Array4<Real const> const& phyd_a = p_hydro->const_array(mfi);

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
                            vel_a(i,j,k,0) = (dir!=0) ?  coeff*vel_a(ii,jj,kk,0) : Real(0.);,
                            vel_a(i,j,k,1) = (dir!=1) ?  coeff*vel_a(ii,jj,kk,1) : Real(0.);,
                            vel_a(i,j,k,2) = (dir!=2) ?  coeff*vel_a(ii,jj,kk,2) : Real(0.););
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
                            vel_a(i,j,k,0) = (dir!=0) ? coeff*vel_a(ii,jj,kk,0) : Real(0.);,
                            vel_a(i,j,k,1) = (dir!=1) ? coeff*vel_a(ii,jj,kk,1) : Real(0.);,
                            vel_a(i,j,k,2) = (dir!=2) ? coeff*vel_a(ii,jj,kk,2) : Real(0.););
                    });
                }
            }
        }
    }
}

// Make the velocity gradients used by the higher-order (HO) granular stress
// consistent with the Coulomb friction BC. MLTensorOp::compVelGrad treats the
// tangential components at a Coulomb wall as homogeneous Neumann (it ignores
// inhomogeneous Neumann values), so:
//   1. on the wall face, d(u_t)/dx_n is replaced by the Coulomb gradient g;
//   2. on the faces of the wall-adjacent cell row that are normal to another
//      direction e, the centered tangential derivative d(u_t)/dx_n used ghost
//      = interior; the consistent ghost (u_0 -/+ dx_n g) adds +g/2, averaged
//      over the two cells sharing the face. On a domain face in e the AMReX
//      stencil uses the in-domain cell row only, so the single-cell g is used,
//      and nothing is added if u_t is Dirichlet there (e.g. no-slip wall).
// The stencils assumed here are those of MLTensorOp (all-regular geometry).
void incflo::apply_coulomb_wall_vel_grad (int lev,
                                          Array<MultiFab*,AMREX_SPACEDIM> const& gradVel,
                                          MultiFab const& vel,
                                          MultiFab const& cc_eta,
                                          MultiFab const& p_hydro)
{
    if (!has_coulomb_wall()) { return; }
#ifdef AMREX_USE_EB
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(EBFactory(lev).isAllRegular(),
        "Coulomb wall correction of HO velocity gradients assumes an all-regular geometry");
#endif

    constexpr int S = AMREX_SPACEDIM;

    // Coulomb wall gradients (as d/dx_n, signed) in the ghost cells of a copy
    MultiFab vel_g(vel.boxArray(), vel.DistributionMap(), S, 1, MFInfo(), vel.Factory());
    vel_g.setVal(Real(0.));
    MultiFab::Copy(vel_g, vel, 0, 0, S, 0);
    fill_coulomb_flux_ghost_cells(lev, vel_g, cc_eta, &p_hydro);

    // Store g in the wall-adjacent cell row, comp = ori*S + t, so that a
    // FillBoundary makes neighbor (and periodic) values available
    MultiFab gw(vel.boxArray(), vel.DistributionMap(), 2*S*S, 1, MFInfo(), vel.Factory());
    gw.setVal(Real(0.));

    const Box& domain = Geom(lev).Domain();
    GpuArray<BC, 2*S> bc_type = m_bc_type;

    for (MFIter mfi(gw); mfi.isValid(); ++mfi)
    {
        Box const& vbx = mfi.validbox();
        Array4<Real const> const& vg = vel_g.const_array(mfi);
        Array4<Real      > const& g  = gw.array(mfi);
        for (OrientationIter oit; oit.isValid(); ++oit) {
            const Orientation ori = oit();
            const int d = ori.coordDir();
            const bool is_lo = ori.isLow();
            if (bc_type[ori] != BC::coulomb_wall) { continue; }
            if ( is_lo && vbx.smallEnd(d) != domain.smallEnd(d)) { continue; }
            if (!is_lo && vbx.bigEnd(d)   != domain.bigEnd(d))   { continue; }
            Box crow = vbx;
            if (is_lo) { crow.setBig(d, vbx.smallEnd(d)); }
            else       { crow.setSmall(d, vbx.bigEnd(d)); }
            const int s  = is_lo ? -1 : 1;
            const int oc = int(ori)*S;
            ParallelFor(crow, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                int ig = i + (d==0 ? s : 0);
                int jg = j + (d==1 ? s : 0);
                int kg = k + (d==2 ? s : 0);
                for (int t = 0; t < S; ++t) {
                    if (t != d) { g(i,j,k,oc+t) = vg(ig,jg,kg,t); }
                }
            });
        }
    }
    gw.FillBoundary(Geom(lev).periodicity());

    // Diffusion BC types: is component t Dirichlet on face orientation o?
    GpuArray<int, 2*S*S> is_dirichlet{};
    {
        auto const bclo = get_diffuse_tensor_bc(Orientation::low);
        auto const bchi = get_diffuse_tensor_bc(Orientation::high);
        for (int e = 0; e < S; ++e) {
            for (int t = 0; t < S; ++t) {
                is_dirichlet[int(Orientation(e,Orientation::low ))*S + t] =
                    (bclo[t][e] == LinOpBCType::Dirichlet) ? 1 : 0;
                is_dirichlet[int(Orientation(e,Orientation::high))*S + t] =
                    (bchi[t][e] == LinOpBCType::Dirichlet) ? 1 : 0;
            }
        }
    }
    GpuArray<int, S> periodic{};
    for (int e = 0; e < S; ++e) { periodic[e] = Geom(lev).isPeriodic(e) ? 1 : 0; }
    const auto dlo = amrex::lbound(domain);
    const auto dhi = amrex::ubound(domain);

    for (MFIter mfi(gw); mfi.isValid(); ++mfi)
    {
        Box const& vbx = mfi.validbox();
        Array4<Real const> const& g = gw.const_array(mfi);
        for (OrientationIter oit; oit.isValid(); ++oit) {
            const Orientation ori = oit();
            const int d = ori.coordDir();
            const bool is_lo = ori.isLow();
            if (bc_type[ori] != BC::coulomb_wall) { continue; }
            if ( is_lo && vbx.smallEnd(d) != domain.smallEnd(d)) { continue; }
            if (!is_lo && vbx.bigEnd(d)   != domain.bigEnd(d))   { continue; }
            Box crow = vbx;
            if (is_lo) { crow.setBig(d, vbx.smallEnd(d)); }
            else       { crow.setSmall(d, vbx.bigEnd(d)); }
            const int oc = int(ori)*S;

            // 1. Wall face: d(u_t)/dx_d = g
            {
                Array4<Real> const& gd = gradVel[d]->array(mfi);
                const int fs = is_lo ? 0 : 1;   // face index = cell index (+1 on hi side)
                ParallelFor(crow, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    int fi = i + (d==0 ? fs : 0);
                    int fj = j + (d==1 ? fs : 0);
                    int fk = k + (d==2 ? fs : 0);
                    for (int t = 0; t < S; ++t) {
                        if (t != d) { gd(fi,fj,fk,t+S*d) = g(i,j,k,oc+t); }
                    }
                });
            }

            // 2. Faces normal to e != d in the wall-adjacent cell row
            //
            // Example: ylo wall (d=y, wall cell row j=0), z-face (e=z) at k-1/2.
            // MLTensorOp (mltensor_dy_on_zface) computes, for interior faces,
            //   du/dy = [u(1,k) + u(1,k-1) - u(-1,k) - u(-1,k-1)] / (4 dy)
            // with ghost u(-1,.) = u(0,.) (homogeneous Neumann). The ghost
            // consistent with the Coulomb gradient g = du/dy at the wall is
            //   lo wall: u(-1) = u(0) - dy*g,   hi wall: u(N) = u(N-1) + dy*g,
            // and in both cases the stencil increases by
            //   dy*(g(k) + g(k-1)) / (4 dy) = 0.5 * [g(k) + g(k-1)]/2.
            // On a non-periodic domain face in e (e.g. the bed, k = dlo.z) the
            // Neumann stencil uses the in-domain row only,
            //   du/dy = [u(1,k) - u(-1,k)] / (2 dy)  ->  correction 0.5*g(k),
            // and for a Dirichlet component there (e.g. no-slip, or the normal
            // velocity) AMReX differences the boundary values instead, which
            // is already consistent, so no correction is added.
            for (int e = 0; e < S; ++e) {
                if (e == d) { continue; }
                Array4<Real> const& ge = gradVel[e]->array(mfi);
                Box const fbx = amrex::surroundingNodes(crow, e);
                const int elo = (e==0) ? dlo.x : ((e==1) ? dlo.y : dlo.z);
                const int ehi = (e==0) ? dhi.x : ((e==1) ? dhi.y : dhi.z);
                const int per = periodic[e];
                const int olo = int(Orientation(e,Orientation::low ))*S;
                const int ohi = int(Orientation(e,Orientation::high))*S;
                auto const dir_bc = is_dirichlet;
                ParallelFor(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    const int f  = (e==0) ? i : ((e==1) ? j : k);
                    // cell on the low side of the face along e
                    int im = i - (e==0 ? 1 : 0);
                    int jm = j - (e==1 ? 1 : 0);
                    int km = k - (e==2 ? 1 : 0);
                    for (int t = 0; t < S; ++t) {
                        if (t == d) { continue; }
                        Real corr;
                        if (!per && f == elo) {
                            corr = dir_bc[olo+t] ? Real(0.) : Real(0.5)*g(i,j,k,oc+t);
                        } else if (!per && f == ehi+1) {
                            corr = dir_bc[ohi+t] ? Real(0.) : Real(0.5)*g(im,jm,km,oc+t);
                        } else {
                            corr = Real(0.25)*(g(im,jm,km,oc+t) + g(i,j,k,oc+t));
                        }
                        ge(i,j,k,t+S*d) += corr;
                    }
                });
            }
        }
    }
}

// Higher-order (HO) granular stress fluxes at Coulomb wall faces:
//   - tangential components are set to zero, so the wall traction is exactly
//     the Coulomb friction carried by the linear operator;
//   - the normal component is kept (ho_normal = 1) or copied from the first
//     interior face, i.e. zero normal gradient (ho_normal = 0).
void incflo::apply_coulomb_wall_ho_fluxes (int lev,
                                           Array<MultiFab*,AMREX_SPACEDIM> const& fluxes)
{
    if (!has_coulomb_wall()) { return; }

    constexpr int S = AMREX_SPACEDIM;
    const Box& domain = Geom(lev).Domain();

    for (OrientationIter oit; oit.isValid(); ++oit) {
        const Orientation ori = oit();
        if (m_bc_type[ori] != BC::coulomb_wall) { continue; }
        const int d = ori.coordDir();
        const bool is_lo = ori.isLow();
        const bool keep_normal = (m_bc_coulomb_ho_normal[ori] != 0);

        for (MFIter mfi(*fluxes[d]); mfi.isValid(); ++mfi)
        {
            Box const cbx = amrex::enclosedCells(mfi.validbox());
            if ( is_lo && cbx.smallEnd(d) != domain.smallEnd(d)) { continue; }
            if (!is_lo && cbx.bigEnd(d)   != domain.bigEnd(d))   { continue; }
            Box const fbx = is_lo ? amrex::bdryLo(cbx, d) : amrex::bdryHi(cbx, d);
            const int off = is_lo ? 1 : -1;   // first interior face
            Array4<Real> const& fx = fluxes[d]->array(mfi);
            ParallelFor(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                for (int t = 0; t < S; ++t) {
                    if (t != d) { fx(i,j,k,t) = Real(0.); }
                }
                if (!keep_normal) {
                    fx(i,j,k,d) = fx(i + (d==0 ? off : 0),
                                     j + (d==1 ? off : 0),
                                     k + (d==2 ? off : 0), d);
                }
            });
        }
    }
}
