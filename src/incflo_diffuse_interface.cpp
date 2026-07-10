#include <incflo.H>
#include <prob_bc.H>

#ifdef AMREX_USE_EB
#include <AMReX_EBCellFlag.H>
#include <AMReX_EBMultiFabUtil.H>
#endif
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PhysBCFunct.H>

#include <cmath>
#include <limits>

using namespace amrex;

namespace {


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real avg_cc_to_face_masked (IntVect ijk_hi, int dir,
                            Array4<int const> const& mask,
                            Array4<Real const> const& cc,
                            int lo_lim, int hi_lim,
                            bool periodic, int comp = 0) noexcept
{
    IntVect ijk_lo = ijk_hi;
    ijk_lo[dir] -= 1;

    if (!periodic && ijk_hi[dir] == lo_lim) {
        return mask(ijk_hi) ? cc(ijk_hi, comp) : Real(0.0);
    }
    if (!periodic && ijk_hi[dir] == hi_lim + 1) {
        return mask(ijk_lo) ? cc(ijk_lo, comp) : Real(0.0);
    }

    int const mask_lo = mask(ijk_lo);
    int const mask_hi = mask(ijk_hi);

    if (mask_lo && mask_hi) {
        return Real(0.5) * (cc(ijk_lo, comp) + cc(ijk_hi, comp));
    }
    if (mask_lo) { return cc(ijk_lo, comp); }
    if (mask_hi) { return cc(ijk_hi, comp); }
    return Real(0.0);
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real cc_grad_dir_with_face_ghost (int i, int j, int k, int dir, int comp,
                                  Array4<Real const> const& mf,
                                  Real dx_dir, Dim3 const& dlo,
                                  Dim3 const& dhi, bool periodic) noexcept
{
    int const di = (dir == 0) ? 1 : 0;
    int const dj = (dir == 1) ? 1 : 0;
    int const dk = (dir == 2) ? 1 : 0;

    int const iv_dir = (dir == 0) ? i : (dir == 1) ? j : k;
    int const lo_lim = (dir == 0) ? dlo.x : (dir == 1) ? dlo.y : dlo.z;
    int const hi_lim = (dir == 0) ? dhi.x : (dir == 1) ? dhi.y : dhi.z;

    Real xm = -dx_dir;
    Real xp =  dx_dir;
    if (!periodic && iv_dir == lo_lim) { xm *= Real(0.5); }
    if (!periodic && iv_dir == hi_lim) { xp *= Real(0.5); }

    Real const fm = mf(i-di,j-dj,k-dk,comp);
    Real const fc = mf(i   ,j   ,k   ,comp);
    Real const fp = mf(i+di,j+dj,k+dk,comp);

    return -(fm * (xp / (xm * (xm - xp)))
           + fc * ((xm + xp) / (xm * xp))
           + fp * (xm / ((xp - xm) * xp)));
}



#ifdef AMREX_USE_EB
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real eb_cell_grad_dir (int i, int j, int k, int dir, int comp,
                       Array4<Real const> const& mf,
                       Array4<EBCellFlag const> const& flag,
                       Real dx_dir) noexcept
{
    if (flag(i,j,k).isCovered()) { return Real(0.0); }

    constexpr Real c0 = Real(-1.5);
    constexpr Real c1 = Real( 2.0);
    constexpr Real c2 = Real(-0.5);

    Real const odx = Real(1.0) / dx_dir;
    int const di = (dir == 0) ? 1 : 0;
    int const dj = (dir == 1) ? 1 : 0;
    int const dk = (dir == 2) ? 1 : 0;

    if (!flag(i,j,k).isConnected( di, dj, dk)) {
        if (!flag(i,j,k).isConnected(-di,-dj,-dk)) {
            return Real(0.0);
        } else if (!flag(i-di,j-dj,k-dk).isConnected(-di,-dj,-dk)) {
            return (mf(i,j,k,comp) - mf(i-di,j-dj,k-dk,comp)) * odx;
        } else {
            return -(c0 * mf(i     ,j     ,k     ,comp)
                   + c1 * mf(i-di  ,j-dj  ,k-dk  ,comp)
                   + c2 * mf(i-2*di,j-2*dj,k-2*dk,comp)) * odx;
        }
    } else if (!flag(i,j,k).isConnected(-di,-dj,-dk)) {
        if (!flag(i,j,k).isConnected(di,dj,dk)) {
            return Real(0.0);
        } else if (!flag(i+di,j+dj,k+dk).isConnected(di,dj,dk)) {
            return (mf(i+di,j+dj,k+dk,comp) - mf(i,j,k,comp)) * odx;
        } else {
            return (c0 * mf(i     ,j     ,k     ,comp)
                  + c1 * mf(i+di  ,j+dj  ,k+dk  ,comp)
                  + c2 * mf(i+2*di,j+2*dj,k+2*dk,comp)) * odx;
        }
    } else {
        return Real(0.5) * (mf(i+di,j+dj,k+dk,comp)
                          - mf(i-di,j-dj,k-dk,comp)) * odx;
    }
}
#endif

Real interface_epsilon (Geometry const& geom, bool fixed_epsilon, Real epsilon, Real epsilon_star)
{
    auto const dx = geom.CellSizeArray();
    Real min_dx = dx[0];
    for (int d = 1; d < AMREX_SPACEDIM; ++d) {
        min_dx = amrex::min(min_dx, dx[d]);
    }
    return fixed_epsilon ? epsilon : epsilon_star * min_dx;
}

Real interface_gamma (int finest_level, bool fixed_gamma, Real gamma, Real gamma_star,
                      Vector<MultiFab const*> const& vel)
{
    if (fixed_gamma) { return gamma; }

    Real umax = Real(0.0);
    for (int lev = 0; lev <= finest_level; ++lev) {
        for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
            umax = amrex::max(umax, vel[lev]->norm0(comp, 0));
        }
    }
    ParallelDescriptor::ReduceRealMax(umax);
    return gamma_star * umax;
}

iMultiFab make_interface_mask (MultiFab const& phi, Real phitol)
{
    iMultiFab mask(phi.boxArray(), phi.DistributionMap(), 1, phi.nGrow(), MFInfo());
    mask.setVal(0);

    Real const tol_lo = phitol;
    Real const tol_hi = Real(1.0) - tol_lo;
    Real const buf_lo = Real(0.1) * tol_lo;
    Real const buf_hi = Real(1.0) - buf_lo;

#ifdef AMREX_USE_EB
    auto const& ebfactory = dynamic_cast<EBFArrayBoxFactory const&>(phi.Factory());
    auto const& flags = ebfactory.getMultiEBCellFlagFab();
#endif

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(phi,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        Box const& bx = mfi.growntilebox(phi.nGrow());
        Array4<Real const> const& p = phi.const_array(mfi);
        Array4<int> const& m = mask.array(mfi);
#ifdef AMREX_USE_EB
        Array4<EBCellFlag const> const& flag = flags.const_array(mfi);
#endif
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
#ifdef AMREX_USE_EB
            if (flag(i,j,k).isCovered()) { return; }
#endif
            Real const v = p(i,j,k,0);
            if ((v > tol_lo && v < tol_hi) ||
                (v > buf_lo && v <= tol_lo) ||
                (v >= tol_hi && v < buf_hi)) {
                m(i,j,k) = 1;
            }
        });
    }
    return mask;
}

void copy_tracer0_to_scratch (Vector<MultiFab>& dst, Vector<MultiFab const*> const& tracer)
{
    dst.clear();
    dst.reserve(tracer.size());
    for (auto const* mf : tracer) {
        dst.emplace_back(mf->boxArray(), mf->DistributionMap(), 1, mf->nGrow(),
                         MFInfo(), mf->Factory());
        MultiFab::Copy(dst.back(), *mf, 0, 0, 1, mf->nGrow());
    }
}

void alias_tracer0 (Vector<MultiFab>& dst, Vector<MultiFab*> const& tracer)
{
    dst.clear();
    dst.reserve(tracer.size());
    for (auto* mf : tracer) {
        dst.emplace_back(*mf, make_alias, 0, 1);
    }
}

}

void incflo::compute_interface_terms (StepType step_type)
{
    if (!m_diffuse_interface) { return; }

    Vector<MultiFab const*> tracer = (step_type == StepType::Predictor)
        ? get_tracer_old_const() : get_tracer_new_const();
    Vector<MultiFab const*> vel = (step_type == StepType::Predictor)
        ? get_velocity_old_const() : get_velocity_new_const();

    Real const Gamma = interface_gamma(finest_level, m_interface_fixed_gamma, m_interface_gamma, m_interface_gamma_star, vel);
    Real const sigma = m_interface_surface_tension;
    Real const smallnum = m_interface_varepsilon;
    Real const smalltol = m_interface_smalltol;
    Real const phitol = m_interface_phitol;
    Real const rhod = m_ro_0_second - m_ro_0;

    Vector<MultiFab> phi;
    copy_tracer0_to_scratch(phi, tracer);

    Vector<MultiFab> psi(finest_level+1);
    Vector<MultiFab> ncc(finest_level+1);
    Vector<MultiFab> dphi(finest_level+1);
    Vector<Array<MultiFab,AMREX_SPACEDIM>> nfc(finest_level+1);
    Vector<Array<MultiFab,AMREX_SPACEDIM>> areg(finest_level+1);
    Vector<Array<MultiFab,AMREX_SPACEDIM>> vel_fc(finest_level+1);
    Vector<Array<MultiFab,AMREX_SPACEDIM>> fxu_fc(finest_level+1);

    for (int lev = 0; lev <= finest_level; ++lev) {
        auto const& ba = grids[lev];
        auto const& dm = dmap[lev];
        auto const& fact = Factory(lev);

        psi[lev].define(ba, dm, 1, 2, MFInfo(), fact);
        ncc[lev].define(ba, dm, AMREX_SPACEDIM, 1, MFInfo(), fact);
        dphi[lev].define(ba, dm, AMREX_SPACEDIM, 1, MFInfo(), fact);
        psi[lev].setVal(0.0);
        ncc[lev].setVal(0.0);
        dphi[lev].setVal(0.0);

        MultiFab& divA = (step_type == StepType::Predictor)
            ? m_leveldata[lev]->interface_divA_o : m_leveldata[lev]->interface_divA;
        divA.setVal(0.0);
        m_leveldata[lev]->interface_force.setVal(0.0);

        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            BoxArray fba = ba;
            fba.surroundingNodes(dir);
            nfc[lev][dir].define(fba, dm, 1, 1, MFInfo(), fact);
            areg[lev][dir].define(fba, dm, 2, 1, MFInfo(), fact);
            vel_fc[lev][dir].define(fba, dm, AMREX_SPACEDIM, 1, MFInfo(), fact);
            fxu_fc[lev][dir].define(fba, dm, AMREX_SPACEDIM, 1, MFInfo(), fact);
            nfc[lev][dir].setVal(0.0);
            areg[lev][dir].setVal(0.0);
            vel_fc[lev][dir].setVal(0.0);
            fxu_fc[lev][dir].setVal(0.0);
        }
    }

    for (int lev = 0; lev <= finest_level; ++lev) {
        Real const epsilon = interface_epsilon(geom[lev], m_interface_fixed_epsilon, m_interface_epsilon, m_interface_epsilon_star);
        auto const dxi = geom[lev].InvCellSizeArray();
        auto const dx = geom[lev].CellSizeArray();
        auto const domain = geom[lev].Domain();
        Dim3 const dlo = lbound(domain);
        Dim3 const dhi = ubound(domain);
        GpuArray<bool, AMREX_SPACEDIM> is_periodic;
        is_periodic[0] = geom[lev].isPeriodic(0);
        is_periodic[1] = geom[lev].isPeriodic(1);
#if (AMREX_SPACEDIM == 3)
        is_periodic[2] = geom[lev].isPeriodic(2);
#endif

        if (m_ntrac > 0) {
            Vector<BCRec> phi_bcrec{m_bcrec_tracer[0]};
            PhysBCFunct<GpuBndryFuncFab<IncfloTracFill> > physbc
                (geom[lev], phi_bcrec,
                 IncfloTracFill{m_probtype, 1, m_bc_tracer_d, m_bc_velocity});
            Real const fill_time = (step_type == StepType::Predictor) ? m_t_old[lev] : m_t_new[lev];
            physbc.FillBoundary(phi[lev], 0, 1, IntVect(phi[lev].nGrow()), fill_time, 0);
        }
        phi[lev].FillBoundary(geom[lev].periodicity());
        iMultiFab mask = make_interface_mask(phi[lev], phitol);

#ifdef AMREX_USE_EB
        auto const& ebfactory_sten = dynamic_cast<EBFArrayBoxFactory const&>(Factory(lev));
        auto const& flags_sten = ebfactory_sten.getMultiEBCellFlagFab();
#endif

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(phi[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Box const& gbx = mfi.growntilebox(2);
            Array4<Real> const& ps = psi[lev].array(mfi);
            Array4<Real const> const& p = phi[lev].const_array(mfi);
#ifdef AMREX_USE_EB
            EBCellFlagFab const& flagfab = flags_sten[mfi];
            FabType const fabtype = flagfab.getType(gbx);
            if (fabtype == FabType::covered) {
                psi[lev][mfi].setVal(Real(0.0), gbx, 0, 1);
                continue;
            }
            if (fabtype == FabType::singlevalued) {
                Array4<EBCellFlag const> const& flag = flagfab.const_array();
                ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (!flag(i,j,k).isCovered()) {
                        Real const pc = amrex::Clamp(p(i,j,k,0), Real(0.0), Real(1.0));
                        ps(i,j,k) = epsilon * std::log((pc + smallnum) / (Real(1.0) - pc + smallnum));
                    } else {
                        ps(i,j,k) = Real(0.0);
                    }
                });
            } else
#endif
            {
                ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const pc = amrex::Clamp(p(i,j,k,0), Real(0.0), Real(1.0));
                    ps(i,j,k) = epsilon * std::log((pc + smallnum) / (Real(1.0) - pc + smallnum));
                });
            }
        }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(phi[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Box const& bx = mfi.growntilebox(1);
            Array4<Real const> const& ps = psi[lev].const_array(mfi);
            Array4<Real const> const& p = phi[lev].const_array(mfi);
            Array4<int const> const& m = mask.const_array(mfi);
            Array4<Real> const& n = ncc[lev].array(mfi);
            Array4<Real> const& gp = dphi[lev].array(mfi);
#ifdef AMREX_USE_EB
            EBCellFlagFab const& flagfab = flags_sten[mfi];
            FabType const fabtype = flagfab.getType(bx);
            if (fabtype == FabType::covered) {
                ncc[lev][mfi].setVal(Real(0.0), bx, 0, AMREX_SPACEDIM);
                dphi[lev][mfi].setVal(Real(0.0), bx, 0, AMREX_SPACEDIM);
                continue;
            }
            if (fabtype == FabType::singlevalued) {
                Array4<EBCellFlag const> const& flag = flagfab.const_array();
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const psx = eb_cell_grad_dir(i,j,k,0,0,ps,flag,dx[0]);
                    Real const psy = eb_cell_grad_dir(i,j,k,1,0,ps,flag,dx[1]);
#if (AMREX_SPACEDIM == 3)
                    Real const psz = eb_cell_grad_dir(i,j,k,2,0,ps,flag,dx[2]);
#else
                    Real const psz = Real(0.0);
#endif
                    Real const mag = std::sqrt(AMREX_D_TERM(psx*psx, + psy*psy, + psz*psz));
                    Real const invmag = (m(i,j,k) && mag > smalltol) ? Real(1.0) / mag : Real(0.0);
                    n(i,j,k,0) = psx * invmag;
                    n(i,j,k,1) = psy * invmag;
#if (AMREX_SPACEDIM == 3)
                    n(i,j,k,2) = psz * invmag;
#endif

                    gp(i,j,k,0) = eb_cell_grad_dir(i,j,k,0,0,p,flag,dx[0]);
                    gp(i,j,k,1) = eb_cell_grad_dir(i,j,k,1,0,p,flag,dx[1]);
#if (AMREX_SPACEDIM == 3)
                    gp(i,j,k,2) = eb_cell_grad_dir(i,j,k,2,0,p,flag,dx[2]);
#endif
                });
            } else
#endif
            {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const psx = cc_grad_dir_with_face_ghost(i,j,k,0,0,ps,dx[0],dlo,dhi,is_periodic[0]);
                    Real const psy = cc_grad_dir_with_face_ghost(i,j,k,1,0,ps,dx[1],dlo,dhi,is_periodic[1]);
#if (AMREX_SPACEDIM == 3)
                    Real const psz = cc_grad_dir_with_face_ghost(i,j,k,2,0,ps,dx[2],dlo,dhi,is_periodic[2]);
#else
                    Real const psz = Real(0.0);
#endif
                    Real const mag = std::sqrt(AMREX_D_TERM(psx*psx, + psy*psy, + psz*psz));
                    Real const invmag = (m(i,j,k) && mag > smalltol) ? Real(1.0) / mag : Real(0.0);
                    n(i,j,k,0) = psx * invmag;
                    n(i,j,k,1) = psy * invmag;
#if (AMREX_SPACEDIM == 3)
                    n(i,j,k,2) = psz * invmag;
#endif

                    gp(i,j,k,0) = cc_grad_dir_with_face_ghost(i,j,k,0,0,p,dx[0],dlo,dhi,is_periodic[0]);
                    gp(i,j,k,1) = cc_grad_dir_with_face_ghost(i,j,k,1,0,p,dx[1],dlo,dhi,is_periodic[1]);
#if (AMREX_SPACEDIM == 3)
                    gp(i,j,k,2) = cc_grad_dir_with_face_ghost(i,j,k,2,0,p,dx[2],dlo,dhi,is_periodic[2]);
#endif
                });
            }
        }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(phi[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Array4<Real const> const& ps = psi[lev].const_array(mfi);
            Array4<Real const> const& n = ncc[lev].const_array(mfi);
            Array4<Real const> const& p = phi[lev].const_array(mfi);
            Array4<int const> const& m = mask.const_array(mfi);

            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                Box const& fbx = mfi.nodaltilebox(dir);
                Array4<Real> const& nf = nfc[lev][dir].array(mfi);
                Array4<Real> const& ar = areg[lev][dir].array(mfi);
                bool const periodic = geom[lev].isPeriodic(dir);
                int const lo_lim = (dir == 0) ? dlo.x : (dir == 1) ? dlo.y : dlo.z;
                int const hi_lim = (dir == 0) ? dhi.x : (dir == 1) ? dhi.y : dhi.z;

                ParallelFor(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    IntVect const iv(AMREX_D_DECL(i,j,k));
                    IntVect lo = iv;
                    lo[dir] -= 1;

                    Real diffusion = Real(0.0);
                    if (!periodic && iv[dir] == lo_lim) {
                        diffusion = m(iv) ? Gamma * epsilon * p(iv,0) : Real(0.0);
                    } else if (!periodic && iv[dir] == hi_lim + 1) {
                        diffusion = m(lo) ? Gamma * epsilon * p(lo,0) : Real(0.0);
                    } else if (m(lo) && m(iv)) {
                        diffusion = Gamma * epsilon * (p(iv,0) - p(lo,0)) * dxi[dir];
                    }

                    Real const pfc = avg_cc_to_face_masked(iv, dir, m, ps, lo_lim, hi_lim, periodic);
                    nf(i,j,k) = avg_cc_to_face_masked(iv, dir, m, n, lo_lim, hi_lim, periodic, dir);
                    Real const th = std::tanh(pfc / (Real(2.0) * epsilon));
                    Real const sharpening = -Real(0.25) * Gamma * (Real(1.0) - th*th) * nf(i,j,k);
                    ar(i,j,k,1) = sharpening;
                    ar(i,j,k,0) = diffusion + sharpening;
                });
            }
        }
    }

    for (int lev = 0; lev <= finest_level; ++lev) {
        MultiFab& divA = (step_type == StepType::Predictor)
            ? m_leveldata[lev]->interface_divA_o : m_leveldata[lev]->interface_divA;
        MultiFab kappa(grids[lev], dmap[lev], 1, 0, MFInfo(), Factory(lev));
        kappa.setVal(0.0);

        Array<MultiFab const*,AMREX_SPACEDIM> ar_ptr;
        Array<MultiFab const*,AMREX_SPACEDIM> nf_ptr;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            ar_ptr[dir] = &areg[lev][dir];
            nf_ptr[dir] = &nfc[lev][dir];
        }

#ifdef AMREX_USE_EB
        auto const& ebfactory = dynamic_cast<EBFArrayBoxFactory const&>(Factory(lev));
        if (ebfactory.isAllRegular()) {
            computeDivergence(divA, ar_ptr, geom[lev]);
            computeDivergence(kappa, nf_ptr, geom[lev]);
        } else {
            EB_computeDivergence(divA, ar_ptr, geom[lev], true);
            EB_computeDivergence(kappa, nf_ptr, geom[lev], true);
            EB_set_covered(divA, 0, divA.nComp(), divA.nGrow(), 0.0);
            EB_set_covered(kappa, 0, 1, 0, 0.0);
        }
#else
        computeDivergence(divA, ar_ptr, geom[lev]);
        computeDivergence(kappa, nf_ptr, geom[lev]);
#endif
        kappa.mult(-1.0, 0, 1, 0);

        Array<MultiFab*,AMREX_SPACEDIM> vel_fc_ptr;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            vel_fc_ptr[dir] = &vel_fc[lev][dir];
        }
        average_cellcenter_to_face(vel_fc_ptr, *vel[lev], geom[lev], AMREX_SPACEDIM);

        iMultiFab force_mask = make_interface_mask(phi[lev], phitol);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(phi[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Box const& bx = mfi.tilebox();
            Array4<Real> const& force = m_leveldata[lev]->interface_force.array(mfi);
            Array4<Real const> const& gp = dphi[lev].const_array(mfi);
            Array4<Real const> const& kap = kappa.const_array(mfi);
            Array4<int const> const& m = force_mask.const_array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (m(i,j,k)) {
                    AMREX_D_TERM(force(i,j,k,0) = sigma * kap(i,j,k) * gp(i,j,k,0);,
                                 force(i,j,k,1) = sigma * kap(i,j,k) * gp(i,j,k,1);,
                                 force(i,j,k,2) = sigma * kap(i,j,k) * gp(i,j,k,2););
                }
            });

            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                Box const& fbx = mfi.nodaltilebox(dir);
                Array4<Real> const& fxu = fxu_fc[lev][dir].array(mfi);
                Array4<Real const> const& vf = vel_fc[lev][dir].const_array(mfi);
                Array4<Real const> const& ar = areg[lev][dir].const_array(mfi);
                ParallelFor(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    AMREX_D_TERM(fxu(i,j,k,0) = rhod * ar(i,j,k,0) * vf(i,j,k,0);,
                                 fxu(i,j,k,1) = rhod * ar(i,j,k,0) * vf(i,j,k,1);,
                                 fxu(i,j,k,2) = rhod * ar(i,j,k,0) * vf(i,j,k,2););
                });
            }
        }

        Array<MultiFab const*,AMREX_SPACEDIM> fxu_ptr;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            fxu_ptr[dir] = &fxu_fc[lev][dir];
        }
        MultiFab freg(grids[lev], dmap[lev], AMREX_SPACEDIM, 0, MFInfo(), Factory(lev));
        freg.setVal(0.0);
#ifdef AMREX_USE_EB
        if (ebfactory.isAllRegular()) {
            computeDivergence(freg, fxu_ptr, geom[lev]);
        } else {
            EB_computeDivergence(freg, fxu_ptr, geom[lev], true);
            EB_set_covered(freg, 0, freg.nComp(), freg.nGrow(), 0.0);
        }
#else
        computeDivergence(freg, fxu_ptr, geom[lev]);
#endif
        MultiFab::Add(m_leveldata[lev]->interface_force, freg, 0, 0, AMREX_SPACEDIM, 0);
    }
}

void incflo::add_interface_regularization (StepType step_type)
{
    if (!m_diffuse_interface) { return; }

    Real const dt = m_dt;
    bool const explicit_diffusion = (m_diff_type == DiffusionType::Explicit);
    bool const crank_nicolson = (m_diff_type == DiffusionType::Crank_Nicolson);
    bool const implicit_diffusion = (m_diff_type == DiffusionType::Implicit);

    for (int lev = 0; lev <= finest_level; ++lev) {
        auto& ld = *m_leveldata[lev];
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(ld.tracer,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Box const& bx = mfi.tilebox();
            Array4<Real> const& tracer = ld.tracer.array(mfi);
            Array4<Real const> const& divA_o = ld.interface_divA_o.const_array(mfi);

            if (step_type == StepType::Predictor) {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (explicit_diffusion) {
                        tracer(i,j,k,0) += dt * divA_o(i,j,k,0);
                    } else if (crank_nicolson) {
                        tracer(i,j,k,0) += Real(0.5) * dt * (divA_o(i,j,k,0) + divA_o(i,j,k,1));
                    } else {
                        tracer(i,j,k,0) += dt * divA_o(i,j,k,1);
                    }
                });
            } else {
                Array4<Real const> const& divA = ld.interface_divA.const_array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (explicit_diffusion) {
                        tracer(i,j,k,0) += Real(0.5) * dt * (divA_o(i,j,k,0) + divA(i,j,k,0));
                    } else if (crank_nicolson) {
                        tracer(i,j,k,0) += Real(0.5) * dt * (divA_o(i,j,k,0) + divA(i,j,k,1));
                    } else {
                        tracer(i,j,k,0) += Real(0.5) * dt * (divA_o(i,j,k,1) + divA(i,j,k,1));
                    }
                });
            }
        }
    }

    if (m_diff_type != DiffusionType::Explicit) {
        Real const diff_dt = implicit_diffusion ? dt : Real(0.5) * dt;
        diffuse_interface(diff_dt);
    }
}

void incflo::diffuse_interface (Real dt_diff)
{
    Vector<MultiFab*> tracer = get_tracer_new();
    Vector<MultiFab> phi;
    alias_tracer0(phi, tracer);
    Vector<MultiFab*> phi_ptr;
    phi_ptr.reserve(phi.size());

    Vector<MultiFab> eta;
    eta.reserve(phi.size());
    Vector<MultiFab const*> eta_ptr;
    eta_ptr.reserve(phi.size());

    Vector<MultiFab> eb_empty;
    Vector<MultiFab*> eb_ptr;
    eb_empty.reserve(phi.size());
    eb_ptr.reserve(phi.size());

    for (int lev = 0; lev <= finest_level; ++lev) {
        phi_ptr.push_back(&phi[lev]);
        eta.emplace_back(grids[lev], dmap[lev], 1, 1, MFInfo(), Factory(lev));
        eta.back().setVal(
            interface_gamma(finest_level, m_interface_fixed_gamma, m_interface_gamma, m_interface_gamma_star, get_velocity_new_const())
          * interface_epsilon(geom[lev], m_interface_fixed_epsilon, m_interface_epsilon, m_interface_epsilon_star)
                         );
        eta_ptr.push_back(&eta.back());
        eb_empty.emplace_back();
        eb_ptr.push_back(&eb_empty.back());
        fillphysbc_tracer(lev, m_cur_time + m_dt, *tracer[lev], 1);
    }

    get_diffusion_scalar_op()->diffuse_scalar(phi_ptr, get_density_new(), eta_ptr,
                                              eb_ptr, {0}, {m_bcrec_tracer[0]}, dt_diff);

    for (int lev = 0; lev <= finest_level; ++lev) {
        int const ng = tracer[lev]->nGrow();
        if (ng > 0) {
            fillpatch_tracer(lev, m_t_new[lev], *tracer[lev], ng);
        }
    }
}
