#include <AMReX_Box.H>

#include <incflo.H>
#include <incflo_derive_K.H>

using namespace amrex;

void incflo::ComputeDivU(Real /*time_in*/)
{
#if 0

    incflo_set_velocity_bcs(time_in, vel);

    int bc_lo[3], bc_hi[3];
    Box domain(geom[0].Domain());

    set_ppe_bcs(bc_lo, bc_hi,
                domain.loVect(), domain.hiVect(),
                &nghost,
                bc_ilo[0]->dataPtr(), bc_ihi[0]->dataPtr(),
                bc_jlo[0]->dataPtr(), bc_jhi[0]->dataPtr(),
                bc_klo[0]->dataPtr(), bc_khi[0]->dataPtr());

    ppe_lobc = {(LinOpBCType)bc_lo[0], (LinOpBCType)bc_lo[1], (LinOpBCType)bc_lo[2]};
    ppe_hibc = {(LinOpBCType)bc_hi[0], (LinOpBCType)bc_hi[1], (LinOpBCType)bc_hi[2]};

    LPInfo lpinfo;

    //
    // This rebuilds integrals each time linop is created -- must find a better way
    //

#ifdef AMREX_USE_EB
    MLNodeLaplacian linop(geom, grids, dmap, lpinfo, GetVecOfConstPtrs(ebfactory));
#else
    MLNodeLaplacian linop(geom, grids, dmap, lpinfo);
#endif
    linop.setDomainBC(ppe_lobc,ppe_hibc);
    linop.compDivergence(GetVecOfPtrs(divu),GetVecOfPtrs(vel));
#endif
}

#ifdef AMREX_USE_EB
void incflo::compute_strainrate_at_level (int lev,
#else
void incflo::compute_strainrate_at_level (int /*lev*/,
#endif
                                          MultiFab* strainrate,
                                          const MultiFab* vel,
                                          Geometry& lev_geom,
                                          Real /*time*/, int nghost)
{

#ifdef AMREX_USE_EB
        auto const& fact = EBFactory(lev);
        auto const& flags = fact.getMultiEBCellFlagFab();
        MultiCutFab const& bcent = fact.getBndryCent();
        MultiCutFab const& ccent = fact.getCentroid();
        MultiCutFab const& bnorm = fact.getBndryNormal();
#endif

        AMREX_D_TERM(Real idx = Real(1.0) / lev_geom.CellSize(0);,
                     Real idy = Real(1.0) / lev_geom.CellSize(1);,
                     Real idz = Real(1.0) / lev_geom.CellSize(2););

        const Dim3 dlo = amrex::lbound(lev_geom.Domain());
        const Dim3 dhi = amrex::ubound(lev_geom.Domain());
        GpuArray<bool, AMREX_SPACEDIM> is_periodic;
        AMREX_D_TERM(is_periodic[0] = lev_geom.isPeriodic(0);,
                     is_periodic[1] = lev_geom.isPeriodic(1);,
                     is_periodic[2] = lev_geom.isPeriodic(2););
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*strainrate,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
                Box const& bx = mfi.growntilebox(nghost);
                Array4<Real> const& sr_arr = strainrate->array(mfi);
                Array4<Real const> const& vel_arr = vel->const_array(mfi);
#ifdef AMREX_USE_EB
                auto const& flag_fab = flags[mfi];
                auto typ = flag_fab.getType(bx);
                if (typ == FabType::covered)
                {
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        sr_arr(i,j,k) = Real(0.0);
                    });
                }
                else if (typ == FabType::singlevalued)
                {
                    Array4<Real const> const& bcfab      = bcent.const_array(mfi);
                    Array4<Real const> const& ccfab      = ccent.const_array(mfi);
                    Array4<Real const> const& bnrmfab    = bnorm.const_array(mfi);
                    auto const& flag_arr = flag_fab.const_array();
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        sr_arr(i,j,k) = incflo_strainrate_eb(i,j,k,AMREX_D_DECL(idx,idy,idz),
                                                             vel_arr,flag_arr, dlo, dhi,
                                                             is_periodic, true, ccfab, bcfab,
                                                             AMREX_D_DECL(bnrmfab(i,j,k,0),
                                                                          bnrmfab(i,j,k,1),
                                                                          bnrmfab(i,j,k,2)));
                    });
                }
                else
#endif
                {
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        sr_arr(i,j,k) = incflo_strainrate(i,j,k,AMREX_D_DECL(idx,idy,idz),
                                                          vel_arr, dlo, dhi, is_periodic);
                    });
                }
        }
}

void incflo::compute_nodal_strainrate_at_level (int /*lev*/,
                                          MultiFab* strainrate,
                                          const MultiFab* vel,
                                          Geometry& lev_geom,
                                          Real /*time*/, int nghost)
{
    AMREX_D_TERM(Real idx = Real(1.0) / lev_geom.CellSize(0);,
                 Real idy = Real(1.0) / lev_geom.CellSize(1);,
                 Real idz = Real(1.0) / lev_geom.CellSize(2););
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
        for (MFIter mfi(*strainrate,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
                Box const& bx = mfi.growntilebox(nghost);
                Array4<Real> const& sr_arr = strainrate->array(mfi);
                Array4<Real const> const& vel_arr = vel->const_array(mfi);
                {
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        sr_arr(i,j,k) = incflo_strainrate_nodal(i,j,k,AMREX_D_DECL(idx,idy,idz),
                                                             vel_arr,dlo,dhi,bc_type);
                    });
                }
        }
}

void incflo::compute_nodal_hydrostatic_pressure_at_level (int lev,
                                          MultiFab* p_static,
                                          const MultiFab* rho_cc,
                                          Real p_surface,
                                          Geometry& lev_geom,
                                          int nghost)
{
    if (lev > 0) {
        amrex::Abort("Hydrostatic pressure is not implemented for lev > 0");
    }

    MultiFab rho_nodal(p_static->boxArray(), p_static->DistributionMap(),1,nghost);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rho_nodal,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& bx = mfi.tilebox();
        Array4<Real const> const& rho_arr = rho_cc->const_array(mfi);
        Array4<Real> const& rho_nodal_arr = rho_nodal.array(mfi);
        const int prob_534 = (m_probtype == 534) ? 1 : 0;
        const Real rho_1 = m_ro_0;
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
           rho_nodal_arr(i,j,k) = incflo_nodal_density(i,j,k,rho_arr);
           // In inclined_plane_granular, the free surface needs to have
           // hydrostatic pressure of zero
           if (prob_534) {
              if (rho_nodal_arr(i,j,k) == rho_1) {
                 rho_nodal_arr(i,j,k) -= rho_1;
              }
           }
        });
    }

    BoxArray pencil_ba(surroundingNodes(lev_geom.Domain()));
#if (AMREX_SPACEDIM==2)
    IntVect pencil_iv(8,1048576);
#else
    IntVect pencil_iv(8,8,1048576);
#endif
    pencil_ba.maxSize(pencil_iv);
    DistributionMapping pencil_dm{pencil_ba};
    MultiFab pencil_rho_nodal(pencil_ba,pencil_dm,1,nghost);
    pencil_rho_nodal.ParallelCopy(rho_nodal,lev_geom.periodicity());
    MultiFab pencil_p_static(pencil_ba,pencil_dm,1,nghost);

    Real idx = Real(1.0) / lev_geom.CellSize(0);
    Real idy = Real(1.0) / lev_geom.CellSize(1);
#if (AMREX_SPACEDIM == 3)
    Real idz = Real(1.0) / lev_geom.CellSize(2);
#endif

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(pencil_rho_nodal,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // Ensure that static pressure is calculated based on validbox
        Box const& bx = mfi.tilebox();
        Array4<Real> const& p_static_arr = pencil_p_static.array(mfi);
        Array4<Real const> const& rho_nodal_arr = pencil_rho_nodal.const_array(mfi);
        const Real gravity = std::abs(m_gravity[AMREX_SPACEDIM-1]);
        // Even for pencil_ba, this needs to be based on validbox
        const Dim3 v_bxlo = amrex::lbound(mfi.validbox());
        const Dim3 v_bxhi = amrex::ubound(mfi.validbox());
        const int level = lev;
        const Real p_srf = p_surface;
#if (AMREX_SPACEDIM == 2)
        int h_end   = v_bxhi.y;
#else
        int h_end   = v_bxhi.z;
#endif
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real p_validbox_top = p_srf;
            if (level > 0) {
#if (AMREX_SPACEDIM == 2)
                p_validbox_top = p_static_arr(i,h_end,k);
#else
                p_validbox_top = p_static_arr(i,j,h_end);
#endif
            }
            p_static_arr(i,j,k) = p_validbox_top;
            p_static_arr(i,j,k) += incflo_local_hydrostatic_pressure_nodal(
                                           i,j,k,AMREX_D_DECL(idx,idy,idz),
                                           gravity,rho_nodal_arr,v_bxlo,v_bxhi);
        });
    }
    // nodal MultiFab for hydrostatic pressure
    p_static->ParallelCopy(pencil_p_static,lev_geom.periodicity());
}

void incflo::compute_cc_hydrostatic_pressure_at_level (int lev,
                                          MultiFab* p_static,
                                          const MultiFab* rho,
                                          Real p_surface,
                                          Geometry& lev_geom,
                                          int nghost)
{
    if (lev > 0) {
        amrex::Abort("Hydrostatic pressure is not implemented for lev > 0");
    }
    // This is a copy of cell-centered rho
    // because prob_534 requires special handling
    MultiFab rho_cc(p_static->boxArray(), p_static->DistributionMap(),1,nghost);
    MultiFab::Copy(rho_cc, *rho, 0, 0, 1, nghost);
    if (m_probtype == 534) {
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
       for (MFIter mfi(rho_cc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
       {
           Box const& bx = mfi.tilebox();
           Array4<Real> const& rho_arr = rho_cc.array(mfi);
           const Real rho_1 = m_ro_0;
           amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
           {
              // In inclined_plane_granular, the free surface needs to have
              // hydrostatic pressure of zero
                 if (rho_arr(i,j,k) == rho_1) {
                    rho_arr(i,j,k) -= rho_1;
                 }
           });
       }
    }
    BoxArray pencil_ba(lev_geom.Domain());
#if (AMREX_SPACEDIM==2)
    IntVect pencil_iv(8,1048576);
#else
    IntVect pencil_iv(8,8,1048576);
#endif
    pencil_ba.maxSize(pencil_iv);
    DistributionMapping pencil_dm{pencil_ba};
    MultiFab pencil_rho_cc(pencil_ba,pencil_dm,1,nghost);
    pencil_rho_cc.ParallelCopy(rho_cc,lev_geom.periodicity());
    MultiFab pencil_p_static(pencil_ba,pencil_dm,1,nghost);

    Real idx = Real(1.0) / lev_geom.CellSize(0);
    Real idy = Real(1.0) / lev_geom.CellSize(1);
#if (AMREX_SPACEDIM == 3)
    Real idz = Real(1.0) / lev_geom.CellSize(2);
#endif

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(pencil_rho_cc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // Ensure that static pressure is calculated based on validbox
        Box const& bx = mfi.tilebox();
        Array4<Real      > const& p_static_arr = pencil_p_static.array(mfi);
        Array4<Real const> const& rho_cc_arr   = pencil_rho_cc.const_array(mfi);
        const Real gravity = std::abs(m_gravity[AMREX_SPACEDIM-1]);
        // Even for pencil_ba, this needs to be based on validbox
        const Dim3 v_bxlo = amrex::lbound(mfi.validbox());
        const Dim3 v_bxhi = amrex::ubound(mfi.validbox());
        const int level = lev;
        const Real p_srf = p_surface;
#if (AMREX_SPACEDIM == 2)
        int h_end   = v_bxhi.y;
#else
        int h_end   = v_bxhi.z;
#endif
        // Note: p_valid_box logic ONLY works for max_level = 0
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real p_validbox_top = p_srf;
            if (level > 0) {
#if (AMREX_SPACEDIM == 2)
                p_validbox_top = p_static_arr(i,h_end,k);
#else
                p_validbox_top = p_static_arr(i,j,h_end);
#endif
            }
            p_static_arr(i,j,k) = p_validbox_top;
            p_static_arr(i,j,k) += incflo_local_hydrostatic_pressure_cc(
                                           i,j,k,AMREX_D_DECL(idx,idy,idz),
                                           gravity,rho_cc_arr,v_bxlo,v_bxhi);
        });
    }
    // MultiFab for hydrostatic pressure
    p_static->ParallelCopy(pencil_p_static,lev_geom.periodicity());
}

void incflo::compute_inertial_num_at_level (int lev,
                                          MultiFab* inertial_num,
                                          const MultiFab* strainrate,
                                          const MultiFab* press,
                                          Real p_eps, Real ro_grain,
                                          Real diam_grain,
                                          int nghost)
{
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*inertial_num,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& bx = mfi.growntilebox(nghost);
        Array4<Real const> const& p_nd_arr = press->const_array(mfi);
        Array4<Real const> const& sr_arr = strainrate->const_array(mfi);
        Array4<Real> const& inrt_num_arr = inertial_num->array(mfi);
        const Real eps = p_eps;
        const Real diam_scnd = diam_grain;
        const Real ro_scnd = ro_grain;
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
             // Regularized Pressure
             //Real p_reg = std::sqrt(p_nd_arr(i,j,k)*p_nd_arr(i,j,k)
             //                       + eps*eps);
             //p_reg += p_nd_arr(i,j,k);
             //p_reg *= Real(0.5);

             // Note: This version of Regularized Pressure only works for
             // Static pressure, i.e., p_s >= 0
             Real p_reg = p_nd_arr(i,j,k) + eps;

             // Strainrate in incflo is two-times the actual value
             inrt_num_arr(i,j,k) = std::sqrt(ro_scnd/p_reg)*
                                   diam_scnd*Real(0.5)*sr_arr(i,j,k);
        });
    }
    amrex::ignore_unused<int>(lev);
}

void incflo::compute_nodal_second_fluid_conc (MultiFab* conc_second_nd,
                                              const MultiFab* rho, int nghost) const
{
    // A cell-centered MultiFab for concentration of second fluid,
    // needs to have ghost cells
    MultiFab conc_second_cc(rho->boxArray(),rho->DistributionMap(),1,nghost+1);
    conc_second_cc.setVal(-1.0);
    if (m_two_fluid_cc_rho_conc) {
       compute_cc_second_fluid_conc(&conc_second_cc, rho, nghost+1);
    }
    // Obtain concentration of the second fluid, based on nodal density
    MultiFab rho_nodal(conc_second_nd->boxArray(),
                       conc_second_nd->DistributionMap(),1,nghost);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*conc_second_nd,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& bx = mfi.tilebox();
        Array4<Real const> const& rho_arr = rho->const_array(mfi);
        Array4<Real const> const& conc_second_cc_arr = conc_second_cc.const_array(mfi);
        Array4<Real> const& rho_nodal_arr = rho_nodal.array(mfi);
        Array4<Real> const& conc_second_nd_arr = conc_second_nd->array(mfi);
        const Real rho_first = m_ro_0;
        const Real rho_second = m_ro_0_second;
        const bool rho_harmonic = m_two_fluid_rho_harmonic;
        // This boolean represents if concentration is calculated based on
        // nodal or cell-centered density
        const bool cc_rho_conc = m_two_fluid_cc_rho_conc;
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
           rho_nodal_arr(i,j,k) = incflo_nodal_density(i,j,k,rho_arr);
           if (cc_rho_conc) {
             conc_second_nd_arr(i,j,k) = incflo_nodal_second_conc(i,j,k,
                                                 conc_second_cc_arr);
           }
           else {
             Real conc_scnd = Real(-1.0);
             if (rho_harmonic) {
                 // Based on weighted harmonic mean for nodal density
                 conc_scnd =
                   ((rho_first*rho_second)/rho_nodal_arr(i,j,k)) - rho_second;
                 conc_scnd /= (rho_first-rho_second);
             }
             else {
                 // Based on weighted arithmetic mean for nodal density
                 conc_scnd = (rho_nodal_arr(i,j,k)-rho_first)/(rho_second-rho_first);
             }
             // Put guards
             conc_second_nd_arr(i,j,k) =
                     amrex::min(Real(1.0),amrex::max(Real(0.0),conc_scnd));
           }
        });
    }
}

// This function assumes the logic that UNUSED COVERED CELLS have density = 0.
void incflo::compute_cc_second_fluid_conc (MultiFab* conc_second_cc,
                                          const  MultiFab* rho, int nghost) const
{
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
   for (MFIter mfi(*conc_second_cc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
   {
       Box const& bx = mfi.growntilebox(nghost);
       Array4<Real const> const& rho_arr = rho->const_array(mfi);
       Array4<Real> const& conc_second_arr = conc_second_cc->array(mfi);
       const Real rho_first = m_ro_0;
       const Real rho_second = m_ro_0_second;
       const bool rho_harmonic = m_two_fluid_rho_harmonic;
       amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
          if (rho_arr(i,j,k) > Real(0.)) {
             Real conc_scnd = Real(-1.0);
             if (rho_harmonic) {
                // Based on weighted harmonic mean for cell-centered density
                conc_scnd =
                  ((rho_first*rho_second)/rho_arr(i,j,k)) - rho_second;
                conc_scnd /= (rho_first-rho_second);
             }
             else {
                // Based on weighted arithmetic mean for cell-centered density
                conc_scnd = (rho_arr(i,j,k)-rho_first)/(rho_second-rho_first);
             }
             // Put guards
             conc_second_arr(i,j,k) =
               amrex::min(Real(1.0),amrex::max(Real(0.0),conc_scnd));
          }
          else {
             conc_second_arr(i,j,k) = Real(-10.);
          }
       });
   }
}

void incflo::compute_cc_second_fluid_conc_from_tracer (
                                          MultiFab* conc_second_cc,
                                          const MultiFab* tracer,
                                          int nghost) const
{
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
   for (MFIter mfi(*conc_second_cc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
   {
       Box const& bx = mfi.growntilebox(nghost);
       Array4<Real const> const& tracer_arr = tracer->const_array(mfi);
       Array4<Real> const& conc_second_arr = conc_second_cc->array(mfi);
       amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
          conc_second_arr(i,j,k) =
              amrex::Clamp(tracer_arr(i,j,k,0), Real(0.), Real(1.));
       });
   }
}

void incflo::compute_nodal_second_fluid_conc_from_tracer (
                                          MultiFab* conc_second_nd,
                                          const MultiFab* tracer,
                                          int nghost) const
{
    const int cc_nghost = nghost + 1;
    MultiFab conc_second_cc(tracer->boxArray(), tracer->DistributionMap(),
                            1, cc_nghost, MFInfo(), tracer->Factory());
    compute_cc_second_fluid_conc_from_tracer(&conc_second_cc, tracer, cc_nghost);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*conc_second_nd,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& bx = mfi.growntilebox(nghost);
        Array4<Real const> const& conc_second_cc_arr = conc_second_cc.const_array(mfi);
        Array4<Real> const& conc_second_nd_arr = conc_second_nd->array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            conc_second_nd_arr(i,j,k) = amrex::Clamp(
                incflo_nodal_second_conc(i,j,k,conc_second_cc_arr),
                Real(0.), Real(1.));
        });
    }
}

void incflo::compute_gradientOfVelocity_on_level (int lev, MultiFab& gradVel,
                                                  const MultiFab& velocity,
                                                  Geometry& lev_geom)
{
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
    for (MFIter mfi(velocity,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
            Box const& bx = mfi.tilebox();
            Array4<Real const> const& vel_arr = velocity.const_array(mfi);
            Array4<Real      > const& gradVel_arr = gradVel.array(mfi);
#ifdef AMREX_USE_EB
            auto const& flag_fab = flags[mfi];
            auto typ = flag_fab.getType(bx);
            if (typ == FabType::covered)
            {
                // Do nothing; already initialised to zero
            }
            else if (typ == FabType::singlevalued)
            {
                auto const& flag_arr = flag_fab.const_array();
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    incflo_gradientOfVelocity_eb(i,j,k,AMREX_D_DECL(idx,idy,idz),
                                                 vel_arr,gradVel_arr,flag_arr(i,j,k));
                });
            }
            else
#endif
            {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    incflo_gradientOfVelocity(i,j,k,AMREX_D_DECL(idx,idy,idz),
                                              vel_arr,gradVel_arr);
                });
            }
    }
}

Real incflo::ComputeKineticEnergy ()
{
#if 0
    BL_PROFILE("incflo::ComputeKineticEnergy");

    // integrated total Kinetic energy
    Real KE = Real(0.0);

    for(int lev = 0; lev <= finest_level; lev++)
    {
        Real cell_vol = geom[lev].CellSize()[0]*geom[lev].CellSize()[1]*geom[lev].CellSize()[2];

        KE += amrex::ReduceSum(*density[lev],*vel[lev],*level_mask[lev],0,
        [=] AMREX_GPU_HOST_DEVICE (Box const& bx,
                                   Array4<Real const> const& den_arr,
                                   Array4<Real const> const& vel_arr,
                                   Array4<int const>  const& mask_arr) -> Real
        {
            Real KE_Fab = Real(0.0);

            amrex::Loop(bx, [=,&KE_Fab] (int i, int j, int k) noexcept
            {
                KE_Fab += cell_vol*mask_arr(i,j,k)*den_arr(i,j,k)*( vel_arr(i,j,k,0)*vel_arr(i,j,k,0)
                                                                   +vel_arr(i,j,k,1)*vel_arr(i,j,k,1)
                                                                   +vel_arr(i,j,k,2)*vel_arr(i,j,k,2));

            });
            return KE_Fab;

        });
    }

    // total volume of grid on level 0
    Real total_vol = geom[0].ProbDomain().volume();

    KE *= Real(0.5)/total_vol/ro_0;

    ParallelDescriptor::ReduceRealSum(KE);

    return KE;

#endif
    return 0;
}

Real incflo::ComputeGranularKineticEnergy ()
{
    BL_PROFILE("incflo::ComputeGranularKineticEnergy");

    // integrated total Kinetic energy
    Real KE = Real(0.0);

    auto density = get_density_new();
    auto vel     = get_velocity_new();
    auto tracer  = get_tracer_new();

    for(int lev = 0; lev <= finest_level; lev++)
    {
        Real cell_vol = geom[lev].CellSize()[0]*geom[lev].CellSize()[1]*geom[lev].CellSize()[2];

        MultiFab gran_dens(density[lev]->boxArray(), density[lev]->DistributionMap(),
                           1, 0, MFInfo(), density[lev]->Factory());
        gran_dens.setVal(Real(0.));
#ifdef AMREX_USE_EB
        if (auto const* factory_eb =
                 dynamic_cast<EBFArrayBoxFactory const*>(&(density[lev]->Factory()))) {
            MultiFab const& temp_mf = factory_eb->getVolFrac();
            MultiFab::AddProduct(gran_dens, *density[lev], 0, temp_mf, 0,
                                 0, 1, 0);
        }
        else
#endif
        {
            MultiFab::Copy(gran_dens,*density[lev],0,0,1,0);
        }
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(gran_dens,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
                Box const& bx = mfi.tilebox();
                Array4<Real> const& gran_dens_arr = gran_dens.array(mfi);
                Array4<Real const> const& conc_arr = tracer[lev]->const_array(mfi);
                Real min_conc_second = m_min_conc_second;
                {
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        if (conc_arr(i,j,k) < min_conc_second) {
                            gran_dens_arr(i,j,k) = Real(0.);
                        }
                    });
                }
        }
        // Level_mask creation
        iMultiFab level_mask(grids[lev], dmap[lev], 1, 0);
        if (lev < finest_level) {
            level_mask = amrex::makeFineMask(grids[lev], dmap[lev],
                                grids[lev+1], refRatio(lev), 1, 0);
        } else {
            level_mask.setVal(1);
        }

        KE += amrex::ReduceSum(gran_dens,*vel[lev],level_mask,0,
        [=] AMREX_GPU_HOST_DEVICE (Box const& bx,
                                   Array4<Real const> const& den_arr,
                                   Array4<Real const> const& vel_arr,
                                   Array4<int const>  const& mask_arr) -> Real
        {
            Real KE_Fab = Real(0.0);

            amrex::Loop(bx, [=,&KE_Fab] (int i, int j, int k) noexcept
            {
                KE_Fab += cell_vol*mask_arr(i,j,k)*den_arr(i,j,k)*(
                                   AMREX_D_TERM(vel_arr(i,j,k,0)*vel_arr(i,j,k,0),
                                                +vel_arr(i,j,k,1)*vel_arr(i,j,k,1),
                                                +vel_arr(i,j,k,2)*vel_arr(i,j,k,2)));
            });
            return KE_Fab;
        });
    }

    KE *= Real(0.5);

    ParallelDescriptor::ReduceRealSum(KE);

    return KE;
}

#ifdef AMREX_USE_EB
void incflo::ComputeMagVel (int lev,
#else
void incflo::ComputeMagVel (int /*lev*/,
#endif
                            Real /*time*/,
                            MultiFab& magvel, MultiFab const& vel)
{
    BL_PROFILE("incflo::ComputeMagVel");

#ifdef AMREX_USE_EB
    auto const& fact = EBFactory(lev);
    auto const& flags_mf = fact.getMultiEBCellFlagFab();
#endif

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for(MFIter mfi(vel, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box bx = mfi.tilebox();
        Array4<Real const> const& ccvel_fab = vel.const_array(mfi);
        Array4<Real> const& magvel_fab = magvel.array(mfi);

#ifdef AMREX_USE_EB
        const EBCellFlagFab& flags = flags_mf[mfi];
        auto typ = flags.getType(bx);
        if (typ == FabType::covered)
        {
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                magvel_fab(i,j,k) = Real(0.0);
            });
        }
        else
        {
            const auto& flag_fab = flags.const_array();
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (flag_fab(i,j,k).isCovered())
                {
                    magvel_fab(i,j,k) = Real(0.0);
                }
                else
                {
                    Real u = ccvel_fab(i,j,k,0);
                    Real v = ccvel_fab(i,j,k,1);
#if (AMREX_SPACEDIM == 2)
                    magvel_fab(i,j,k) = std::sqrt(u*u + v*v);
#elif  (AMREX_SPACEDIM == 3)
                    Real w = ccvel_fab(i,j,k,2);
                    magvel_fab(i,j,k) = std::sqrt(u*u + v*v + w*w);
#endif
                }
            });
        }
#else       // No EB
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real u = ccvel_fab(i,j,k,0);
                Real v = ccvel_fab(i,j,k,1);
#if (AMREX_SPACEDIM == 2)
                magvel_fab(i,j,k) = std::sqrt(u*u + v*v);
#elif (AMREX_SPACEDIM == 3)
                Real w = ccvel_fab(i,j,k,2);
                magvel_fab(i,j,k) = std::sqrt(u*u + v*v + w*w);
#endif
            });
#endif
    } // mfi
}

#if (AMREX_SPACEDIM == 2)
void incflo::ComputeVorticity (int lev, Real /*time*/, MultiFab& vort, MultiFab const& vel)
{
    BL_PROFILE("incflo::ComputeVorticity");
    const Real idx = Geom(lev).InvCellSize(0);
    const Real idy = Geom(lev).InvCellSize(1);

#ifdef AMREX_USE_EB
    const auto& fact = EBFactory(lev);
    const auto& flags_mf = fact.getMultiEBCellFlagFab();
#endif

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for(MFIter mfi(vel, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box bx = mfi.tilebox();
        Array4<Real const> const& ccvel_fab = vel.const_array(mfi);
        Array4<Real> const& vort_fab = vort.array(mfi);

#ifdef AMREX_USE_EB
        const EBCellFlagFab& flags = flags_mf[mfi];
        auto typ = flags.getType(bx);
        if (typ == FabType::covered)
        {
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                vort_fab(i,j,k) = Real(0.0);
            });
        }
        else if (typ == FabType::singlevalued)
        {
            const auto& flag_fab = flags.const_array();
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                constexpr Real c0 = Real(-1.5);
                constexpr Real c1 = Real( 2.0);
                constexpr Real c2 = Real(-0.5);

                if (flag_fab(i,j,k).isCovered())
                {
                    vort_fab(i,j,k) = Real(0.0);
                }
                else
                {
                    Real vx, uy;
                    // Need to check if there are covered cells in neighbours --
                    // -- if so, use one-sided difference computation (but still quadratic)
                    if (!flag_fab(i,j,k).isConnected( 1,0,0))
                    {
                        // Covered cell to the right, go fish left
                        vx = - (c0 * ccvel_fab(i  ,j,k,1)
                              + c1 * ccvel_fab(i-1,j,k,1)
                              + c2 * ccvel_fab(i-2,j,k,1)) * idx;
                    }
                    else if (!flag_fab(i,j,k).isConnected(-1,0,0))
                    {
                        // Covered cell to the left, go fish right
                        vx = (c0 * ccvel_fab(i  ,j,k,1)
                            + c1 * ccvel_fab(i+1,j,k,1)
                            + c2 * ccvel_fab(i+2,j,k,1)) * idx;
                    }
                    else
                    {
                        // No covered cells right or left, use standard stencil
                        vx = Real(0.5) * (ccvel_fab(i+1,j,k,1) - ccvel_fab(i-1,j,k,1)) * idx;
                    }
                    // Do the same in y-direction
                    if (!flag_fab(i,j,k).isConnected(0, 1,0))
                    {
                        uy = - (c0 * ccvel_fab(i,j  ,k,0)
                              + c1 * ccvel_fab(i,j-1,k,0)
                              + c2 * ccvel_fab(i,j-2,k,0)) * idy;
                    }
                    else if (!flag_fab(i,j,k).isConnected(0,-1,0))
                    {
                        uy = (c0 * ccvel_fab(i,j  ,k,0)
                            + c1 * ccvel_fab(i,j+1,k,0)
                            + c2 * ccvel_fab(i,j+2,k,0)) * idy;
                    }
                    else
                    {
                        uy = Real(0.5) * (ccvel_fab(i,j+1,k,0) - ccvel_fab(i,j-1,k,0)) * idy;
                    }
                    vort_fab(i,j,k) = vx-uy;
                }
            });
        }
        else
#endif
        {
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real vx = Real(0.5) * (ccvel_fab(i+1,j,k,1) - ccvel_fab(i-1,j,k,1)) * idx;
                Real uy = Real(0.5) * (ccvel_fab(i,j+1,k,0) - ccvel_fab(i,j-1,k,0)) * idy;
                vort_fab(i,j,k) = vx-uy;
            });
        }
    }
}

#elif (AMREX_SPACEDIM == 3)
void incflo::ComputeVorticity (int lev, Real /*time*/, MultiFab& vort, MultiFab const& vel)
{
    BL_PROFILE("incflo::ComputeVorticity");
    const Real idx = Geom(lev).InvCellSize(0);
    const Real idy = Geom(lev).InvCellSize(1);
    const Real idz = Geom(lev).InvCellSize(2);

#ifdef AMREX_USE_EB
    const auto& fact = EBFactory(lev);
    const auto& flags_mf = fact.getMultiEBCellFlagFab();
#endif

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for(MFIter mfi(vel, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box bx = mfi.tilebox();
        Array4<Real const> const& ccvel_fab = vel.const_array(mfi);
        Array4<Real> const& vort_fab = vort.array(mfi);

#ifdef AMREX_USE_EB
        const EBCellFlagFab& flags = flags_mf[mfi];
        auto typ = flags.getType(bx);
        if (typ == FabType::covered)
        {
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                vort_fab(i,j,k) = Real(0.0);
            });
        }
        else if (typ == FabType::singlevalued)
        {
            const auto& flag_fab = flags.const_array();
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                constexpr Real c0 = Real(-1.5);
                constexpr Real c1 = Real( 2.0);
                constexpr Real c2 = Real(-0.5);

                if (flag_fab(i,j,k).isCovered())
                {
                    vort_fab(i,j,k) = Real(0.0);
                }
                else
                {
                    Real vx, wx, uy, wy, uz, vz;
                    // Need to check if there are covered cells in neighbours --
                    // -- if so, use one-sided difference computation (but still quadratic)
                    if (!flag_fab(i,j,k).isConnected( 1,0,0))
                    {
                        // Covered cell to the right, go fish left
                        vx = - (c0 * ccvel_fab(i  ,j,k,1)
                              + c1 * ccvel_fab(i-1,j,k,1)
                              + c2 * ccvel_fab(i-2,j,k,1)) * idx;
                        wx = - (c0 * ccvel_fab(i  ,j,k,2)
                              + c1 * ccvel_fab(i-1,j,k,2)
                              + c2 * ccvel_fab(i-2,j,k,2)) * idx;
                    }
                    else if (!flag_fab(i,j,k).isConnected(-1,0,0))
                    {
                        // Covered cell to the left, go fish right
                        vx = (c0 * ccvel_fab(i  ,j,k,1)
                            + c1 * ccvel_fab(i+1,j,k,1)
                            + c2 * ccvel_fab(i+2,j,k,1)) * idx;
                        wx = (c0 * ccvel_fab(i  ,j,k,2)
                            + c1 * ccvel_fab(i+1,j,k,2)
                            + c2 * ccvel_fab(i+2,j,k,2)) * idx;
                    }
                    else
                    {
                        // No covered cells right or left, use standard stencil
                        vx = Real(0.5) * (ccvel_fab(i+1,j,k,1) - ccvel_fab(i-1,j,k,1)) * idx;
                        wx = Real(0.5) * (ccvel_fab(i+1,j,k,2) - ccvel_fab(i-1,j,k,2)) * idx;
                    }
                    // Do the same in y-direction
                    if (!flag_fab(i,j,k).isConnected(0, 1,0))
                    {
                        uy = - (c0 * ccvel_fab(i,j  ,k,0)
                              + c1 * ccvel_fab(i,j-1,k,0)
                              + c2 * ccvel_fab(i,j-2,k,0)) * idy;
                        wy = - (c0 * ccvel_fab(i,j  ,k,2)
                              + c1 * ccvel_fab(i,j-1,k,2)
                              + c2 * ccvel_fab(i,j-2,k,2)) * idy;
                    }
                    else if (!flag_fab(i,j,k).isConnected(0,-1,0))
                    {
                        uy = (c0 * ccvel_fab(i,j  ,k,0)
                            + c1 * ccvel_fab(i,j+1,k,0)
                            + c2 * ccvel_fab(i,j+2,k,0)) * idy;
                        wy = (c0 * ccvel_fab(i,j  ,k,2)
                            + c1 * ccvel_fab(i,j+1,k,2)
                            + c2 * ccvel_fab(i,j+2,k,2)) * idy;
                    }
                    else
                    {
                        uy = Real(0.5) * (ccvel_fab(i,j+1,k,0) - ccvel_fab(i,j-1,k,0)) * idy;
                        wy = Real(0.5) * (ccvel_fab(i,j+1,k,2) - ccvel_fab(i,j-1,k,2)) * idy;
                    }
                    // Do the same in z-direction
                    if (!flag_fab(i,j,k).isConnected(0,0, 1))
                    {
                        uz = - (c0 * ccvel_fab(i,j,k  ,0)
                              + c1 * ccvel_fab(i,j,k-1,0)
                              + c2 * ccvel_fab(i,j,k-2,0)) * idz;
                        vz = - (c0 * ccvel_fab(i,j,k  ,1)
                              + c1 * ccvel_fab(i,j,k-1,1)
                              + c2 * ccvel_fab(i,j,k-2,1)) * idz;
                    }
                    else if (!flag_fab(i,j,k).isConnected(0,0,-1))
                    {
                        uz = (c0 * ccvel_fab(i,j,k  ,0)
                            + c1 * ccvel_fab(i,j,k+1,0)
                            + c2 * ccvel_fab(i,j,k+2,0)) * idz;
                        vz = (c0 * ccvel_fab(i,j,k  ,1)
                            + c1 * ccvel_fab(i,j,k+1,1)
                            + c2 * ccvel_fab(i,j,k+2,1)) * idz;
                    }
                    else
                    {
                        uz = Real(0.5) * (ccvel_fab(i,j,k+1,0) - ccvel_fab(i,j,k-1,0)) * idz;
                        vz = Real(0.5) * (ccvel_fab(i,j,k+1,1) - ccvel_fab(i,j,k-1,1)) * idz;
                    }
                    vort_fab(i,j,k) = std::sqrt((wy-vz)*(wy-vz) + (uz-wx)*(uz-wx) + (vx-uy)*(vx-uy));
                }
            });
        }
        else
#endif
        {
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real vx = Real(0.5) * (ccvel_fab(i+1,j,k,1) - ccvel_fab(i-1,j,k,1)) * idx;
                Real wx = Real(0.5) * (ccvel_fab(i+1,j,k,2) - ccvel_fab(i-1,j,k,2)) * idx;

                Real uy = Real(0.5) * (ccvel_fab(i,j+1,k,0) - ccvel_fab(i,j-1,k,0)) * idy;
                Real wy = Real(0.5) * (ccvel_fab(i,j+1,k,2) - ccvel_fab(i,j-1,k,2)) * idy;

                Real uz = Real(0.5) * (ccvel_fab(i,j,k+1,0) - ccvel_fab(i,j,k-1,0)) * idz;
                Real vz = Real(0.5) * (ccvel_fab(i,j,k+1,1) - ccvel_fab(i,j,k-1,1)) * idz;

                vort_fab(i,j,k) = std::sqrt((wy-vz)*(wy-vz) + (uz-wx)*(uz-wx) + (vx-uy)*(vx-uy));
            });
        }
    }
}
#endif

void incflo::ComputeDrag()
{
#if 0
    BL_PROFILE("incflo::ComputeDrag");

    // Coefficients for one-sided difference estimation
    Real c0 = -1.5;
    Real c1 = 2.0;
    Real c2 = -0.5;

    for(int lev = 0; lev <= finest_level; lev++)
    {
        Box domain(geom[lev].Domain());
        Real dx = geom[lev].CellSize()[0];

        drag[lev]->setVal(0.0);

#ifdef AMREX_USE_EB
        // Get EB geometric info
        const amrex::MultiCutFab* bndryarea;
        const amrex::MultiCutFab* bndrynorm;
        bndryarea = &(ebfactory[lev]->getBndryArea());
        bndrynorm = &(ebfactory[lev]->getBndryNormal());

#ifdef _OPENMP
#pragma omp parallel for reduction(+:drag) if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi(*vel[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            // Tilebox
            Box bx = mfi.tilebox();

            // This is to check efficiently if this tile contains any eb stuff
            const EBFArrayBox& vel_fab = static_cast<EBFArrayBox const&>((*vel[lev])[mfi]);
            const EBCellFlagFab& flags = vel_fab.getEBCellFlagFab();

            if (flags.getType(bx) == FabType::singlevalued)
            {
                const auto& drag_arr = drag[lev]->array(mfi);
                const auto& vel_arr = vel[lev]->array(mfi);
                const auto& eta_arr = eta[lev]->array(mfi);
                const auto& p_arr = p[lev]->array(mfi);
                const auto& bndryarea_arr = bndryarea->array(mfi);
                const auto& bndrynorm_arr = bndrynorm->array(mfi);
                const auto& flag_fab = flags.array();

                for(int i = bx.smallEnd(0); i <= bx.bigEnd(0); i++)
                for(int j = bx.smallEnd(1); j <= bx.bigEnd(1); j++)
                for(int k = bx.smallEnd(2); k <= bx.bigEnd(2); k++)
                {
                    if(flag_fab(i,j,k).isSingleValued())
                    {
                        Real area = bndryarea_arr(i,j,k);
                        Real nx = bndrynorm_arr(i,j,k,0);
                        Real ny = bndrynorm_arr(i,j,k,1);
                        Real nz = bndrynorm_arr(i,j,k,2);

                        Real uz, vz, wx, wy, wz;

                        if (!flag_fab(i,j,k).isConnected(0,0, 1))
                        {
                            uz = - (c0 * vel_arr(i,j,k,0) + c1 * vel_arr(i,j,k-1,0) + c2 * vel_arr(i,j,k-2,0)) / dx;
                            vz = - (c0 * vel_arr(i,j,k,1) + c1 * vel_arr(i,j,k-1,1) + c2 * vel_arr(i,j,k-2,1)) / dx;
                            wz = - (c0 * vel_arr(i,j,k,2) + c1 * vel_arr(i,j,k-1,2) + c2 * vel_arr(i,j,k-2,2)) / dx;
                        }
                        else if (!flag_fab(i,j,k).isConnected(0,0,-1))
                        {
                            uz = (c0 * vel_arr(i,j,k,0) + c1 * vel_arr(i,j,k+1,0) + c2 * vel_arr(i,j,k+2,0)) / dx;
                            vz = (c0 * vel_arr(i,j,k,1) + c1 * vel_arr(i,j,k+1,1) + c2 * vel_arr(i,j,k+2,1)) / dx;
                            wz = (c0 * vel_arr(i,j,k,2) + c1 * vel_arr(i,j,k+1,2) + c2 * vel_arr(i,j,k+2,2)) / dx;
                        }
                        else
                        {
                            uz = 0.5 * (vel_arr(i,j,k+1,0) - vel_arr(i,j,k-1,0)) / dx;
                            vz = 0.5 * (vel_arr(i,j,k+1,1) - vel_arr(i,j,k-1,1)) / dx;
                            wz = 0.5 * (vel_arr(i,j,k+1,2) - vel_arr(i,j,k-1,2)) / dx;
                        }

                        if (!flag_fab(i,j,k).isConnected(0, 1,0))
                        {
                            wy = - (c0 * vel_arr(i,j,k,2) + c1 * vel_arr(i,j-1,k,2) + c2 * vel_arr(i,j-2,k,2)) / dx;
                        }
                        else if (!flag_fab(i,j,k).isConnected(0,-1,0))
                        {
                            wy = (c0 * vel_arr(i,j,k,2) + c1 * vel_arr(i,j+1,k,2) + c2 * vel_arr(i,j+2,k,2)) / dx;
                        }
                        else
                        {
                            wy = 0.5 * (vel_arr(i,j+1,k,2) - vel_arr(i,j-1,k,2)) / dx;
                        }

                        if (!flag_fab(i,j,k).isConnected( 1,0,0))
                        {
                            wx = - (c0 * vel_arr(i,j,k,2) + c1 * vel_arr(i-1,j,k,2) + c2 * vel_arr(i-2,j,k,2)) / dx;
                        }
                        else if (!flag_fab(i,j,k).isConnected(-1,0,0))
                        {
                            wx = (c0 * vel_arr(i,j,k,2) + c1 * vel_arr(i+1,j,k,2) + c2 * vel_arr(i+2,j,k,2)) / dx;
                        }
                        else
                        {
                            wx = 0.5 * (vel_arr(i+1,j,k,2) - vel_arr(i-1,j,k,2)) / dx;
                        }

                        Real p_contrib = p_arr(i,j,k) * nz;
                        Real tau_contrib = - eta_arr(i,j,k) * ( (uz + wx) * nx + (vz + wy) * ny + (wz + wz) * nz );

                        // TODO: Get values on EB centroid,
                        //       not the default CC and nodal values
                        drag_arr(i,j,k) = (p_contrib + tau_contrib) * area * dx * dx;
                    }
                    else
                    {
                        drag_arr(i,j,k) = 0.0;
                    }
                }
            }
        } // MFIter
#endif
    }
#endif
}
