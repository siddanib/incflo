#include <incflo.H>

using namespace amrex;

#ifdef AMREX_USE_EB
void
incflo::incflo_correct_small_cells (Vector<MultiFab*      > const& vel_in,
                                    AMREX_D_DECL(Vector<MultiFab const*> const& u_mac,
                                                 Vector<MultiFab const*> const& v_mac,
                                                 Vector<MultiFab const*> const& w_mac))
{
    BL_PROFILE("incflo::incflo_correct_small_cells");

    for (int lev = 0; lev <= finest_level; lev++)
    {
       for (MFIter mfi(*vel_in[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
       {
          // Tilebox
          const Box bx = mfi.tilebox();

          EBCellFlagFab const& flags = EBFactory(lev).getMultiEBCellFlagFab()[mfi];

          // Face-centered velocity components
          AMREX_D_TERM(const auto& umac_fab = (u_mac[lev])->array(mfi);,
                       const auto& vmac_fab = (v_mac[lev])->array(mfi);,
                       const auto& wmac_fab = (w_mac[lev])->array(mfi););

          if (flags.getType(amrex::grow(bx,0)) == FabType::covered )
          {
            // do nothing
          }

          // No cut cells in this FAB
          else if (flags.getType(amrex::grow(bx,1)) == FabType::regular )
          {
            // do nothing
          }

          // Cut cells in this FAB
          else
          {
             // Face-centered areas
             AMREX_D_TERM(const auto& apx_fab   = EBFactory(lev).getAreaFrac()[0]->const_array(mfi);,
                          const auto& apy_fab   = EBFactory(lev).getAreaFrac()[1]->const_array(mfi);,
                          const auto& apz_fab   = EBFactory(lev).getAreaFrac()[2]->const_array(mfi););

             const auto& vfrac_fab = EBFactory(lev).getVolFrac().const_array(mfi);

             const auto& ccvel_fab = vel_in[lev]->array(mfi);
             const Real repair_vfrac = m_eb_ccvel_repair_vfrac;

             if (!m_eb_flow.enabled) {
                // This FAB has cut cells -- we define the centroid value in terms of the MAC velocities onfaces
                ParallelFor(bx,
                  [vfrac_fab,repair_vfrac,
                   AMREX_D_DECL(apx_fab,apy_fab,apz_fab),ccvel_fab,
                   AMREX_D_DECL(umac_fab,vmac_fab,wmac_fab)]
                  AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real vfrac = vfrac_fab(i,j,k);
                    if (vfrac > 0.0_rt && vfrac < repair_vfrac)
                    {
                       AMREX_D_TERM(Real u_avg = (apx_fab(i,j,k) * umac_fab(i,j,k) + apx_fab(i+1,j,k) * umac_fab(i+1,j,k))
                                               / (apx_fab(i,j,k) + apx_fab(i+1,j,k));,
                                    Real v_avg = (apy_fab(i,j,k) * vmac_fab(i,j,k) + apy_fab(i,j+1,k) * vmac_fab(i,j+1,k))
                                               / (apy_fab(i,j,k) + apy_fab(i,j+1,k));,
                                    Real w_avg = (apz_fab(i,j,k) * wmac_fab(i,j,k) + apz_fab(i,j,k+1) * wmac_fab(i,j,k+1))
                                               / (apz_fab(i,j,k) + apz_fab(i,j,k+1)););

                       AMREX_D_TERM(
                         Real u_old = ccvel_fab(i,j,k,0);
                         Real u_lo = amrex::min(u_avg, 0.0_rt);
                         Real u_hi = amrex::max(u_avg, 0.0_rt);
                         if (u_old < u_lo || u_old > u_hi) {
                             ccvel_fab(i,j,k,0) = amrex::Clamp(u_old, u_lo, u_hi);
                         }
                         ,
                         Real v_old = ccvel_fab(i,j,k,1);
                         Real v_lo = amrex::min(v_avg, 0.0_rt);
                         Real v_hi = amrex::max(v_avg, 0.0_rt);
                         if (v_old < v_lo || v_old > v_hi) {
                             ccvel_fab(i,j,k,1) = amrex::Clamp(v_old, v_lo, v_hi);
                         }
                         ,
                         Real w_old = ccvel_fab(i,j,k,2);
                         Real w_lo = amrex::min(w_avg, 0.0_rt);
                         Real w_hi = amrex::max(w_avg, 0.0_rt);
                         if (w_old < w_lo || w_old > w_hi) {
                             ccvel_fab(i,j,k,2) = amrex::Clamp(w_old, w_lo, w_hi);
                         });

                    }
                });
             } else { // EB has flow
                Array4<Real const> const& eb_vel = get_velocity_eb()[lev]->const_array(mfi);

                // This FAB has cut cells -- we define the centroid value in terms of the MAC velocities onfaces
                ParallelFor(bx,
                  [vfrac_fab,repair_vfrac,
                   AMREX_D_DECL(apx_fab,apy_fab,apz_fab),ccvel_fab,
                   AMREX_D_DECL(umac_fab,vmac_fab,wmac_fab),eb_vel]
                  AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real vfrac = vfrac_fab(i,j,k);
                    if (vfrac > 0.0_rt && vfrac < repair_vfrac)
                    {
                       AMREX_D_TERM(const Real ucc = ccvel_fab(i,j,k,0);,
                                    const Real vcc = ccvel_fab(i,j,k,1);,
                                    const Real wcc = ccvel_fab(i,j,k,2););

                       AMREX_D_TERM(Real u_avg = (apx_fab(i,j,k) * umac_fab(i,j,k) + apx_fab(i+1,j,k) * umac_fab(i+1,j,k))
                                               / (apx_fab(i,j,k) + apx_fab(i+1,j,k));,
                                    Real v_avg = (apy_fab(i,j,k) * vmac_fab(i,j,k) + apy_fab(i,j+1,k) * vmac_fab(i,j+1,k))
                                               / (apy_fab(i,j,k) + apy_fab(i,j+1,k));,
                                    Real w_avg = (apz_fab(i,j,k) * wmac_fab(i,j,k) + apz_fab(i,j,k+1) * wmac_fab(i,j,k+1))
                                               / (apz_fab(i,j,k) + apz_fab(i,j,k+1)););


                       AMREX_D_TERM(ccvel_fab(i,j,k,0) = u_avg;,
                                    ccvel_fab(i,j,k,1) = v_avg;,
                                    ccvel_fab(i,j,k,2) = w_avg;);

                       AMREX_D_TERM(const Real u_eb = eb_vel(i,j,k,0);,
                                    const Real v_eb = eb_vel(i,j,k,1);,
                                    const Real w_eb = eb_vel(i,j,k,2););
#if (AMREX_SPACEDIM == 2)
                       const Real eb_vel_mag = std::sqrt(u_eb*u_eb + v_eb*v_eb);
#elif (AMREX_SPACEDIM == 3)
                       const Real eb_vel_mag = std::sqrt(u_eb*u_eb + v_eb*v_eb + w_eb*w_eb);
#endif
                       constexpr amrex::Real tolerance = std::numeric_limits<amrex::Real>::epsilon();

                       if (eb_vel_mag > tolerance) {
                          if ( u_eb > tolerance ) {
                              ccvel_fab(i,j,k,0) = amrex::min(amrex::max(ucc, u_avg), u_eb);
                          } else if( u_eb < -tolerance) {
                              ccvel_fab(i,j,k,0) = amrex::max(amrex::min(ucc, u_avg), u_eb);
                          }
                          if ( v_eb > tolerance ) {
                              ccvel_fab(i,j,k,1) = amrex::min(amrex::max(vcc, v_avg), v_eb);
                          } else if( v_eb < -tolerance) {
                              ccvel_fab(i,j,k,1) = amrex::max(amrex::min(vcc, v_avg), v_eb);
                          }
#if (AMREX_SPACEDIM == 3)
                          if ( w_eb > tolerance ) {
                              ccvel_fab(i,j,k,2) = amrex::min(amrex::max(wcc, w_avg), w_eb);
                          } else if( w_eb < -tolerance) {
                              ccvel_fab(i,j,k,2) = amrex::max(amrex::min(wcc, w_avg), w_eb);
                          }
#endif
                       } else {
                          AMREX_D_TERM(
                            Real u_lo = amrex::min(u_avg, 0.0_rt);
                            Real u_hi = amrex::max(u_avg, 0.0_rt);
                            if (ucc < u_lo || ucc > u_hi) {
                                ccvel_fab(i,j,k,0) = amrex::Clamp(ucc, u_lo, u_hi);
                            }
                            ,
                            Real v_lo = amrex::min(v_avg, 0.0_rt);
                            Real v_hi = amrex::max(v_avg, 0.0_rt);
                            if (vcc < v_lo || vcc > v_hi) {
                                ccvel_fab(i,j,k,1) = amrex::Clamp(vcc, v_lo, v_hi);
                            }
                            ,
                            Real w_lo = amrex::min(w_avg, 0.0_rt);
                            Real w_hi = amrex::max(w_avg, 0.0_rt);
                            if (wcc < w_lo || wcc > w_hi) {
                                ccvel_fab(i,j,k,2) = amrex::Clamp(wcc, w_lo, w_hi);
                            });
                       }
                    }
                });
             }
          }
       }
    }
}
#endif
