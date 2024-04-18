#ifndef DERIVE_K_H_
#define DERIVE_K_H_

#include <AMReX_FArrayBox.H>
#include <cmath>

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real incflo_strainrate (int i, int j, int k,
                               AMREX_D_DECL(amrex::Real idx, amrex::Real idy, amrex::Real idz),
                               amrex::Array4<amrex::Real const> const& vel) noexcept
{
    using namespace amrex;

#if (AMREX_SPACEDIM == 3)
    Real ux = amrex::Real(0.5) * (vel(i+1,j,k,0) - vel(i-1,j,k,0)) * idx;
    Real vx = amrex::Real(0.5) * (vel(i+1,j,k,1) - vel(i-1,j,k,1)) * idx;
    Real wx = amrex::Real(0.5) * (vel(i+1,j,k,2) - vel(i-1,j,k,2)) * idx;

    Real uy = amrex::Real(0.5) * (vel(i,j+1,k,0) - vel(i,j-1,k,0)) * idy;
    Real vy = amrex::Real(0.5) * (vel(i,j+1,k,1) - vel(i,j-1,k,1)) * idy;
    Real wy = amrex::Real(0.5) * (vel(i,j+1,k,2) - vel(i,j-1,k,2)) * idy;

    Real uz = amrex::Real(0.5) * (vel(i,j,k+1,0) - vel(i,j,k-1,0)) * idz;
    Real vz = amrex::Real(0.5) * (vel(i,j,k+1,1) - vel(i,j,k-1,1)) * idz;
    Real wz = amrex::Real(0.5) * (vel(i,j,k+1,2) - vel(i,j,k-1,2)) * idz;

    return std::sqrt(amrex::Real(2.0) * ux*ux + amrex::Real(2.0) * vy*vy + amrex::Real(2.0) * wz*wz
                     + (uy+vx)*(uy+vx) + (vz+wy)*(vz+wy) + (wx+uz)*(wx+uz));
#else
    Real ux = amrex::Real(0.5) * (vel(i+1,j,k,0) - vel(i-1,j,k,0)) * idx;
    Real vx = amrex::Real(0.5) * (vel(i+1,j,k,1) - vel(i-1,j,k,1)) * idx;

    Real uy = amrex::Real(0.5) * (vel(i,j+1,k,0) - vel(i,j-1,k,0)) * idy;
    Real vy = amrex::Real(0.5) * (vel(i,j+1,k,1) - vel(i,j-1,k,1)) * idy;

    return std::sqrt(amrex::Real(2.0) * ux*ux + amrex::Real(2.0) * vy*vy
                     + (uy+vx)*(uy+vx));
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real incflo_strainrate_nd (int i, int j, int k,
             AMREX_D_DECL(amrex::Real idx, amrex::Real idy, amrex::Real idz),
             amrex::Array4<amrex::Real const> const& vel,
             const amrex::Dim3& dlo, const amrex::Dim3& dhi,
             amrex::GpuArray<amrex::GpuArray<int,2>,AMREX_SPACEDIM> bc_type) noexcept
{
    using namespace amrex;

#if (AMREX_SPACEDIM == 3)
    Real uxpp = (vel(i,j  ,k  ,0) - vel(i-1,j  ,k  ,0)) * idx;
    Real uxpm = (vel(i,j-1,k  ,0) - vel(i-1,j-1,k  ,0)) * idx;
    Real uxmp = (vel(i,j  ,k-1,0) - vel(i-1,j  ,k-1,0)) * idx;
    Real uxmm = (vel(i,j-1,k-1,0) - vel(i-1,j-1,k-1,0)) * idx;
    Real vxpp = (vel(i,j  ,k  ,1) - vel(i-1,j  ,k  ,1)) * idx;
    Real vxpm = (vel(i,j-1,k  ,1) - vel(i-1,j-1,k  ,1)) * idx;
    Real vxmp = (vel(i,j  ,k-1,1) - vel(i-1,j  ,k-1,1)) * idx;
    Real vxmm = (vel(i,j-1,k-1,1) - vel(i-1,j-1,k-1,1)) * idx;
    Real wxpp = (vel(i,j  ,k  ,2) - vel(i-1,j  ,k  ,2)) * idx;
    Real wxpm = (vel(i,j-1,k  ,2) - vel(i-1,j-1,k  ,2)) * idx;
    Real wxmp = (vel(i,j  ,k-1,2) - vel(i-1,j  ,k-1,2)) * idx;
    Real wxmm = (vel(i,j-1,k-1,2) - vel(i-1,j-1,k-1,2)) * idx;

    Real uypp = (vel(i  ,j,k  ,0) - vel(i  ,j-1,k  ,0)) * idy;
    Real uypm = (vel(i-1,j,k  ,0) - vel(i-1,j-1,k  ,0)) * idy;
    Real uymp = (vel(i  ,j,k-1,0) - vel(i  ,j-1,k-1,0)) * idy;
    Real uymm = (vel(i-1,j,k-1,0) - vel(i-1,j-1,k-1,0)) * idy;
    Real vypp = (vel(i  ,j,k  ,1) - vel(i  ,j-1,k  ,1)) * idy;
    Real vypm = (vel(i-1,j,k  ,1) - vel(i-1,j-1,k  ,1)) * idy;
    Real vymp = (vel(i  ,j,k-1,1) - vel(i  ,j-1,k-1,1)) * idy;
    Real vymm = (vel(i-1,j,k-1,1) - vel(i-1,j-1,k-1,1)) * idy;
    Real wypp = (vel(i  ,j,k  ,2) - vel(i  ,j-1,k  ,2)) * idy;
    Real wypm = (vel(i-1,j,k  ,2) - vel(i-1,j-1,k  ,2)) * idy;
    Real wymp = (vel(i  ,j,k-1,2) - vel(i  ,j-1,k-1,2)) * idy;
    Real wymm = (vel(i-1,j,k-1,2) - vel(i-1,j-1,k-1,2)) * idy;

    Real uzpp = (vel(i  ,j  ,k,0) - vel(i  ,j  ,k-1,0)) * idz;
    Real uzpm = (vel(i-1,j  ,k,0) - vel(i-1,j  ,k-1,0)) * idz;
    Real uzmp = (vel(i  ,j-1,k,0) - vel(i  ,j-1,k-1,0)) * idz;
    Real uzmm = (vel(i-1,j-1,k,0) - vel(i-1,j-1,k-1,0)) * idz;
    Real vzpp = (vel(i  ,j  ,k,1) - vel(i  ,j  ,k-1,1)) * idz;
    Real vzpm = (vel(i-1,j  ,k,1) - vel(i-1,j  ,k-1,1)) * idz;
    Real vzmp = (vel(i  ,j-1,k,1) - vel(i  ,j-1,k-1,1)) * idz;
    Real vzmm = (vel(i-1,j-1,k,1) - vel(i-1,j-1,k-1,1)) * idz;
    Real wzpp = (vel(i  ,j  ,k,2) - vel(i  ,j  ,k-1,2)) * idz;
    Real wzpm = (vel(i-1,j  ,k,2) - vel(i-1,j  ,k-1,2)) * idz;
    Real wzmp = (vel(i  ,j-1,k,2) - vel(i  ,j-1,k-1,2)) * idz;
    Real wzmm = (vel(i-1,j-1,k,2) - vel(i-1,j-1,k-1,2)) * idz;

    // Applying Boundary Conditions
    // First ghost cells of boundary are populated
    // with domain boundary value
    int idim, jdim;
    if ((i == dlo.x) or (i == dhi.x+1) ){
        idim = 0;
        jdim = (i == dlo.x)? 0:1;
        if (bc_type[idim][jdim] == 2) {
            uxpp *= Real(2.0); vxpp *= Real(2.0); wxpp *= Real(2.0);
            uxpm *= Real(2.0); vxpm *= Real(2.0); wxpm *= Real(2.0);
            uxmp *= Real(2.0); vxmp *= Real(2.0); wxmp *= Real(2.0);
            uxmm *= Real(2.0); vxmm *= Real(2.0); wxmm *= Real(2.0);
        }
        else if (bc_type[idim][jdim] == 1) {
            uxpp *= Real(2.0); vxpp = Real(0.0); wxpp = Real(0.0);
            uxpm *= Real(2.0); vxpm = Real(0.0); wxpm = Real(0.0);
            uxmp *= Real(2.0); vxmp = Real(0.0); wxmp = Real(0.0);
            uxmm *= Real(2.0); vxmm = Real(0.0); wxmm = Real(0.0);
        }
    }

    if ((j == dlo.y) or (j == dhi.y+1)){
        idim = 1;
        jdim = (j == dlo.y)? 0:1;
        if (bc_type[idim][jdim] == 2) {
            uypp *= Real(2.0); vypp *= Real(2.0); wypp *= Real(2.0);
            uypm *= Real(2.0); vypm *= Real(2.0); wypm *= Real(2.0);
            uymp *= Real(2.0); vymp *= Real(2.0); wymp *= Real(2.0);
            uymm *= Real(2.0); vymm *= Real(2.0); wymm *= Real(2.0);
        }
        else if (bc_type[idim][jdim] == 1) {
            uypp = Real(0.0); vypp *= Real(2.0); wypp = Real(0.0);
            uypm = Real(0.0); vypm *= Real(2.0); wypm = Real(0.0);
            uymp = Real(0.0); vymp *= Real(2.0); wymp = Real(0.0);
            uymm = Real(0.0); vymm *= Real(2.0); wymm = Real(0.0);
        }
    }

    if ((k == dlo.z) or (k == dhi.z+1)) {
        idim = 2;
        jdim = (k == dlo.z)? 0:1;
        if (bc_type[idim][jdim] == 2) {
            uzpp *= Real(2.0); vzpp *= Real(2.0); wzpp *= Real(2.0);
            uzpm *= Real(2.0); vzpm *= Real(2.0); wzpm *= Real(2.0);
            uzmp *= Real(2.0); vzmp *= Real(2.0); wzmp *= Real(2.0);
            uzmm *= Real(2.0); vzmm *= Real(2.0); wzmm *= Real(2.0);
        }
        else if (bc_type[idim][jdim] == 1) {
            uzpp = Real(0.0); vzpp = Real(0.0); wzpp *= Real(2.0);
            uzpm = Real(0.0); vzpm = Real(0.0); wzpm *= Real(2.0);
            uzmp = Real(0.0); vzmp = Real(0.0); wzmp *= Real(2.0);
            uzmm = Real(0.0); vzmm = Real(0.0); wzmm *= Real(2.0);
        }
    }

    // Corner cases
    // Slip walls are opposite of no-slip walls
    if ((i == dlo.x) or (i == dhi.x+1)) {
        idim = 0;
        jdim = (i == dlo.x)? 0:1;
        if (bc_type[idim][jdim] == 2) {
            if (jdim == 0) {
                uypp = uypm; uymp = uymm;
                vypp = vypm; vymp = vymm;
                wypp = wypm; wymp = wymm;

                uzpp = uzpm; uzmp = uzmm;
                vzpp = vzpm; vzmp = vzmm;
                wzpp = wzpm; wzmp = wzmm;
            }
            else {
                uypm = uypp; uymm = uymp;
                vypm = vypp; vymm = vymp;
                wypm = wypp; wymm = wymp;

                uzpm = uzpp; uzmm = uzmp;
                vzpm = vzpp; vzmm = vzmp;
                wzpm = wzpp; wzmm = wzmp;
            }
        }
        else if (bc_type[idim][jdim] == 1) {
            if (jdim == 0) {
                uypm = uypp; uymm = uymp;
                vypm = vypp; vymm = vymp;
                wypm = wypp; wymm = wymp;

                uzpm = uzpp; uzmm = uzmp;
                vzpm = vzpp; vzmm = vzmp;
                wzpm = wzpp; wzmm = wzmp;
            }
            else {
                uypp = uypm; uymp = uymm;
                vypp = vypm; vymp = vymm;
                wypp = wypm; wymp = wymm;

                uzpp = uzpm; uzmp = uzmm;
                vzpp = vzpm; vzmp = vzmm;
                wzpp = wzpm; wzmp = wzmm;
            }
        }
    }

    if ((j == dlo.y) or (j == dhi.y+1)) {
        idim = 1;
        jdim = (j == dlo.y)? 0:1;
        if (bc_type[idim][jdim] == 2) {
            if (jdim == 0) {
                uxpp = uxpm; uxmp = uxmm;
                vxpp = vxpm; vxmp = vxmm;
                wxpp = wxpm; wxmp = wxmm;

                uzpp = uzmp; uzpm = uzmm;
                vzpp = vzmp; vzpm = vzmm;
                wzpp = wzmp; wzpm = wzmm;
            }
            else {
                uxpm = uxpp; uxmm = uxmp;
                vxpm = vxpp; vxmm = vxmp;
                wxpm = wxpp; wxmm = wxmp;

                uzmp = uzpp; uzmm = uzpm;
                vzmp = vzpp; vzmm = vzpm;
                wzmp = wzpp; wzmm = wzpm;
            }
        }
        else if (bc_type[idim][jdim] == 1) {
            if (jdim == 0) {
                uxpm = uxpp; uxmm = uxmp;
                vxpm = vxpp; vxmm = vxmp;
                wxpm = wxpp; wxmm = wxmp;

                uzmp = uzpp; uzmm = uzpm;
                vzmp = vzpp; vzmm = vzpm;
                wzmp = wzpp; wzmm = wzpm;
            }
            else {
                uxpp = uxpm; uxmp = uxmm;
                vxpp = vxpm; vxmp = vxmm;
                wxpp = wxpm; wxmp = wxmm;

                uzpp = uzmp; uzpm = uzmm;
                vzpp = vzmp; vzpm = vzmm;
                wzpp = wzmp; wzpm = wzmm;
            }
        }
    }

    if ((k == dlo.z) or (k == dhi.z+1)) {
        idim = 2;
        jdim = (k == dlo.z)? 0:1;
        if (bc_type[idim][jdim] == 2) {
            if (jdim == 0) {
                uxpp = uxmp; uxpm = uxmm;
                vxpp = vxmp; vxpm = vxmm;
                wxpp = wxmp; wxpm = wxmm;

                uypp = uymp; uypm = uymm;
                vypp = vymp; vypm = vymm;
                wypp = wymp; wypm = wymm;
            }
            else {
                uxmp = uxpp; uxmm = uxpm;
                vxmp = vxpp; vxmm = vxpm;
                wxmp = wxpp; wxmm = wxpm;

                uymp = uypp; uymm = uypm;
                vymp = vypp; vymm = vypm;
                wymp = wypp; wymm = wypm;
            }
        }
        else if(bc_type[idim][jdim] == 1) {
            if (jdim == 0) {
                uxmp = uxpp; uxmm = uxpm;
                vxmp = vxpp; vxmm = vxpm;
                wxmp = wxpp; wxmm = wxpm;

                uymp = uypp; uymm = uypm;
                vymp = vypp; vymm = vypm;
                wymp = wypp; wymm = wypm;
            }
            else {
                uxpp = uxmp; uxpm = uxmm;
                vxpp = vxmp; vxpm = vxmm;
                wxpp = wxmp; wxpm = wxmm;

                uypp = uymp; uypm = uymm;
                vypp = vymp; vypm = vymm;
                wypp = wymp; wypm = wymm;
            }
        }
    }

    Real ux = Real(0.25) * (uxpp + uxpm + uxmp + uxmm);
    Real uy = Real(0.25) * (uypp + uypm + uymp + uymm);
    Real uz = Real(0.25) * (uzpp + uzpm + uzmp + uzmm);

    Real vx = Real(0.25) * (vxpp + vxpm + vxmp + vxmm);
    Real vy = Real(0.25) * (vypp + vypm + vymp + vymm);
    Real vz = Real(0.25) * (vzpp + vzpm + vzmp + vzmm);

    Real wx = Real(0.25) * (wxpp + wxpm + wxmp + wxmm);
    Real wy = Real(0.25) * (wypp + wypm + wymp + wymm);
    Real wz = Real(0.25) * (wzpp + wzpm + wzmp + wzmm);

    return std::sqrt(Real(2.0) * ux*ux + Real(2.0) * vy*vy + Real(2.0) * wz*wz
                     + (uy+vx)*(uy+vx) + (vz+wy)*(vz+wy) + (wx+uz)*(wx+uz));
#else
    Real uxp = (vel(i,j  ,k,0) - vel(i-1,j  ,k,0)) * idx;
    Real uxm = (vel(i,j-1,k,0) - vel(i-1,j-1,k,0)) * idx;
    Real vxp = (vel(i,j  ,k,1) - vel(i-1,j  ,k,1)) * idx;
    Real vxm = (vel(i,j-1,k,1) - vel(i-1,j-1,k,1)) * idx;

    Real uyp = (vel(i  ,j,k,0) - vel(i  ,j-1,k,0)) * idy;
    Real uym = (vel(i-1,j,k,0) - vel(i-1,j-1,k,0)) * idy;
    Real vyp = (vel(i  ,j,k,1) - vel(i  ,j-1,k,1)) * idy;
    Real vym = (vel(i-1,j,k,1) - vel(i-1,j-1,k,1)) * idy;

    // Applying Boundary Conditions
    // First ghost cells of boundary are populated
    // with domain boundary value
    int idim, jdim;
    if ((i == dlo.x) or (i == dhi.x+1)) {
        idim = 0;
        jdim = (i == dlo.x)? 0:1;
        if (bc_type[idim][jdim] == 2) {
            uxp *= Real(2.0); vxp *= Real(2.0);
            uxm *= Real(2.0); vxm *= Real(2.0);
        }
        else if (bc_type[idim][jdim] == 1) {
            uxp *= Real(2.0); vxp = Real(0.0);
            uxm *= Real(2.0); vxm = Real(0.0);
        }
    }

    if ((j == dlo.y) or (j == dhi.y+1)) {
        idim = 1;
        jdim = (j == dlo.y)? 0:1;
        if (bc_type[idim][jdim] == 2) {
            uyp *= Real(2.0); vyp *= Real(2.0);
            uym *= Real(2.0); vym *= Real(2.0);
        }
        else if (bc_type[idim][jdim] == 1) {
            uyp = Real(0.0); vyp *= Real(2.0);
            uym = Real(0.0); vym *= Real(2.0);
        }
    }
    // Corner cases
    // Slip walls are opposite of no-slip walls
    if ((i == dlo.x) or (i == dhi.x+1)) {
        idim = 0;
        jdim = (i == dlo.x)? 0:1;
        if (bc_type[idim][jdim] == 2) {
            if (jdim == 0) {
                uyp = uym;
                vyp = vym;
            }
            else {
                uym = uyp;
                vym = vyp;
            }
        }
        else if (bc_type[idim][jdim] == 1) {
            if (jdim == 0) {
                uym = uyp;
                vym = vyp;
            }
            else {
                uyp = uym;
                vyp = vym;
            }
        }
    }

    if ((j == dlo.y) or (j == dhi.y+1)) {
        idim = 1;
        jdim = (j == dlo.y)? 0:1;
        if (bc_type[idim][jdim] == 2) {
            if (jdim == 0) {
                uxp = uxm;
                vxp = vxm;
            }
            else {
                uxm = uxp;
                vxm = vxp;
            }
        }
        else if (bc_type[idim][jdim] == 1) {
            if (jdim == 0) {
                uxm = uxp;
                vxm = vxp;
            }
            else {
                uxp = uxm;
                vxp = vxm;
            }
        }
    }

    Real ux = Real(0.5)*(uxp + uxm);
    Real uy = Real(0.5)*(uyp + uym);
    Real vx = Real(0.5)*(vxp + vxm);
    Real vy = Real(0.5)*(vyp + vym);

    return std::sqrt(Real(2.0) * ux*ux + Real(2.0) * vy*vy
                     + (uy+vx)*(uy+vx));
#endif
}

#ifdef AMREX_USE_EB
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real incflo_strainrate_eb (int i, int j, int k,
                                  AMREX_D_DECL(amrex::Real idx, amrex::Real idy, amrex::Real idz),
                                  amrex::Array4<amrex::Real const> const& vel,
                                  amrex::EBCellFlag flag) noexcept
{
    using namespace amrex;

    if (flag.isRegular()) {
        return incflo_strainrate(i,j,k,AMREX_D_DECL(idx,idy,idz),vel);
    } else if (flag.isSingleValued()) {
        constexpr Real c0 = -1.5;
        constexpr Real c1 =  amrex::Real(2.0);
        constexpr Real c2 = -amrex::Real(0.5);
        Real ux, vx, uy, vy;
#if (AMREX_SPACEDIM == 3)
        Real wx, wy, uz, vz, wz;
#endif
        if (!flag.isConnected( 1,0,0))
        {
            // Covered cell to the right, go fish left
            ux = - (c0 * vel(i  ,j,k,0)
                  + c1 * vel(i-1,j,k,0)
                  + c2 * vel(i-2,j,k,0)) * idx;
            vx = - (c0 * vel(i  ,j,k,1)
                  + c1 * vel(i-1,j,k,1)
                  + c2 * vel(i-2,j,k,1)) * idx;
#if (AMREX_SPACEDIM == 3)
            wx = - (c0 * vel(i  ,j,k,2)
                  + c1 * vel(i-1,j,k,2)
                  + c2 * vel(i-2,j,k,2)) * idx;
#endif
        }
        else if (!flag.isConnected(-1,0,0))
        {
            // Covered cell to the left, go fish right
            ux = (c0 * vel(i  ,j,k,0)
                + c1 * vel(i+1,j,k,0)
                + c2 * vel(i+2,j,k,0)) * idx;
            vx = (c0 * vel(i  ,j,k,1)
                + c1 * vel(i+1,j,k,1)
                + c2 * vel(i+2,j,k,1)) * idx;
#if (AMREX_SPACEDIM == 3)
            wx = (c0 * vel(i  ,j,k,2)
                + c1 * vel(i+1,j,k,2)
                + c2 * vel(i+2,j,k,2)) * idx;
#endif
        }
        else
        {
            // No covered cells right or left, use standard stencil
            ux = amrex::Real(0.5) * (vel(i+1,j,k,0) - vel(i-1,j,k,0)) * idx;
            vx = amrex::Real(0.5) * (vel(i+1,j,k,1) - vel(i-1,j,k,1)) * idx;
#if (AMREX_SPACEDIM == 3)
            wx = amrex::Real(0.5) * (vel(i+1,j,k,2) - vel(i-1,j,k,2)) * idx;
#endif
        }

        // Do the same in y-direction
        if (!flag.isConnected(0, 1,0))
        {
            uy = - (c0 * vel(i,j  ,k,0)
                  + c1 * vel(i,j-1,k,0)
                  + c2 * vel(i,j-2,k,0)) * idy;
            vy = - (c0 * vel(i,j  ,k,1)
                  + c1 * vel(i,j-1,k,1)
                  + c2 * vel(i,j-2,k,1)) * idy;
#if (AMREX_SPACEDIM == 3)
            wy = - (c0 * vel(i,j  ,k,2)
                  + c1 * vel(i,j-1,k,2)
                  + c2 * vel(i,j-2,k,2)) * idy;
#endif
        }
        else if (!flag.isConnected(0,-1,0))
        {
            uy = (c0 * vel(i,j  ,k,0)
                + c1 * vel(i,j+1,k,0)
                + c2 * vel(i,j+2,k,0)) * idy;
            vy = (c0 * vel(i,j  ,k,1)
                + c1 * vel(i,j+1,k,1)
                + c2 * vel(i,j+2,k,1)) * idy;
#if (AMREX_SPACEDIM == 3)
            wy = (c0 * vel(i,j  ,k,2)
                + c1 * vel(i,j+1,k,2)
                + c2 * vel(i,j+2,k,2)) * idy;
#endif
        }
        else
        {
            uy = amrex::Real(0.5) * (vel(i,j+1,k,0) - vel(i,j-1,k,0)) * idy;
            vy = amrex::Real(0.5) * (vel(i,j+1,k,1) - vel(i,j-1,k,1)) * idy;
#if (AMREX_SPACEDIM == 3)
            wy = amrex::Real(0.5) * (vel(i,j+1,k,2) - vel(i,j-1,k,2)) * idy;
#endif
        }
#if (AMREX_SPACEDIM == 3)
        // Do the same in z-direction
        if (!flag.isConnected(0,0, 1))
        {
            uz = - (c0 * vel(i,j,k  ,0)
                  + c1 * vel(i,j,k-1,0)
                  + c2 * vel(i,j,k-2,0)) * idz;
            vz = - (c0 * vel(i,j,k  ,1)
                  + c1 * vel(i,j,k-1,1)
                  + c2 * vel(i,j,k-2,1)) * idz;
            wz = - (c0 * vel(i,j,k  ,2)
                  + c1 * vel(i,j,k-1,2)
                  + c2 * vel(i,j,k-2,2)) * idz;
        }
        else if (!flag.isConnected(0,0,-1))
        {
            uz = (c0 * vel(i,j,k  ,0)
                + c1 * vel(i,j,k+1,0)
                + c2 * vel(i,j,k+2,0)) * idz;
            vz = (c0 * vel(i,j,k  ,1)
                + c1 * vel(i,j,k+1,1)
                + c2 * vel(i,j,k+2,1)) * idz;
            wz = (c0 * vel(i,j,k  ,2)
                + c1 * vel(i,j,k+1,2)
                + c2 * vel(i,j,k+2,2)) * idz;
        }
        else
        {
            uz = amrex::Real(0.5) * (vel(i,j,k+1,0) - vel(i,j,k-1,0)) * idz;
            vz = amrex::Real(0.5) * (vel(i,j,k+1,1) - vel(i,j,k-1,1)) * idz;
            wz = amrex::Real(0.5) * (vel(i,j,k+1,2) - vel(i,j,k-1,2)) * idz;
        }

        return std::sqrt(amrex::Real(2.0) * ux*ux + amrex::Real(2.0) * vy*vy + amrex::Real(2.0) * wz*wz
                         + (uy+vx)*(uy+vx) + (vz+wy)*(vz+wy) + (wx+uz)*(wx+uz));
#else
        return std::sqrt(amrex::Real(2.0) * ux*ux + amrex::Real(2.0) * vy*vy
                         + (uy+vx)*(uy+vx));
#endif
    } else {
        return 0.0;
    }
}
#endif

#endif
