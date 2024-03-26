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
             amrex::GpuArray<amrex::GpuArray<int,2>,AMREX_SPACEDIM> bc_type,
             amrex::GpuArray<amrex::GpuArray<amrex::Real,2*AMREX_SPACEDIM>,AMREX_SPACEDIM> bc_velocity) noexcept
{
    using namespace amrex;

#if (AMREX_SPACEDIM == 3)
    Real uxpp = (vel(i,j,k,0) - vel(i-1,j,k,0)) * idx;
    Real uxpm = (vel(i,j-1,k,0) - vel(i-1,j-1,k,0)) * idx;
    Real uxmp = (vel(i,j,k-1,0) - vel(i-1,j,k-1,0)) * idx;
    Real uxmm = (vel(i,j-1,k-1,0) - vel(i-1,j-1,k-1,0)) * idx;
    Real vxpp = (vel(i,j,k,1) - vel(i-1,j,k,1)) * idx;
    Real vxpm = (vel(i,j-1,k,1) - vel(i-1,j-1,k,1)) * idx;
    Real vxmp = (vel(i,j,k-1,1) - vel(i-1,j,k-1,1)) * idx;
    Real vxmm = (vel(i,j-1,k-1,1) - vel(i-1,j-1,k-1,1)) * idx;
    Real wxpp = (vel(i,j,k,2) - vel(i-1,j,k,2)) * idx;
    Real wxpm = (vel(i,j-1,k,2) - vel(i-1,j-1,k,2)) * idx;
    Real wxmp = (vel(i,j,k-1,2) - vel(i-1,j,k-1,2)) * idx;
    Real wxmm = (vel(i,j-1,k-1,2) - vel(i-1,j-1,k-1,2)) * idx;

    Real uypp = (vel(i,j,k,0) - vel(i,j-1,k,0)) * idy;
    Real uypm = (vel(i-1,j,k,0) - vel(i-1,j-1,k,0)) * idy;
    Real uymp = (vel(i,j,k-1,0) - vel(i,j-1,k-1,0)) * idy;
    Real uymm = (vel(i-1,j,k-1,0) - vel(i-1,j-1,k-1,0)) * idy;
    Real vypp = (vel(i,j,k,1) - vel(i,j-1,k,1)) * idy;
    Real vypm = (vel(i-1,j,k,1) - vel(i-1,j-1,k,1)) * idy;
    Real vymp = (vel(i,j,k-1,1) - vel(i,j-1,k-1,1)) * idy;
    Real vymm = (vel(i-1,j,k-1,1) - vel(i-1,j-1,k-1,1)) * idy;
    Real wypp = (vel(i,j,k,2) - vel(i,j-1,k,2)) * idy;
    Real wypm = (vel(i-1,j,k,2) - vel(i-1,j-1,k,2)) * idy;
    Real wymp = (vel(i,j,k-1,2) - vel(i,j-1,k-1,2)) * idy;
    Real wymm = (vel(i-1,j,k-1,2) - vel(i-1,j-1,k-1,2)) * idy;

    Real uzpp = (vel(i,j,k,0) - vel(i,j,k-1,0)) * idz;
    Real uzpm = (vel(i-1,j,k,0) - vel(i-1,j,k-1,0)) * idz;
    Real uzmp = (vel(i,j-1,k,0) - vel(i,j-1,k-1,0)) * idz;
    Real uzmm = (vel(i-1,j-1,k,0) - vel(i-1,j-1,k-1,0)) * idz;
    Real vzpp = (vel(i,j,k,1) - vel(i,j,k-1,1)) * idz;
    Real vzpm = (vel(i-1,j,k,1) - vel(i-1,j,k-1,1)) * idz;
    Real vzmp = (vel(i,j-1,k,1) - vel(i,j-1,k-1,1)) * idz;
    Real vzmm = (vel(i-1,j-1,k,1) - vel(i-1,j-1,k-1,1)) * idz;
    Real wzpp = (vel(i,j,k,2) - vel(i,j,k-1,2)) * idz;
    Real wzpm = (vel(i-1,j,k,2) - vel(i-1,j,k-1,2)) * idz;
    Real wzmp = (vel(i,j-1,k,2) - vel(i,j-1,k-1,2)) * idz;
    Real wzmm = (vel(i-1,j-1,k,2) - vel(i-1,j-1,k-1,2)) * idz;

    // Applying Boundary Conditions
    int idim, vdim;
    if (i == dlo.x) {
        idim = 0;
        vdim = 0;
        if (bc_type[idim][0] == 2) {
            uxpp = Real(2.0) * (vel(i,j,k,0) - bc_velocity[idim][vdim]) * idx;
            uxpm = Real(2.0) * (vel(i,j-1,k,0) - bc_velocity[idim][vdim]) * idx;
            uxmp = Real(2.0) * (vel(i,j,k-1,0) - bc_velocity[idim][vdim]) * idx;
            uxmm = Real(2.0) * (vel(i,j-1,k-1,0) - bc_velocity[idim][vdim]) * idx;

            vxpp = Real(2.0) * (vel(i,j,k,1) - bc_velocity[idim][vdim+1]) * idx;
            vxpm = Real(2.0) * (vel(i,j-1,k,1) - bc_velocity[idim][vdim+1]) * idx;
            vxmp = Real(2.0) * (vel(i,j,k-1,1) - bc_velocity[idim][vdim+1]) * idx;
            vxmm = Real(2.0) * (vel(i,j-1,k-1,1) - bc_velocity[idim][vdim+1]) * idx;

            wxpp = Real(2.0) * (vel(i,j,k,2) - bc_velocity[idim][vdim+2]) * idx;
            wxpm = Real(2.0) * (vel(i,j-1,k,2) - bc_velocity[idim][vdim+2]) * idx;
            wxmp = Real(2.0) * (vel(i,j,k-1,2) - bc_velocity[idim][vdim+2]) * idx;
            wxmm = Real(2.0) * (vel(i,j-1,k-1,2) - bc_velocity[idim][vdim+2]) * idx;
        }
        else if (bc_type[idim][0] == 1) {
            uxpp = Real(2.0) * (vel(i,j,k,0) - bc_velocity[idim][vdim]) * idx;
            uxpm = Real(2.0) * (vel(i,j-1,k,0) - bc_velocity[idim][vdim]) * idx;
            uxmp = Real(2.0) * (vel(i,j,k-1,0) - bc_velocity[idim][vdim]) * idx;
            uxmm = Real(2.0) * (vel(i,j-1,k-1,0) - bc_velocity[idim][vdim]) * idx;

            vxpp = Real(0.0);
            vxpm = Real(0.0);
            vxmp = Real(0.0);
            vxmm = Real(0.0);

            wxpp = Real(0.0);
            wxpm = Real(0.0);
            wxmp = Real(0.0);
            wxmm = Real(0.0);

        }
    }

    if (i == dhi.x + 1) {
        idim = 0;
        vdim = AMREX_SPACEDIM;
        if (bc_type[idim][1] == 2) {
            uxpp = Real(-2.0) * (vel(i-1,j,k,0) - bc_velocity[idim][vdim]) * idx;
            uxpm = Real(-2.0) * (vel(i-1,j-1,k,0) - bc_velocity[idim][vdim]) * idx;
            uxmp = Real(-2.0) * (vel(i-1,j,k-1,0) - bc_velocity[idim][vdim]) * idx;
            uxmm = Real(-2.0) * (vel(i-1,j-1,k-1,0) - bc_velocity[idim][vdim]) * idx;

            vxpp = Real(-2.0) * (vel(i-1,j,k,1) - bc_velocity[idim][vdim+1]) * idx;
            vxpm = Real(-2.0) * (vel(i-1,j-1,k,1) - bc_velocity[idim][vdim+1]) * idx;
            vxmp = Real(-2.0) * (vel(i-1,j,k-1,1) - bc_velocity[idim][vdim+1]) * idx;
            vxmm = Real(-2.0) * (vel(i-1,j-1,k-1,1) - bc_velocity[idim][vdim+1]) * idx;

            wxpp = Real(-2.0) * (vel(i-1,j,k,2) - bc_velocity[idim][vdim+2]) * idx;
            wxpm = Real(-2.0) * (vel(i-1,j-1,k,2) - bc_velocity[idim][vdim+2]) * idx;
            wxmp = Real(-2.0) * (vel(i-1,j,k-1,2) - bc_velocity[idim][vdim+2]) * idx;
            wxmm = Real(-2.0) * (vel(i-1,j-1,k-1,2) - bc_velocity[idim][vdim+2]) * idx;

        }
        else if (bc_type[idim][1] == 1) {
            uxpp = Real(-2.0) * (vel(i-1,j,k,0) - bc_velocity[idim][vdim]) * idx;
            uxpm = Real(-2.0) * (vel(i-1,j-1,k,0) - bc_velocity[idim][vdim]) * idx;
            uxmp = Real(-2.0) * (vel(i-1,j,k-1,0) - bc_velocity[idim][vdim]) * idx;
            uxmm = Real(-2.0) * (vel(i-1,j-1,k-1,0) - bc_velocity[idim][vdim]) * idx;

            vxpp = Real(0.0);
            vxpm = Real(0.0);
            vxmp = Real(0.0);
            vxmm = Real(0.0);

            wxpp = Real(0.0);
            wxpm = Real(0.0);
            wxmp = Real(0.0);
            wxmm = Real(0.0);
        }
    }

    if (j == dlo.y) {
        idim = 1;
        vdim = 0;
        if (bc_type[idim][0] == 2) {
            uypp = Real(2.0) * (vel(i,j,k,0) - bc_velocity[idim][vdim]) * idy;
            uypm = Real(2.0) * (vel(i-1,j,k,0) - bc_velocity[idim][vdim]) * idy;
            uymp = Real(2.0) * (vel(i,j,k-1,0) - bc_velocity[idim][vdim]) * idy;
            uymm = Real(2.0) * (vel(i-1,j,k-1,0) - bc_velocity[idim][vdim]) * idy;

            vypp = Real(2.0) * (vel(i,j,k,1) - bc_velocity[idim][vdim+1]) * idy;
            vypm = Real(2.0) * (vel(i-1,j,k,1) - bc_velocity[idim][vdim+1]) * idy;
            vymp = Real(2.0) * (vel(i,j,k-1,1) - bc_velocity[idim][vdim+1]) * idy;
            vymm = Real(2.0) * (vel(i-1,j,k-1,1) - bc_velocity[idim][vdim+1]) * idy;

            wypp = Real(2.0) * (vel(i,j,k,2) - bc_velocity[idim][vdim+2]) * idy;
            wypm = Real(2.0) * (vel(i-1,j,k,2) - bc_velocity[idim][vdim+2]) * idy;
            wymp = Real(2.0) * (vel(i,j,k-1,2) - bc_velocity[idim][vdim+2]) * idy;
            wymm = Real(2.0) * (vel(i-1,j,k-1,2) - bc_velocity[idim][vdim+2]) * idy;

        }
        else if (bc_type[idim][0] == 1){
            uypp = Real(0.0);
            uypm = Real(0.0);
            uymp = Real(0.0);
            uymm = Real(0.0);

            vypp = Real(2.0) * (vel(i,j,k,1) - bc_velocity[idim][vdim+1]) * idy;
            vypm = Real(2.0) * (vel(i-1,j,k,1) - bc_velocity[idim][vdim+1]) * idy;
            vymp = Real(2.0) * (vel(i,j,k-1,1) - bc_velocity[idim][vdim+1]) * idy;
            vymm = Real(2.0) * (vel(i-1,j,k-1,1) - bc_velocity[idim][vdim+1]) * idy;

            wypp = Real(0.0);
            wypm = Real(0.0);
            wymp = Real(0.0);
            wymm = Real(0.0);
        }
    }

    if (j == dhi.y + 1) {
        idim = 1;
        vdim = AMREX_SPACEDIM;
        if (bc_type[idim][1] == 2) {
            uypp = Real(-2.0) * (vel(i,j-1,k,0) - bc_velocity[idim][vdim]) * idy;
            uypm = Real(-2.0) * (vel(i-1,j-1,k,0) - bc_velocity[idim][vdim]) * idy;
            uymp = Real(-2.0) * (vel(i,j-1,k-1,0) - bc_velocity[idim][vdim]) * idy;
            uymm = Real(-2.0) * (vel(i-1,j-1,k-1,0) - bc_velocity[idim][vdim]) * idy;

            vypp = Real(-2.0) * (vel(i,j-1,k,1) - bc_velocity[idim][vdim+1]) * idy;
            vypm = Real(-2.0) * (vel(i-1,j-1,k,1) - bc_velocity[idim][vdim+1]) * idy;
            vymp = Real(-2.0) * (vel(i,j-1,k-1,1) - bc_velocity[idim][vdim+1]) * idy;
            vymm = Real(-2.0) * (vel(i-1,j-1,k-1,1) - bc_velocity[idim][vdim+1]) * idy;

            wypp = Real(-2.0) * (vel(i,j-1,k,2) - bc_velocity[idim][vdim+2]) * idy;
            wypm = Real(-2.0) * (vel(i-1,j-1,k,2) - bc_velocity[idim][vdim+2]) * idy;
            wymp = Real(-2.0) * (vel(i,j-1,k-1,2) - bc_velocity[idim][vdim+2]) * idy;
            wymm = Real(-2.0) * (vel(i-1,j-1,k-1,2) - bc_velocity[idim][vdim+2]) * idy;

        }
        else if (bc_type[idim][1] == 1){
            uypp = Real(0.0);
            uypm = Real(0.0);
            uymp = Real(0.0);
            uymm = Real(0.0);

            vypp = Real(-2.0) * (vel(i,j-1,k,1) - bc_velocity[idim][vdim+1]) * idy;
            vypm = Real(-2.0) * (vel(i-1,j-1,k,1) - bc_velocity[idim][vdim+1]) * idy;
            vymp = Real(-2.0) * (vel(i,j-1,k-1,1) - bc_velocity[idim][vdim+1]) * idy;
            vymm = Real(-2.0) * (vel(i-1,j-1,k-1,1) - bc_velocity[idim][vdim+1]) * idy;

            wypp = Real(0.0);
            wypm = Real(0.0);
            wymp = Real(0.0);
            wymm = Real(0.0);
        }
    }

    if (k == dlo.z) {
        idim = 2;
        vdim = 0;
        if (bc_type[idim][0] == 2) {
            uzpp = Real(2.0) * (vel(i,j,k,0) - bc_velocity[idim][vdim]) * idz;
            uzpm = Real(2.0) * (vel(i-1,j,k,0) - bc_velocity[idim][vdim]) * idz;
            uzmp = Real(2.0) * (vel(i,j-1,k,0) - bc_velocity[idim][vdim]) * idz;
            uzmm = Real(2.0) * (vel(i-1,j-1,k,0) - bc_velocity[idim][vdim]) * idz;

            vzpp = Real(2.0) * (vel(i,j,k,1) - bc_velocity[idim][vdim+1]) * idz;
            vzpm = Real(2.0) * (vel(i-1,j,k,1) - bc_velocity[idim][vdim+1]) * idz;
            vzmp = Real(2.0) * (vel(i,j-1,k,1) - bc_velocity[idim][vdim+1]) * idz;
            vzmm = Real(2.0) * (vel(i-1,j-1,k,1) - bc_velocity[idim][vdim+1]) * idz;

            wzpp = Real(2.0) * (vel(i,j,k,2) - bc_velocity[idim][vdim+2]) * idz;
            wzpm = Real(2.0) * (vel(i-1,j,k,2) - bc_velocity[idim][vdim+2]) * idz;
            wzmp = Real(2.0) * (vel(i,j-1,k,2) - bc_velocity[idim][vdim+2]) * idz;
            wzmm = Real(2.0) * (vel(i-1,j-1,k,2) - bc_velocity[idim][vdim+2]) * idz;
        }
        else if (bc_type[idim][0] == 1){
            uzpp = Real(0.0);
            uzpm = Real(0.0);
            uzmp = Real(0.0);
            uzmm = Real(0.0);

            vzpp = Real(0.0);
            vzpm = Real(0.0);
            vzmp = Real(0.0);
            vzmm = Real(0.0);

            wzpp = Real(2.0) * (vel(i,j,k,2) - bc_velocity[idim][vdim+2]) * idz;
            wzpm = Real(2.0) * (vel(i-1,j,k,2) - bc_velocity[idim][vdim+2]) * idz;
            wzmp = Real(2.0) * (vel(i,j-1,k,2) - bc_velocity[idim][vdim+2]) * idz;
            wzmm = Real(2.0) * (vel(i-1,j-1,k,2) - bc_velocity[idim][vdim+2]) * idz;
        }
    }

    if (k == dhi.z + 1) {
        idim = 2;
        vdim = AMREX_SPACEDIM;
        if (bc_type[idim][1] == 2) {
            uzpp = Real(-2.0) * (vel(i,j,k-1,0) - bc_velocity[idim][vdim]) * idz;
            uzpm = Real(-2.0) * (vel(i-1,j,k-1,0) - bc_velocity[idim][vdim]) * idz;
            uzmp = Real(-2.0) * (vel(i,j-1,k-1,0) - bc_velocity[idim][vdim]) * idz;
            uzmm = Real(-2.0) * (vel(i-1,j-1,k-1,0) - bc_velocity[idim][vdim]) * idz;

            vzpp = Real(-2.0) * (vel(i,j,k-1,1) - bc_velocity[idim][vdim+1]) * idz;
            vzpm = Real(-2.0) * (vel(i-1,j,k-1,1) - bc_velocity[idim][vdim+1]) * idz;
            vzmp = Real(-2.0) * (vel(i,j-1,k-1,1) - bc_velocity[idim][vdim+1]) * idz;
            vzmm = Real(-2.0) * (vel(i-1,j-1,k-1,1) - bc_velocity[idim][vdim+1]) * idz;

            wzpp = Real(-2.0) * (vel(i,j,k-1,2) - bc_velocity[idim][vdim+2]) * idz;
            wzpm = Real(-2.0) * (vel(i-1,j,k-1,2) - bc_velocity[idim][vdim+2]) * idz;
            wzmp = Real(-2.0) * (vel(i,j-1,k-1,2) - bc_velocity[idim][vdim+2]) * idz;
            wzmm = Real(-2.0) * (vel(i-1,j-1,k-1,2) - bc_velocity[idim][vdim+2]) * idz;
        }
        else if (bc_type[idim][1] == 1){
            uzpp = Real(0.0);
            uzpm = Real(0.0);
            uzmp = Real(0.0);
            uzmm = Real(0.0);

            vzpp = Real(0.0);
            vzpm = Real(0.0);
            vzmp = Real(0.0);
            vzmm = Real(0.0);

            wzpp = Real(-2.0) * (vel(i,j,k-1,2) - bc_velocity[idim][vdim+2]) * idz;
            wzpm = Real(-2.0) * (vel(i-1,j,k-1,2) - bc_velocity[idim][vdim+2]) * idz;
            wzmp = Real(-2.0) * (vel(i,j-1,k-1,2) - bc_velocity[idim][vdim+2]) * idz;
            wzmm = Real(-2.0) * (vel(i-1,j-1,k-1,2) - bc_velocity[idim][vdim+2]) * idz;
        }
    }
    // Correction for all boundary cells
    Real dnmtr_x = Real(4.0);
    Real dnmtr_y = Real(4.0);
    Real dnmtr_z = Real(4.0);
    if (i == dlo.x) {
        idim = 0;
        if ((bc_type[idim][0] == 2) or (bc_type[idim][0] == 1)) {

            uymm = Real(0.0); uypm = Real(0.0); uzmm = Real(0.0); uzpm = Real(0.0);
            vymm = Real(0.0); vypm = Real(0.0); vzmm = Real(0.0); vzpm = Real(0.0);
            wymm = Real(0.0); wypm = Real(0.0); wzmm = Real(0.0); wzpm = Real(0.0);
            dnmtr_y -= Real(2.0);
            dnmtr_z -= Real(2.0);

        }
    }

    if (i == dhi.x + 1) {
        idim = 0;
        if ((bc_type[idim][1] == 2) or (bc_type[idim][1] == 1)) {

            uymp = Real(0.0); uypp = Real(0.0); uzmp = Real(0.0); uzpp = Real(0.0);
            vymp = Real(0.0); vypp = Real(0.0); vzmp = Real(0.0); vzpp = Real(0.0);
            wymp = Real(0.0); wypp = Real(0.0); wzmp = Real(0.0); wzpp = Real(0.0);
            dnmtr_y -= Real(2.0);
            dnmtr_z -= Real(2.0);

        }
    }

    if (j == dlo.y) {
        idim = 1;
        if ((bc_type[idim][0] == 2) or (bc_type[idim][0] == 1)) {

            uxmm = Real(0.0); uxpm = Real(0.0); uzmm = Real(0.0); uzmp = Real(0.0);
            vxmm = Real(0.0); vxpm = Real(0.0); vzmm = Real(0.0); vzmp = Real(0.0);
            wxmm = Real(0.0); wxpm = Real(0.0); wzmm = Real(0.0); wzmp = Real(0.0);
            dnmtr_x -= Real(2.0);
            dnmtr_z -= Real(2.0);

        }
    }

    if (j == dhi.y + 1) {
        idim = 1;
        if ((bc_type[idim][1] == 2) or (bc_type[idim][1] == 1)) {

            uxmp = Real(0.0); uxpp = Real(0.0); uzpp = Real(0.0); uzpm = Real(0.0);
            vxmp = Real(0.0); vxpp = Real(0.0); vzpp = Real(0.0); vzpm = Real(0.0);
            wxmp = Real(0.0); wxpp = Real(0.0); wzpp = Real(0.0); wzpm = Real(0.0);
            dnmtr_x -= Real(2.0);
            dnmtr_z -= Real(2.0);

        }
    }

    if (k == dlo.z) {
        idim = 2;
        if ((bc_type[idim][0] == 2) or (bc_type[idim][0] == 1)) {

            uxmm = Real(0.0); uxmp = Real(0.0); uymm = Real(0.0); uymp = Real(0.0);
            vxmm = Real(0.0); vxmp = Real(0.0); vymm = Real(0.0); vymp = Real(0.0);
            wxmm = Real(0.0); wxmp = Real(0.0); wymm = Real(0.0); wymp = Real(0.0);
            dnmtr_x -= Real(2.0);
            dnmtr_y -= Real(2.0);

        }
    }

    if (k == dhi.z + 1) {
        idim = 2;
        if ((bc_type[idim][1] == 2) or (bc_type[idim][1] == 1)) {

            uxpm = Real(0.0); uxpp = Real(0.0); uypp = Real(0.0); uypm = Real(0.0);
            vxpm = Real(0.0); vxpp = Real(0.0); vypp = Real(0.0); vypm = Real(0.0);
            wxpm = Real(0.0); wxpp = Real(0.0); wypp = Real(0.0); wypm = Real(0.0);
            dnmtr_x -= Real(2.0);
            dnmtr_y -= Real(2.0);

        }
    }

    AMREX_ALWAYS_ASSERT(dnmtr_x >= Real(0.0));
    AMREX_ALWAYS_ASSERT(dnmtr_y >= Real(0.0));
    AMREX_ALWAYS_ASSERT(dnmtr_z >= Real(0.0));
    // Remove double-counting effect
    if (dnmtr_x == Real(0.0)) dnmtr_x = Real(1.0);
    if (dnmtr_y == Real(0.0)) dnmtr_y = Real(1.0);
    if (dnmtr_z == Real(0.0)) dnmtr_z = Real(1.0);
    // This is only nearest point approximation for domain corner nodes
    Real ux = (uxpp + uxpm + uxmp + uxmm)/dnmtr_x;
    Real uy = (uypp + uypm + uymp + uymm)/dnmtr_y;
    Real uz = (uzpp + uzpm + uzmp + uzmm)/dnmtr_z;

    Real vx = (vxpp + vxpm + vxmp + vxmm)/dnmtr_x;
    Real vy = (vypp + vypm + vymp + vymm)/dnmtr_y;
    Real vz = (vzpp + vzpm + vzmp + vzmm)/dnmtr_z;

    Real wx = (wxpp + wxpm + wxmp + wxmm)/dnmtr_x;
    Real wy = (wypp + wypm + wymp + wymm)/dnmtr_y;
    Real wz = (wzpp + wzpm + wzmp + wzmm)/dnmtr_z;

    return std::sqrt(amrex::Real(2.0) * ux*ux + amrex::Real(2.0) * vy*vy + amrex::Real(2.0) * wz*wz
                     + (uy+vx)*(uy+vx) + (vz+wy)*(vz+wy) + (wx+uz)*(wx+uz));
#else

    Real uxp = (vel(i,j,k,0) - vel(i-1,j,k,0)) * idx;
    Real uxm = (vel(i,j-1,k,0) - vel(i-1,j-1,k,0)) * idx;
    Real vxp = (vel(i,j,k,1) - vel(i-1,j,k,1)) * idx;
    Real vxm = (vel(i,j-1,k,1) - vel(i-1,j-1,k,1)) * idx;

    Real uyp = (vel(i,j,k,0) - vel(i,j-1,k,0)) * idy;
    Real uym = (vel(i-1,j,k,0) - vel(i-1,j-1,k,0)) * idy;
    Real vyp = (vel(i,j,k,1) - vel(i,j-1,k,1)) * idy;
    Real vym = (vel(i-1,j,k,1) - vel(i-1,j-1,k,1)) * idy;

    Real ux = amrex::Real(0.5)*(uxp + uxm);
    Real uy = amrex::Real(0.5)*(uyp + uym);
    Real vx = amrex::Real(0.5)*(vxp + vxm);
    Real vy = amrex::Real(0.5)*(vyp + vym);

    return std::sqrt(amrex::Real(2.0) * ux*ux + amrex::Real(2.0) * vy*vy
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
