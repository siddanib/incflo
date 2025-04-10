#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>

#include <AMReX_ParmParse.H>

#include <algorithm>
#include <incflo.H>

using namespace amrex;

/********************************************************************************
 *                                                                              *
 * Function to create a annular cylinder EB.                                     *
 *                                                                              *
 ********************************************************************************/
void incflo::make_eb_impeller()
{
    // Initialise parameters
    int direction = 2;
    Real radius1 = 0.5;
    Real radius2 = 0.5;
    Real thickness = 0.0;
    Vector<Real> centervec1(3);
    Vector<Real> centervec2(3);

    // Get information from inputs file.                               *
    ParmParse pp("impeller");

    pp.query("direction", direction);
    pp.query("radius1", radius1);
    pp.query("radius2", radius2);
    if (radius1 <= radius2) {
       amrex::Abort("Impeller radius1 needs to be greater than radius2");
    }
    pp.query("thickness",thickness);
    pp.getarr("center1", centervec1, 0, 3);
    pp.getarr("center2", centervec2, 0, 3);
    Array<Real, 3> center1 = {centervec1[0], centervec1[1], centervec1[2]};
    Array<Real, 3> center2 = {centervec2[0], centervec2[1], centervec2[2]};

    // m_eb_flow related
    m_eb_flow.enabled = true;
    m_eb_flow.has_rotation = true;
    m_eb_flow.rotation_center.resize(AMREX_SPACEDIM);
    m_eb_flow.rotation_center = centervec2;
    m_eb_flow.omega_unit_vec.resize(AMREX_SPACEDIM);
    for (int i=0; i<AMREX_SPACEDIM;i++) {
         if (i == direction) {
            m_eb_flow.omega_unit_vec[i] = Real(1.0);
         }
         else {
            m_eb_flow.omega_unit_vec[i] = Real(0.0);
         }
    }
    m_eb_flow.rotation_max_r = Real(0.5)*(radius1 + radius2);

    // Compute distance between cylinder centres
    Real offset = 0.0;
    for(int i = 0; i < 3; i++)
        offset += pow(center1[i] - center2[i], 2);
    offset = sqrt(offset);

    // Print info about cylinders
    amrex::Print() << " CYLINDER 1" << std::endl;
    amrex::Print() << " Direction:       " << direction << std::endl;
    amrex::Print() << " Radius:    " << radius1 << std::endl;
    amrex::Print() << " Center:    "
                   << center1[0] << ", " << center1[1] << ", " << center1[2] << std::endl;

    amrex::Print() << " CYLINDER 2" << std::endl;
    amrex::Print() << " Direction:       " << direction << std::endl;
    amrex::Print() << " Radius:    " << radius2 << std::endl;
    amrex::Print() << " Height:    " << thickness << std::endl;
    amrex::Print() << " Center:    "
                   << center2[0] << ", " << center2[1] << ", " << center2[2] << std::endl;

    amrex::Print() << "\n Offset:          " << offset << std::endl;

    // Build the implicit function from the two cylinders
    EB2::CylinderIF cyl1(radius1, direction, center1,false);
    EB2::CylinderIF cyl2 = (thickness > Real(0.0)) ?
                    EB2::CylinderIF(radius2, thickness, direction, center2,true):
                    EB2::CylinderIF(radius2, direction, center2,true);
    auto twocylinders = EB2::makeComplement(EB2::makeIntersection(cyl1,cyl2));

    // Generate GeometryShop
    auto gshop = EB2::makeShop(twocylinders);

    // Build index space
    int max_level_here = 0;
    int max_coarsening_level = 100;
    EB2::Build(gshop, geom.back(), max_level_here, max_level_here + max_coarsening_level);
}
