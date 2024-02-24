AMREX_HOME = /home/bsiddani/siddanib_github/amrex
AMREX_HYDRO_HOME = /home/bsiddani/siddanib_github/AMReX-Hydro
TOP = ..

EBASE = incflo

DIM = 3

PRECISION  = DOUBLE

TINY_PROFILE = FALSE

DEBUG      = TRUE
DEBUG      = FALSE

COMP	   = gnu

USE_MPI    = TRUE
USE_AMREX_MPMD = TRUE

USE_OMP    = FALSE
USE_CUDA   = FALSE

USE_HYPRE = FALSE
BL_NO_FORT = TRUE

USE_EB = TRUE

USE_CSG = FALSE

include ../src/Make.incflo

ifeq ($(USE_AMREX_MPMD), TRUE)
  DEFINES += -DUSE_AMREX_MPMD
endif
