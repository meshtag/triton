add_triton_library(TritonIMToLLVM
    IMOperandResidencyLayout.cpp
    IMTileBoundary.cpp
    RewriteIMLayout.cpp
    TargetInfo.cpp
    TritonIMToLLVM.cpp

    DEPENDS
    TritonIMConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUToLLVM
    TritonGPUTransforms
)
