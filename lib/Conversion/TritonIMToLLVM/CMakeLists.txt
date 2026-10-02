add_triton_library(TritonIMToLLVM
    IMOperandResidencyLayout.cpp
    IMLaneFold.cpp
    IMLoadCluster.cpp
    IMDccTileMac.cpp
    IMReluOpcode.cpp
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
